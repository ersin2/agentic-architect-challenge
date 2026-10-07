"""The escalation gate: deterministic rules that run BEFORE any model call.

Why code and not the model:
- it cannot be talked out of escalating by text inside the email (prompt injection);
- it gives the same answer every time, and every rule has a unit test;
- it still works when the model provider is down, so the most urgent emails
  are routed correctly even during an outage.

Trade-off we accept on purpose: keyword rules favour recall over precision.
"No data loss happened" still escalates (we do not try to handle negation),
because sending one extra email to a human is cheap and missing a breach is not.
Paraphrases the rules miss can still be caught later by the model's
`critical_issue` flag, which can only add an escalation, never remove one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentkit.injection import find_injection_signals, normalize

from .models import Email, Reason

FREQUENT_CONTACT_LIMIT = 3  # "more than 3 times in 7 days" -> the 4th contact escalates

_WORDS_DATA = r"(data|files?|notes?|notebooks?|documents?|records?|projects?|work)"

KEYWORD_RULES: dict[Reason, list[re.Pattern[str]]] = {
    Reason.DATA_LOSS: [re.compile(p) for p in (
        r"\bdata ?loss\b",
        rf"\blost (all |some |most |half )?(of )?(my |our |the )?{_WORDS_DATA}\b",
        rf"\b{_WORDS_DATA} (have |has |are |is |were |was )?(been |got )?(deleted|wiped|erased|gone|missing|"
        r"disappeared|vanished|corrupted|lost)\b",
    )],
    Reason.SERVICE_OUTAGE: [re.compile(p) for p in (
        r"\boutages?\b",
        r"\b(service|site|website|app|server|servers|platform|api|system|everything|acmesync) "
        r"(is |are |was |went |seems |appears )?(to be )?(completely )?(down|offline|unreachable|unavailable|"
        r"not responding)\b",
        r"\bdown for (everyone|all|hours|our|the whole)\b",
        r"\b(50[0234]) errors?\b|\berrors? 50[0234]\b",
    )],
    Reason.SECURITY_BREACH: [re.compile(p) for p in (
        r"\bbreach(ed|es)?\b",
        r"\bhack(ed|er|ers|ing)?\b",
        r"\bcompromised\b",
        r"\b(account|password|credentials?|data|card) (was |were |has been |have been |got )?(stolen|leaked)\b",
        r"\bunauthori[sz]ed (access|login|log-in|sign-?in|charges?|transactions?|activity)\b",
        r"\b(someone|somebody) (else )?(has )?(logged|signed) (in|into)\b",
        r"\bsuspicious (login|log-in|sign-?in|activity)\b",
        r"\b(phishing|ransomware|malware)\b",
    )],
}


@dataclass(frozen=True)
class GateResult:
    reasons: list[Reason]
    evidence: dict[str, str] = field(default_factory=dict)  # reason -> what triggered it
    contacts_7d: int = 0

    @property
    def escalate(self) -> bool:
        return bool(self.reasons)


def check_escalation(email: Email, contacts_7d: int) -> GateResult:
    """Decide, without any model, whether this email must go to a human.

    `contacts_7d` counts this email too (see ContactHistory.count_in_window).
    """
    text = normalize(email.text)
    reasons: list[Reason] = []
    evidence: dict[str, str] = {}
    for reason, patterns in KEYWORD_RULES.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                reasons.append(reason)
                evidence[reason.value] = match.group(0)
                break
    if contacts_7d > FREQUENT_CONTACT_LIMIT:
        reasons.append(Reason.FREQUENT_CONTACT)
        evidence[Reason.FREQUENT_CONTACT.value] = f"{contacts_7d} contacts in 7 days"
    signals = find_injection_signals(email.text)
    if signals:
        reasons.append(Reason.PROMPT_INJECTION)
        evidence[Reason.PROMPT_INJECTION.value] = ", ".join(signals)
    return GateResult(reasons, evidence, contacts_7d)
