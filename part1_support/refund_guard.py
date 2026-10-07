"""Refund guard: the model may not state refund facts in its own words.

How it works:
1. The drafting prompt says: to give a refund rule, write a placeholder like
   {{policy:R1}}. The model chooses WHICH rule applies; it never writes the rule.
2. This module checks the draft in code:
   - every placeholder names a real clause, and that clause was in the retrieved context;
   - no sentence outside a placeholder talks about refunds AND makes a claim
     (a number, a duration, a percentage, "full", "eligible", "will refund", ...).
3. render_policy_slots() then replaces each placeholder with the exact clause text.

So refund facts in a sent reply can only come from the policy document,
character for character. If any check fails, the pipeline asks the model to
revise once, and if it fails again the email goes to a human (fail closed).

Known limits (stated in the docs): the claim detector is a keyword heuristic.
It is strict on purpose, so it can reject harmless sentences (more human work),
and a refund promise written without any refund word can slip past it.
The model can also pick a real but wrong clause; the text is still accurate,
but it may not fit the customer's case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .kb import POLICY_DOC

SLOT = re.compile(r"\{\{\s*policy\s*:\s*([A-Za-z0-9_#-]+)\s*\}\}")
_SLOT_MARK = "§"  # stands in for a placeholder while sentences are analysed

REFUND_TOPIC = re.compile(
    r"\b(refund\w*|money[- ]back|your money|reimburs\w*|charge-?backs?|prorat\w*|pro-rat\w*|"
    r"credit(ed)? back|credit\w* (to )?(your|the) (card|account)|"
    r"revers\w* (the |this |that |your )?(charge|payment|transaction)s?|repa(y|id|yment)\w*)\b", re.IGNORECASE)

CLAIM_CUES = re.compile(r"""
      \d                                                   # any number: amounts, days, percentages
    | [%$€£]
    | \b(one|two|three|four|five|six|seven|eight|nine|ten|fourteen|thirty|sixty|ninety|hundred)\b
    | \b(full|fully|partial|partially|entire)\b
    | \b(guarantee\w*|eligib\w*|entitled|qualif\w*|approv\w*|automatic\w*|immediate\w*)\b
    | \b(within|business\ days?)\b | \b(few|several|couple\ of)\s+(days?|weeks?)\b
    | \b(will|we'll|you'll|can|could|shall|going\ to)\s+(be\s+)?(\w+\s+)?
         (refund\w*|reimburs\w*|credit\w*|revers\w*|receive|get|issu\w*|process\w*|return\w*|send\w*|give\w*)\b
    | \b(not|non-?)\s*refundable\b | \bno\s+refunds?\b
    | \b(must|required?|requires|need(s)?\ to|ha(ve|s)\ to|ensure|only\ if|as\ long\ as)\b   # conditions = rules too
""", re.IGNORECASE | re.VERBOSE)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class GuardResult:
    violations: list[str] = field(default_factory=list)
    clause_ids: list[str] = field(default_factory=list)  # placeholders used, in order

    @property
    def ok(self) -> bool:
        return not self.violations


def mentions_refund(text: str) -> bool:
    return bool(REFUND_TOPIC.search(text))


def _clause_id(raw: str) -> str:
    """Accept both {{policy:R1}} and {{policy:refund_policy#R1}}."""
    return raw.split("#", 1)[1] if "#" in raw else raw


def check_refund_claims(reply: str, policy: dict[str, str], retrieved_ids: set[str]) -> GuardResult:
    violations: list[str] = []
    clause_ids = [_clause_id(m.group(1)) for m in SLOT.finditer(reply)]

    for cid in clause_ids:
        if cid not in policy:
            violations.append(f"unknown refund policy clause {{{{policy:{cid}}}}}: no such clause exists")
        elif f"{POLICY_DOC}#{cid}" not in retrieved_ids:
            violations.append(f"refund policy clause {cid} was not in the provided knowledge")

    policy_numbers = set(_NUMBER.findall(" ".join(policy.get(c, "") for c in policy
                                                  if f"{POLICY_DOC}#{c}" in retrieved_ids)))
    for sentence in _SENTENCE_SPLIT.split(SLOT.sub(_SLOT_MARK, reply)):
        if not (mentions_refund(sentence) and CLAIM_CUES.search(sentence)):
            continue
        detail = ""
        unsupported = [n for n in _NUMBER.findall(sentence) if n not in policy_numbers]
        if unsupported:
            detail = f" It contains {', '.join(unsupported)}, which no provided policy clause contains."
        quoted = sentence.replace(_SLOT_MARK, "[policy]").strip()
        violations.append(
            f'refund claim written in your own words: "{quoted[:160]}".{detail} '
            "State refund rules only with {{policy:ID}} placeholders.")
    return GuardResult(violations, clause_ids)


def render_policy_slots(reply: str, policy: dict[str, str]) -> str:
    """Replace each placeholder with the exact clause text, in quotes. Call only after a clean check."""
    def quote(match: re.Match[str]) -> str:
        text = policy[_clause_id(match.group(1))]
        period = match.group(2) or ""
        if text.endswith((".", "!", "?")):
            period = ""  # '{{policy:R4}}.' must not become '... confirms it.".'
        return f'"{text}"{period}'

    return _SLOT_WITH_PERIOD.sub(quote, reply)


_SLOT_WITH_PERIOD = re.compile(SLOT.pattern + r"(\.)?")
