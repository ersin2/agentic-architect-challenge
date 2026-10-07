"""Data types for Part 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

CATEGORIES = ("Billing", "Technical", "Feedback", "Other")

DRAFT_READY = "draft_ready"  # a verified draft waits for an agent to approve and send it
HUMAN_REVIEW = "human_review"  # a person handles the email; no automatic draft


class Reason(str, Enum):
    """Why an email went to a human. Every human route has at least one reason."""

    # Escalation gate (deterministic, runs before any model call)
    DATA_LOSS = "data_loss"
    SERVICE_OUTAGE = "service_outage"
    SECURITY_BREACH = "security_breach"
    FREQUENT_CONTACT = "frequent_contact"
    PROMPT_INJECTION = "prompt_injection_suspected"
    # Model-driven escalation (the model may add a reason, never remove one)
    MODEL_FLAGGED_CRITICAL = "model_flagged_critical"
    NOT_ANSWERABLE = "not_answerable_from_kb"
    # Failures: always fail closed
    LLM_UNAVAILABLE = "llm_unavailable"
    TRIAGE_FAILED = "triage_failed"
    DRAFT_FAILED = "draft_failed"
    VERIFICATION_FAILED = "verification_failed"
    INTERNAL_ERROR = "internal_error"


# Which human queue gets the email, by the most serious reason present.
TEAM_BY_REASON = [
    (Reason.SECURITY_BREACH, "security"),
    (Reason.DATA_LOSS, "technical-oncall"),
    (Reason.SERVICE_OUTAGE, "technical-oncall"),
    (Reason.PROMPT_INJECTION, "trust-and-safety"),
    (Reason.MODEL_FLAGGED_CRITICAL, "technical-oncall"),
    (Reason.FREQUENT_CONTACT, "customer-success"),
]


def team_for(reasons: list[Reason]) -> str:
    for reason, team in TEAM_BY_REASON:
        if reason in reasons:
            return team
    return "support-general"


@dataclass(frozen=True)
class Email:
    message_id: str
    sender: str
    subject: str
    body: str
    received_at: datetime  # timezone-aware

    @property
    def customer_id(self) -> str:
        """Customers are identified by their lowercased email address."""
        return self.sender.strip().lower()

    @property
    def text(self) -> str:
        return f"{self.subject}\n\n{self.body}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Email:
        missing = [k for k in ("message_id", "sender", "received_at") if not data.get(k)]
        if missing:
            raise ValueError(f"email record is missing {missing}")
        received = datetime.fromisoformat(str(data["received_at"]).replace("Z", "+00:00"))
        if received.tzinfo is None:
            raise ValueError("received_at must include a timezone")
        return cls(str(data["message_id"]), str(data["sender"]), str(data.get("subject", "")),
                   str(data.get("body", "")), received)


@dataclass
class Decision:
    """The outcome for one email. Written to out/decisions.jsonl."""

    message_id: str
    trace_id: str
    route: str
    reasons: list[str] = field(default_factory=list)
    team: str | None = None
    evidence: dict[str, str] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    draft: str | None = None
    citations: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    llm_calls: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
