"""Triage: the model classifies the email and plans the knowledge-base search.

The model returns JSON. Code validates every field. Categories must come from
a fixed list, so the model cannot invent a new route.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentkit.injection import untrusted_note, wrap_untrusted
from agentkit.jsonout import generate_json
from agentkit.llm import LLMClient, Message

from .models import CATEGORIES, Email

SYSTEM = f"""You triage customer support emails for AcmeSync, a note-taking and sync app.
Return one JSON object with these fields:
- categories: every category that applies (an email can have several):
  Billing = payments, invoices, plans, cancellations, refunds;
  Technical = bugs, errors, crashes, sync, sign-in, performance;
  Feedback = praise, complaints, feature requests;
  Other = anything else.
- critical_issue: true only if the email describes data loss, a service outage, a security
  problem, or a risk of similar seriousness. Otherwise false.
- kb_queries: 1 to 3 short search queries for our help-center knowledge base that would find
  the facts needed to answer the customer.
- summary: one neutral sentence saying what the customer wants.
{untrusted_note("email")}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {"type": "array", "items": {"type": "string", "enum": list(CATEGORIES)}, "minItems": 1},
        "critical_issue": {"type": "boolean"},
        "kb_queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "summary": {"type": "string"},
    },
    "required": ["categories", "critical_issue", "kb_queries", "summary"],
}


@dataclass(frozen=True)
class Triage:
    categories: list[str]
    critical_issue: bool
    kb_queries: list[str]
    summary: str


def validate(data: dict) -> Triage:
    raw = data.get("categories")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'categories' must be a non-empty list")
    categories: list[str] = []
    for item in raw:
        name = str(item).strip().capitalize()
        if name not in CATEGORIES:
            raise ValueError(f"unknown category {item!r}; allowed: {', '.join(CATEGORIES)}")
        if name not in categories:
            categories.append(name)
    if len(categories) > 1 and "Other" in categories:
        categories.remove("Other")
    critical = data.get("critical_issue")
    if not isinstance(critical, bool):
        raise ValueError("'critical_issue' must be true or false")
    queries = [q.strip() for q in data.get("kb_queries") or [] if isinstance(q, str) and q.strip()][:3]
    if not queries:
        raise ValueError("'kb_queries' must contain at least one query")
    return Triage(categories, critical, queries, str(data.get("summary", "")).strip()[:300])


def email_block(email: Email) -> str:
    return wrap_untrusted("email", f"Subject: {email.subject}\n\n{email.body}")


def run_triage(client: LLMClient, email: Email) -> Triage:
    prompt = f"Triage this customer email.\n\n{email_block(email)}"
    return generate_json(client, [Message("user", prompt)], system=SYSTEM, schema=SCHEMA,
                         validate=validate, max_output_tokens=1024)
