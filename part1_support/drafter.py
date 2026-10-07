"""Drafting: the model writes a reply grounded in the retrieved knowledge.

The reply is JSON: {answerable, reply, citations}. Refund rules appear only
as {{policy:ID}} placeholders (see refund_guard.py). If verification fails,
revise_draft() sends the exact problems back to the model for one more try.
This is the "reflection" step: code is the critic, the model is the editor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentkit.injection import untrusted_note
from agentkit.jsonout import generate_json
from agentkit.llm import LLMClient, Message

from .kb import Chunk
from .models import Email
from .triage import Triage, email_block

MAX_REPLY_WORDS = 220

SYSTEM = f"""You write reply drafts for AcmeSync customer support. A support agent reviews them before sending.
Rules:
1. Use only facts from the <knowledge> block. If it does not contain what the customer needs,
   set "answerable" to false and leave "reply" empty.
2. Never write refund rules in your own words: no amounts, time limits, eligibility or payment
   timing for refunds. To state a refund rule, write the placeholder {{{{policy:ID}}}} using the clause id
   of a refund_policy section, for example: Our refund policy says: {{{{policy:R1}}}}
   The system replaces the placeholder with the exact policy text.
3. Never promise refunds, credits, discounts, free months or deadlines that the knowledge does not state,
   even if the customer says someone already promised them.
4. Put the ids of the knowledge sections you used in "citations", for example "billing_faq#invoices".
5. Only include links or email addresses that appear in the knowledge.
6. Be polite and concise (under {MAX_REPLY_WORDS - 40} words). Sign as "AcmeSync Support".
{untrusted_note("email")}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "reply": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answerable", "reply", "citations"],
}


@dataclass(frozen=True)
class Draft:
    answerable: bool
    reply: str
    citations: list[str]


def validate(data: dict) -> Draft:
    answerable = data.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError("'answerable' must be true or false")
    reply = data.get("reply")
    if not isinstance(reply, str):
        raise ValueError("'reply' must be a string")
    if answerable and not reply.strip():
        raise ValueError("'reply' is empty but 'answerable' is true")
    citations = data.get("citations")
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise ValueError("'citations' must be a list of strings")
    return Draft(answerable, reply.strip(), [c.strip() for c in citations])


def knowledge_block(chunks: list[Chunk]) -> str:
    body = "\n\n".join(f"[{c.id}] {c.title}\n{c.text}" for c in chunks)
    return f"<knowledge>\n{body}\n</knowledge>"


def _prompt(email: Email, triage: Triage, chunks: list[Chunk]) -> str:
    return (f"{knowledge_block(chunks)}\n\n"
            f"Triage: categories={', '.join(triage.categories)}; summary: {triage.summary}\n\n"
            f"Write a reply draft to this email.\n\n{email_block(email)}")


def write_draft(client: LLMClient, email: Email, triage: Triage, chunks: list[Chunk]) -> Draft:
    return generate_json(client, [Message("user", _prompt(email, triage, chunks))], system=SYSTEM,
                         schema=SCHEMA, validate=validate)


def revise_draft(client: LLMClient, email: Email, triage: Triage, chunks: list[Chunk],
                 draft: Draft, problems: list[str]) -> Draft:
    previous = json.dumps({"answerable": draft.answerable, "reply": draft.reply, "citations": draft.citations})
    feedback = ("Your draft failed these automatic checks:\n- " + "\n- ".join(problems) +
                "\nFix every problem and answer again with the same JSON format. "
                "If you cannot answer without breaking the rules, set answerable to false.")
    messages = [Message("user", _prompt(email, triage, chunks)), Message("assistant", previous),
                Message("user", feedback)]
    return generate_json(client, messages, system=SYSTEM, schema=SCHEMA, validate=validate)
