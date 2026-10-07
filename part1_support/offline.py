"""Offline stand-in for the model (LLM_PROVIDER=fake).

It lets anyone run the whole pipeline with no API key. It is keyword-based and
NOT a model: it shows the plumbing (gate, routing, verification, revision),
not answer quality. On purpose, it "falls for" the bait in email e11 on the
first draft, so the refund guard and the revision loop can be seen offline.
"""

from __future__ import annotations

import re

from agentkit.llm import FakeRequest

_EMAIL = re.compile(r"<email>\s*Subject: (?P<subject>.*?)\n(?P<body>.*?)</email>", re.DOTALL)
_KNOWLEDGE_ITEM = re.compile(r"^\[(?P<id>[^\]]+)\] (?P<title>.*)\n(?P<text>.*)$", re.MULTILINE)

_CATEGORY_WORDS = {
    "Billing": r"invoice|charge|refund|plan|payment|cancel|subscription|price|money",
    "Technical": r"crash|error|sync|export|sign-?in|password|bug|slow|login",
    "Feedback": r"love|great|request|suggest|feature",
}


def responder(request: FakeRequest) -> dict:
    properties = (request.json_schema or {}).get("properties", {})
    original = request.messages[0].content
    if "kb_queries" in properties:
        return _triage(original)
    if "answerable" in properties:
        return _draft(original, revising=len(request.messages) > 1)
    raise ValueError("offline responder: unexpected request")


def _email_parts(prompt: str) -> tuple[str, str]:
    match = _EMAIL.search(prompt)
    return (match["subject"].strip(), match["body"].strip()) if match else ("", prompt)


def _triage(prompt: str) -> dict:
    subject, body = _email_parts(prompt)
    text = f"{subject} {body}".lower()
    categories = [name for name, words in _CATEGORY_WORDS.items() if re.search(words, text)] or ["Other"]
    return {"categories": categories, "critical_issue": False, "kb_queries": [subject, body[:120]],
            "summary": f"Customer writes about: {subject}"}


def _draft(prompt: str, revising: bool) -> dict:
    _, body = _email_parts(prompt)
    items = {m["id"]: m["text"] for m in _KNOWLEDGE_ITEM.finditer(prompt)}
    faq_ids = [i for i in items if not i.startswith("refund_policy#")][:1]
    has_policy = "refund_policy#R1" in items
    if not faq_ids and not has_policy:
        return {"answerable": False, "reply": "", "citations": []}
    lines = ["Hello,", "", "Thank you for contacting AcmeSync."]
    if faq_ids:
        lines[-1] += f" {items[faq_ids[0]].split('. ')[0].rstrip('.')}."
    citations = list(faq_ids)
    if has_policy:
        lower = body.lower()
        clause = "R1" if "annual" in lower else "R2" if "monthly" in lower else "R4" if "twice" in lower else "R6"
        lines.append(f"Our refund policy says: {{{{policy:{clause}}}}}")
        citations.append(f"refund_policy#{clause}")
        if "100%" in body and not revising:
            lines.append("As promised, you will get a 100% refund of both charges plus 3 free months.")
    lines += ["", "Best regards,", "AcmeSync Support"]
    return {"answerable": True, "reply": "\n".join(lines), "citations": citations}
