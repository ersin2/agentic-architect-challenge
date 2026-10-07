"""Part 1 pipeline end to end, with a scripted fake model.

Each test scripts exactly the model replies it expects. If the code makes an
extra model call, the FakeClient raises, so "zero calls" claims are enforced.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentkit.errors import LLMUnavailable
from agentkit.llm import FakeClient, FaultyClient
from part1_support import offline
from part1_support.history import ContactHistory
from part1_support.kb import KnowledgeBase
from part1_support.models import DRAFT_READY, HUMAN_REVIEW, Email
from part1_support.pipeline import SupportAgent

KB = KnowledgeBase.load(Path(__file__).resolve().parent.parent / "part1_support" / "data" / "kb")
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)

INVOICE = Email("m1", "ana@example.com", "Invoice copy", "Where can I download my invoice as a PDF?", NOW)
REFUND = Email("m2", "ben@example.com", "Refund", "I bought the annual plan 10 days ago. Can I get a refund?", NOW)
OUTAGE = Email("m3", "cy@example.com", "Everything is down", "Is there an outage? Nothing loads.", NOW)

TRIAGE_BILLING = {"categories": ["Billing"], "critical_issue": False,
                  "kb_queries": ["download invoice pdf"], "summary": "Wants an invoice copy."}
TRIAGE_REFUND = {"categories": ["Billing"], "critical_issue": False,
                 "kb_queries": ["annual plan refund"], "summary": "Wants a refund for an annual plan."}
GOOD_INVOICE_REPLY = "You can download it from Settings > Billing > Invoice history. AcmeSync Support"
INVENTED = "You are eligible for a full refund within 60 days. AcmeSync Support"
GROUNDED = "Our refund policy says: {{policy:R1}} AcmeSync Support"


def draft(reply: str, citations: list[str], answerable: bool = True) -> dict:
    return {"answerable": answerable, "reply": reply, "citations": citations}


def make_agent(script=None, client=None):
    fake = client or FakeClient(script=script)
    return SupportAgent(fake, KB, ContactHistory()), fake


# ------------------------- gate runs before any model call ------------------------- #

def test_escalated_email_never_reaches_the_model():
    agent, fake = make_agent(script=[])  # any model call would raise
    decision = agent.process(OUTAGE)
    assert (decision.route, decision.reasons, decision.team) == (HUMAN_REVIEW, ["service_outage"], "technical-oncall")
    assert decision.llm_calls == 0 and fake.calls == []


def test_fourth_contact_is_escalated_before_any_model_call():
    agent, fake = make_agent(script=[])
    for days in (1, 2, 3):
        agent.history.record(INVOICE.customer_id, f"old-{days}", NOW - timedelta(days=days))
    decision = agent.process(INVOICE)
    assert decision.reasons == ["frequent_contact"] and fake.calls == []


# ----------------------------------- happy paths ----------------------------------- #

def test_answerable_email_gets_a_verified_draft():
    agent, _ = make_agent(script=[TRIAGE_BILLING, draft(GOOD_INVOICE_REPLY, ["billing_faq#invoices"])])
    decision = agent.process(INVOICE)
    assert decision.route == DRAFT_READY
    assert decision.citations == ["billing_faq#invoices"] and decision.llm_calls == 2


def test_refund_reply_contains_the_policy_text_verbatim():
    agent, fake = make_agent(script=[TRIAGE_REFUND, draft(GROUNDED, ["refund_policy#R1"])])
    decision = agent.process(REFUND)
    assert decision.route == DRAFT_READY
    assert f'"{KB.policy["R1"]}"' in decision.draft and "{{" not in decision.draft
    assert "refund_policy#R1" in fake.calls[1].messages[0].content  # policy was pinned into the context


# ------------------------------ refund hallucination ------------------------------ #

def test_model_that_invents_a_refund_term_twice_goes_to_a_human():
    agent, _ = make_agent(script=[TRIAGE_REFUND, draft(INVENTED, ["refund_policy#R1"]),
                                  draft(INVENTED, ["refund_policy#R1"])])
    decision = agent.process(REFUND)
    assert (decision.route, decision.reasons) == (HUMAN_REVIEW, ["verification_failed"])
    assert decision.draft is None and any("60" in v for v in decision.violations)
    assert decision.llm_calls == 3  # triage, draft, one revision - then stop


def test_model_that_fixes_its_draft_on_revision_gets_through():
    agent, fake = make_agent(script=[TRIAGE_REFUND, draft(INVENTED, ["refund_policy#R1"]),
                                     draft(GROUNDED, ["refund_policy#R1"])])
    decision = agent.process(REFUND)
    assert decision.route == DRAFT_READY and decision.llm_calls == 3
    feedback = fake.calls[2].messages[-1].content
    assert "failed these automatic checks" in feedback and "60" in feedback


# --------------------------- model can only escalate up --------------------------- #

def test_model_can_add_an_escalation():
    agent, _ = make_agent(script=[{**TRIAGE_BILLING, "critical_issue": True}])
    decision = agent.process(INVOICE)
    assert decision.reasons == ["model_flagged_critical"] and decision.llm_calls == 1


def test_model_saying_not_answerable_sends_email_to_a_human():
    agent, _ = make_agent(script=[TRIAGE_BILLING, draft("", [], answerable=False)])
    assert agent.process(INVOICE).reasons == ["not_answerable_from_kb"]


# ------------------------------- malformed output ------------------------------- #

def test_unusable_triage_twice_fails_closed():
    agent, _ = make_agent(script=["not json", "still not json"])
    decision = agent.process(INVOICE)
    assert decision.reasons == ["triage_failed"] and decision.llm_calls == 2


def test_invented_category_is_rejected_then_repaired():
    agent, fake = make_agent(script=[{**TRIAGE_BILLING, "categories": ["Sales"]}, TRIAGE_BILLING,
                                     draft(GOOD_INVOICE_REPLY, ["billing_faq#invoices"])])
    assert agent.process(INVOICE).route == DRAFT_READY
    assert "unknown category 'Sales'" in fake.calls[1].messages[-1].content


def test_citation_outside_the_context_triggers_a_revision():
    agent, _ = make_agent(script=[TRIAGE_BILLING, draft(GOOD_INVOICE_REPLY, ["made_up#section"]),
                                  draft(GOOD_INVOICE_REPLY, ["billing_faq#invoices"])])
    decision = agent.process(INVOICE)
    assert decision.route == DRAFT_READY and decision.llm_calls == 3


def test_link_that_is_not_in_the_kb_is_never_sent():
    phishing = "Claim your refund at http://acme-refunds.example/claim now. AcmeSync Support"
    agent, _ = make_agent(script=[TRIAGE_BILLING, draft(phishing, ["billing_faq#invoices"]),
                                  draft(phishing, ["billing_faq#invoices"])])
    decision = agent.process(INVOICE)
    assert decision.route == HUMAN_REVIEW
    assert any("acme-refunds.example" in v for v in decision.violations)


def test_link_from_the_kb_is_allowed():
    triage = {**TRIAGE_BILLING, "categories": ["Technical"], "kb_queries": ["service status incident updates"]}
    reply = "You can follow live status at https://status.acmesync.example. AcmeSync Support"
    agent, _ = make_agent(script=[triage, draft(reply, ["technical_faq#service-status"])])
    question = Email("m9", "dee@example.com", "Status page?", "Where do you post incident updates?", NOW)
    assert agent.process(question).route == DRAFT_READY


def test_feedback_email_gets_the_feedback_guidance_even_without_matching_words():
    """Seen live: 'One request: could you add a shortcut?' never matched the feedback FAQ by words."""
    triage = {"categories": ["Feedback"], "critical_issue": False,
              "kb_queries": ["keyboard shortcut switch themes"], "summary": "Asks for a theme shortcut."}
    reply = "Thank you! We share every request with the product team. AcmeSync Support"
    agent, fake = make_agent(script=[triage, draft(reply, ["general_faq#feedback-and-feature-requests"])])
    email = Email("m8", "fay@example.com", "Love it", "One request: could you add a shortcut to switch themes?", NOW)
    assert agent.process(email).route == DRAFT_READY
    assert "general_faq#feedback-and-feature-requests" in fake.calls[1].messages[0].content


# --------------------------------- model outages --------------------------------- #

def test_model_outage_at_triage_fails_closed():
    agent, _ = make_agent(script=[LLMUnavailable("HTTP 503")])
    decision = agent.process(INVOICE)
    assert (decision.route, decision.reasons, decision.team) == (HUMAN_REVIEW, ["llm_unavailable"], "support-general")


def test_model_outage_while_drafting_keeps_the_triage_categories():
    agent, _ = make_agent(script=[TRIAGE_BILLING, LLMUnavailable("timeout")])
    decision = agent.process(INVOICE)
    assert decision.reasons == ["llm_unavailable"] and decision.categories == ["Billing"]


# --------------------------- batches, load and tracing --------------------------- #

def test_batch_counts_do_not_depend_on_processing_order():
    agent, _ = make_agent(client=FakeClient(offline.responder))
    emails = [Email(f"b{i}", "eve@example.com", "Invoice", "Where is my invoice?", NOW - timedelta(days=3 - i))
              for i in range(4)]
    decisions = {d.message_id: d for d in agent.process_batch(reversed(emails), workers=4)}
    assert decisions["b3"].reasons == ["frequent_contact"]  # the 4th contact by time
    assert all("frequent_contact" not in decisions[f"b{i}"].reasons for i in range(3))


def test_every_email_gets_a_defined_route_under_heavy_faults():
    flaky = FaultyClient(FakeClient(offline.responder), failure_rate=0.3, seed=7)
    agent, _ = make_agent(client=flaky)
    emails = [Email(f"x{i}", f"user{i}@example.com", "Invoice", "Where can I download my invoice?", NOW)
              for i in range(120)]
    decisions = agent.process_batch(emails, workers=8)
    assert len(decisions) == 120
    assert {d.route for d in decisions} == {DRAFT_READY, HUMAN_REVIEW}
    assert all(d.reasons == ["llm_unavailable"] for d in decisions if d.route == HUMAN_REVIEW)


def test_a_bug_in_processing_becomes_a_human_route_not_a_crash():
    def broken(_request):
        raise RuntimeError("bug")
    agent, _ = make_agent(client=FakeClient(broken))
    (decision,) = agent.process_batch([INVOICE])
    assert decision.reasons == ["internal_error"]


def test_one_email_can_be_followed_by_its_trace_id(json_logs):
    agent, _ = make_agent(script=[TRIAGE_BILLING, draft(GOOD_INVOICE_REPLY, ["billing_faq#invoices"])])
    decision = agent.process(INVOICE)
    lines = json_logs()
    events = [line["event"] for line in lines if line["trace_id"] == decision.trace_id]
    assert events == ["step.record_contact", "step.escalation_gate", "llm.call", "step.triage",
                      "step.retrieve", "llm.call", "step.draft", "email.decision"]
    assert all("ana@example.com" not in json.dumps(line) for line in lines)  # address is hashed
