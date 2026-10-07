"""Refund guard: refund facts must come from the policy text, never from the model."""

from __future__ import annotations

from pathlib import Path

import pytest

from part1_support.kb import KnowledgeBase
from part1_support.refund_guard import check_refund_claims, render_policy_slots

KB = KnowledgeBase.load(Path(__file__).resolve().parent.parent / "part1_support" / "data" / "kb")
POLICY = KB.policy
ALL_POLICY = {f"refund_policy#{cid}" for cid in POLICY}


def check(reply: str, retrieved: set[str] = ALL_POLICY):
    return check_refund_claims(reply, POLICY, retrieved)


def test_policy_has_the_six_clauses():
    assert sorted(POLICY) == ["R1", "R2", "R3", "R4", "R5", "R6"]


def test_invented_refund_term_is_rejected_and_the_number_is_named():
    result = check("Good news: you can get a full refund within 60 days.")
    assert not result.ok
    assert "60" in result.violations[0]


def test_placeholder_is_accepted_and_rendered_with_exact_policy_text():
    reply = "Our refund policy says: {{policy:R1}}"
    assert check(reply).ok
    assert render_policy_slots(reply, POLICY) == f'Our refund policy says: "{POLICY["R1"]}"'


def test_full_chunk_id_in_placeholder_is_accepted():
    assert check("Per our policy: {{policy:refund_policy#R3}}").ok


def test_unknown_clause_is_rejected():
    result = check("Per our policy: {{policy:R9}}")
    assert not result.ok and "R9" in result.violations[0]


def test_clause_that_was_not_retrieved_is_rejected():
    result = check("Per our policy: {{policy:R2}}", retrieved={"billing_faq#invoices"})
    assert not result.ok and "not in the provided knowledge" in result.violations[0]


def test_paraphrase_next_to_a_placeholder_is_rejected():
    assert not check("{{policy:R1}} So you are eligible for a full refund.").ok


@pytest.mark.parametrize("reply", [
    "Monthly plans are non-refundable, sorry.",        # a negative claim is still a policy claim
    "We will refund the duplicate charge right away.",
    "Your money back will arrive in a few days.",
    "Refunds usually take 3 business days.",
    "We will credit your card tomorrow.",
    "We will reverse the charge today.",
])
def test_refund_claims_in_own_words_are_rejected(reply):
    assert not check(reply).ok


def test_refund_promise_inside_a_technical_reply_is_caught():
    """The guard runs on every draft, not only on Billing emails."""
    reply = "Sorry the export crashed. Update the app. We will refund this month as an apology."
    assert not check(reply, retrieved={"technical_faq#export-to-pdf-fails"}).ok


@pytest.mark.parametrize("reply", [
    "I understand you would like a refund. Our refund policy says: {{policy:R1}}",
    "I have passed your refund request to our billing team.",
    "Please update to version 4.3 and restart the app 2 times.",   # numbers, but not about refunds
    "Our billing team checks duplicate charges within 2 business days.",  # FAQ fact, no refund wording
])
def test_harmless_sentences_pass(reply):
    assert check(reply).ok


def test_known_blind_spot_promise_without_any_refund_word():
    """Documented limit: the claim detector needs a refund-related word. This promise has none,
    so it passes. The placeholder rule still means the policy text itself is never misquoted."""
    assert check("Don't worry, the 50 USD will be back on your card on Friday.").ok
