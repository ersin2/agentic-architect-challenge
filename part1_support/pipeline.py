"""The Part 1 agent: one explicit state machine per email.

  1 record contact      (code, idempotent)
  2 escalation gate     (code)  -> human, with zero model calls
  3 triage              (model) -> classify + plan the KB search
  4 retrieve            (code)  -> BM25; whole refund policy pinned if refunds come up
  5 draft               (model) -> grounded reply with policy placeholders
  6 verify              (code)  -> refund guard, citations, links, length
  7 revise once         (model) -> fix the exact problems found in step 6, then verify again

Principles: the model plans and writes; code decides and verifies. Any doubt
or failure sends the email to a human (fail closed). The model can add an
escalation (critical_issue, answerable=false) but can never remove one.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from agentkit.errors import LLMError, LLMOutputError
from agentkit.llm import CountingClient, LLMClient
from agentkit.metrics import METRICS
from agentkit.obs import hash_id, log_event, span, trace

from .drafter import revise_draft, write_draft
from .escalation import check_escalation
from .history import ContactHistory
from .kb import KnowledgeBase
from .models import DRAFT_READY, HUMAN_REVIEW, Decision, Email, Reason, team_for
from .refund_guard import mentions_refund, render_policy_slots
from .triage import run_triage
from .verify import verify_draft

log = logging.getLogger("part1.pipeline")


@dataclass
class SupportAgent:
    client: LLMClient
    kb: KnowledgeBase
    history: ContactHistory

    def ingest(self, email: Email) -> None:
        """Record the contact. Safe to call more than once for the same email."""
        self.history.record(email.customer_id, email.message_id, email.received_at)

    def process(self, email: Email) -> Decision:
        with trace() as trace_id:
            start = time.perf_counter()
            llm = CountingClient(self.client)
            decision = self._run(email, llm, trace_id)
            decision.llm_calls = llm.calls
            decision.duration_ms = round((time.perf_counter() - start) * 1000, 1)
            self._report(email, decision)
            return decision

    def process_batch(self, emails: Iterable[Email], workers: int = 4) -> list[Decision]:
        """Process emails in parallel. Every email gets a decision, even if the code has a bug."""
        emails = list(emails)
        for email in emails:  # record all contacts first, so counts never depend on processing order
            self.ingest(email)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(self._process_safely, emails))

    # ------------------------------------------------------------------ #

    def _process_safely(self, email: Email) -> Decision:
        try:
            return self.process(email)
        except Exception:  # last line of defence: a bug must not lose an email
            log.exception("email.internal_error", extra={"fields": {"message_id": email.message_id}})
            METRICS.incr("part1.route.human_review")
            return Decision(email.message_id, "-", HUMAN_REVIEW, [Reason.INTERNAL_ERROR.value],
                            team=team_for([]))

    def _run(self, email: Email, llm: CountingClient, trace_id: str) -> Decision:
        def human(*reasons: Reason, **extra) -> Decision:
            return Decision(email.message_id, trace_id, HUMAN_REVIEW, [r.value for r in reasons],
                            team=team_for(list(reasons)), **extra)

        # 1. Record this contact and count contacts in the last 7 days (this one included).
        with span(log, "step.record_contact", message_id=email.message_id,
                  customer=hash_id(email.customer_id)) as s:
            self.ingest(email)
            contacts = self.history.count_in_window(email.customer_id, email.received_at)
            s["contacts_7d"] = contacts

        # 2. Escalation gate: deterministic, before any model call.
        with span(log, "step.escalation_gate") as s:
            gate = check_escalation(email, contacts)
            s["escalate"] = gate.escalate
            s["reasons"] = [r.value for r in gate.reasons]
        if gate.escalate:
            return human(*gate.reasons, evidence=gate.evidence)

        # 3. Triage: classify and plan the knowledge-base search.
        try:
            with span(log, "step.triage") as s:
                triage = run_triage(llm, email)
                s["categories"] = triage.categories
                s["critical_issue"] = triage.critical_issue
        except LLMOutputError:
            return human(Reason.TRIAGE_FAILED)
        except LLMError:
            return human(Reason.LLM_UNAVAILABLE)
        if triage.critical_issue:
            return human(Reason.MODEL_FLAGGED_CRITICAL, categories=triage.categories)

        # 4. Retrieve. The refund policy is pinned whole whenever refunds come up.
        with span(log, "step.retrieve") as s:
            include_policy = mentions_refund(email.text) or any(mentions_refund(q) for q in triage.kb_queries)
            chunks = self.kb.retrieve(triage.kb_queries + [email.subject], include_policy=include_policy)
            s["chunk_ids"] = [c.id for c in chunks]
            s["policy_pinned"] = include_policy
        if not chunks:
            return human(Reason.NOT_ANSWERABLE, categories=triage.categories)

        # 5-7. Draft, verify in code, revise once if needed.
        try:
            with span(log, "step.draft") as s:
                draft = write_draft(llm, email, triage, chunks)
                s["answerable"] = draft.answerable
            problems = verify_draft(draft, chunks, self.kb) if draft.answerable else []
            if problems:
                METRICS.incr("part1.draft_rejected")
                log_event(log, "draft.rejected", logging.WARNING, problems=problems)
                with span(log, "step.revise") as s:
                    draft = revise_draft(llm, email, triage, chunks, draft, problems)
                    problems = verify_draft(draft, chunks, self.kb) if draft.answerable else []
                    s["problems_left"] = len(problems)
        except LLMOutputError:
            return human(Reason.DRAFT_FAILED, categories=triage.categories)
        except LLMError:
            return human(Reason.LLM_UNAVAILABLE, categories=triage.categories)

        if not draft.answerable:
            return human(Reason.NOT_ANSWERABLE, categories=triage.categories)
        if problems:
            return human(Reason.VERIFICATION_FAILED, categories=triage.categories, violations=problems)

        return Decision(email.message_id, trace_id, DRAFT_READY, categories=triage.categories,
                        draft=render_policy_slots(draft.reply, self.kb.policy), citations=draft.citations)

    @staticmethod
    def _report(email: Email, decision: Decision) -> None:
        METRICS.incr(f"part1.route.{decision.route}")
        for reason in decision.reasons:
            METRICS.incr(f"part1.reason.{reason}")
        METRICS.observe("part1.email_ms", decision.duration_ms)
        log_event(log, "email.decision", message_id=email.message_id, route=decision.route,
                  reasons=decision.reasons, team=decision.team, categories=decision.categories,
                  llm_calls=decision.llm_calls, duration_ms=decision.duration_ms)
