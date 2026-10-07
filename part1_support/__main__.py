"""Run the support-email agent over a batch of emails.

    python -m part1_support                      # sample emails, provider from .env
    LLM_PROVIDER=fake python -m part1_support    # offline, no key needed
    python -m part1_support --only e02 e11       # a subset

Human-readable results go to stdout, JSON logs to stderr, decisions to out/decisions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agentkit.cli import prepare_cli
from agentkit.llm import get_client
from agentkit.metrics import METRICS
from agentkit.obs import log_event

from . import offline
from .history import ContactHistory
from .kb import KnowledgeBase
from .models import Email
from .pipeline import SupportAgent

DATA = Path(__file__).resolve().parent / "data"
log = logging.getLogger("part1.cli")


def load_emails(path: Path) -> list[Email]:
    emails = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            emails.append(Email.from_dict(json.loads(line)))
        except (ValueError, TypeError) as exc:
            # Defined outcome for a broken record: counted, logged, reported; the batch continues.
            METRICS.incr("part1.invalid_records")
            log_event(log, "email.invalid_record", logging.ERROR, line=number, error=str(exc)[:200])
            print(f"Skipped invalid record on line {number}: {exc}", file=sys.stderr)
    return emails


def load_history(path: Path) -> ContactHistory:
    history = ContactHistory()
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")).get("contacts", []):
            seed = Email.from_dict({**item, "sender": item["customer"]})
            history.record(seed.customer_id, seed.message_id, seed.received_at)
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m part1_support", description=__doc__.splitlines()[1])
    parser.add_argument("--emails", type=Path, default=DATA / "emails.jsonl")
    parser.add_argument("--history", type=Path, default=DATA / "contact_history.json")
    parser.add_argument("--kb", type=Path, default=DATA / "kb")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("out") / "decisions.jsonl")
    parser.add_argument("--only", nargs="*", help="process only these message ids")
    args = parser.parse_args(argv)

    settings = prepare_cli()
    if settings is None:
        return 2
    client = get_client(settings, fake_responder=offline.responder)
    agent = SupportAgent(client, KnowledgeBase.load(args.kb), load_history(args.history))
    emails = load_emails(args.emails)
    if args.only:
        emails = [e for e in emails if e.message_id in set(args.only)]

    decisions = agent.process_batch(emails, workers=args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    print(f"Provider: {client.provider} ({client.model})   emails: {len(decisions)}\n")
    print(f"{'ID':<5} {'ROUTE':<13} {'TEAM':<17} {'CALLS':>5}  {'TRACE':<12}  REASONS / CATEGORIES")
    for d in decisions:
        detail = ", ".join(d.reasons) or ", ".join(d.categories)
        if d.reasons and d.categories:
            detail += f"  [{', '.join(d.categories)}]"
        print(f"{d.message_id:<5} {d.route:<13} {d.team or '-':<17} {d.llm_calls:>5}  {d.trace_id:<12}  {detail}")
    for d in decisions:
        if d.draft:
            print(f"\n--- {d.message_id} draft (citations: {', '.join(d.citations)}) ---\n{d.draft}")
        if d.violations:
            print(f"\n--- {d.message_id} rejected after revision ---\n- " + "\n- ".join(d.violations))
    print("\nRun report:\n" + json.dumps(METRICS.snapshot(), indent=2))
    print(f"\nDecisions written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
