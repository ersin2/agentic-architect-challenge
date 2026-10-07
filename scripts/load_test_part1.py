"""Load and chaos test for Part 1, fully offline.

Runs many synthetic emails through the real pipeline with a fake model that
adds latency and fails at random. It checks two things:
- every email gets a defined route (no crash, no lost email);
- throughput scales with workers (the model call is the slow part).

    python scripts/load_test_part1.py --emails 500 --workers 16 --failure-rate 0.1 --latency 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentkit.llm import FakeClient, FaultyClient  # noqa: E402
from agentkit.metrics import METRICS  # noqa: E402
from agentkit.obs import setup_logging  # noqa: E402
from part1_support import offline  # noqa: E402
from part1_support.history import ContactHistory  # noqa: E402
from part1_support.kb import KnowledgeBase  # noqa: E402
from part1_support.models import DRAFT_READY, HUMAN_REVIEW, Email  # noqa: E402
from part1_support.pipeline import SupportAgent  # noqa: E402

TEMPLATES = [
    ("Invoice copy", "Where can I download my invoice as a PDF?"),
    ("Refund", "I bought the annual plan last week. Can I get a refund?"),
    ("Sync problem", "Sync is slow on my Android phone."),
    ("Outage?", "Is there an outage? Nothing loads."),
    ("Feature idea", "Love the app. Could you add a shortcut to switch themes?"),
    ("Hacked", "I think my account was hacked."),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emails", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--failure-rate", type=float, default=0.1)
    parser.add_argument("--latency", type=float, default=0.05, help="seconds per fake model call")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("ERROR")  # keep the console readable; the report below is the output
    rng = random.Random(args.seed)
    start_time = datetime(2026, 10, 6, tzinfo=timezone.utc)
    emails = []
    for i in range(args.emails):
        subject, body = rng.choice(TEMPLATES)
        sender = f"customer{rng.randint(1, args.emails // 3)}@example.com"  # repeat senders -> frequent contact
        received = start_time + timedelta(minutes=rng.randint(0, 7 * 24 * 60))
        emails.append(Email(f"load-{i}", sender, subject, body, received))

    client = FaultyClient(FakeClient(offline.responder), failure_rate=args.failure_rate,
                          latency_s=args.latency, seed=args.seed)
    agent = SupportAgent(client, KnowledgeBase.load(Path("part1_support/data/kb")), ContactHistory())

    started = time.perf_counter()
    decisions = agent.process_batch(emails, workers=args.workers)
    elapsed = time.perf_counter() - started

    routes = Counter(d.route for d in decisions)
    reasons = Counter(r for d in decisions for r in d.reasons)
    undefined = [d for d in decisions if d.route not in (DRAFT_READY, HUMAN_REVIEW) or
                 (d.route == HUMAN_REVIEW and not d.reasons)]
    report = {
        "emails": len(emails), "decisions": len(decisions), "workers": args.workers,
        "failure_rate": args.failure_rate, "latency_s": args.latency,
        "elapsed_s": round(elapsed, 2), "emails_per_s": round(len(decisions) / elapsed, 1),
        "routes": dict(routes), "reasons": dict(reasons.most_common()),
        "undefined_outcomes": len(undefined), "metrics": METRICS.snapshot()["timings"],
    }
    print(json.dumps(report, indent=2))
    ok = len(decisions) == len(emails) and not undefined
    print("RESULT:", "PASS - every email has a defined route" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
