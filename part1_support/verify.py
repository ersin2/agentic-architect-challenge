"""Verify a draft in code before anyone sees it. Returns a list of problems (empty = pass).

Each problem is written so it can be sent back to the model as revision feedback.
"""

from __future__ import annotations

import re

from .drafter import MAX_REPLY_WORDS, Draft
from .kb import Chunk, KnowledgeBase
from .refund_guard import check_refund_claims

_LINK = re.compile(r"https?://[^\s)>\]\"']+|[\w.+-]+@[\w-]+\.[\w.-]+")


def verify_draft(draft: Draft, chunks: list[Chunk], kb: KnowledgeBase) -> list[str]:
    problems: list[str] = []
    provided = {c.id for c in chunks}

    if not draft.citations:
        problems.append("no citations: list the knowledge section ids you used")
    unknown = [c for c in draft.citations if c not in provided]
    if unknown:
        problems.append(f"citations {unknown} were not in the provided knowledge")

    problems += check_refund_claims(draft.reply, kb.policy, provided).violations

    for link in _LINK.findall(draft.reply):
        if link.rstrip(".,") not in kb.allowed_links:
            problems.append(f"link or address {link!r} does not appear in the knowledge base")

    words = len(draft.reply.split())
    if words > MAX_REPLY_WORDS:
        problems.append(f"reply has {words} words; the limit is {MAX_REPLY_WORDS}")
    return problems
