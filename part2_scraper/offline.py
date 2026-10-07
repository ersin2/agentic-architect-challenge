"""Offline stand-in for the model (LLM_PROVIDER=fake). NOT a model.

It behaves like a careless model on purpose: it copies sentences and ignores
the word limit, even when asked to compress. That way the offline demo shows
the guardrail doing its job (measure, ask once, then cut).
"""

from __future__ import annotations

import re

from agentkit.llm import FakeRequest

_BLOCK = re.compile(r"<(page|notes|summary)>\n(.*)\n</\1>", re.DOTALL)


def responder(request: FakeRequest) -> str:
    match = _BLOCK.search(request.last_user_text)
    content = match.group(2) if match else request.last_user_text
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(content.split()))
    if match and match.group(1) == "page" and "This is part" in request.last_user_text:
        return "\n".join(f"- {s}" for s in sentences[:4])  # map step: a few notes
    return " ".join(s.lstrip("- ") for s in sentences)  # single, reduce, compress: everything, too long
