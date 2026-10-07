"""Offline stand-in for the model (LLM_PROVIDER=fake). NOT a model.

It only knows the demo conversation. It exists so the loop, the tool call and
the memory can be run and seen without an API key. It reads the user's name
from the system prompt (the pinned profile), exactly where a real model would.
"""

from __future__ import annotations

import json
import re

from agentkit.llm import FakeRequest, tool_call

TIER1 = ("london", "new york", "tokyo", "singapore")


def _asked_city(request: FakeRequest) -> str:
    for message in request.messages:
        if message.role == "user":
            for city in TIER1:
                if city in message.content.lower():
                    return city.title()
    return ""


def responder(request: FakeRequest):
    last = request.messages[-1]
    if last.role == "tool":
        result = json.loads(last.content)
        if "error" in result:
            return f"I could not calculate that: {result['error']}"
        return f"4 nights x 220 USD = {result['result']} USD. That is the most you can claim for the hotel."
    question = last.content.lower()
    known = re.search(r"name: ([^;.]+)", request.system or "")
    name = known.group(1) if known else None
    if "my name is" in question:
        return f"Nice to meet you, {name}! Ask me anything about the travel policy."
    if "my name" in question:
        city = _asked_city(request)
        return (f"Your name is {name}." if name else "You have not told me your name.") + \
            (f" You asked about {city}." if city else "")
    if "nights" in question:
        nights = re.search(r"(\d+) nights", question)
        return [tool_call("calculator", expression=f"{nights.group(1) if nights else 1} * 220")]
    if "hotel limit" in question:
        return "The hotel limit in Tokyo (a Tier 1 city) is 220 USD per night."
    if "receipt" in question:
        return "Yes. Every expense over 25 USD needs a receipt."
    return "(offline mode) I only know the demo questions. Set a real provider in .env to ask anything."
