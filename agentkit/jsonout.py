"""Turn model text into a validated JSON object.

JSON mode makes valid JSON likely, not certain. So the answer is always parsed
and validated in code. If it is wrong, the model is asked once more, with the
exact error. If it is still wrong, LLMOutputError is raised and the caller
chooses a safe outcome (in Part 1: route the email to a human).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Sequence, TypeVar

from .errors import LLMOutputError
from .llm import LLMClient, Message
from .obs import log_event

log = logging.getLogger("agentkit.jsonout")
T = TypeVar("T")

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences and short text around it."""
    s = text.strip()
    fenced = _FENCE.match(s)
    if fenced:
        s = fenced.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("no JSON object in the model output") from None
        try:
            obj = json.loads(s[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid JSON: {exc.msg} at char {exc.pos}") from None
    if not isinstance(obj, dict):
        raise LLMOutputError("expected a JSON object")
    return obj


def generate_json(client: LLMClient, messages: Sequence[Message], *, system: str,
                  schema: dict[str, Any], validate: Callable[[dict[str, Any]], T],
                  max_output_tokens: int = 2048, repair_attempts: int = 1) -> T:
    """Ask for JSON, validate it, and repair once. `validate` raises ValueError on bad data."""
    history = list(messages)
    problem: Exception | None = None
    for attempt in range(1, repair_attempts + 2):
        resp = client.generate(history, system=system, json_schema=schema, max_output_tokens=max_output_tokens)
        try:
            return validate(parse_json_object(resp.text))
        except (LLMOutputError, ValueError) as exc:
            problem = exc
            log_event(log, "llm.json_invalid", logging.WARNING, attempt=attempt, error=str(exc)[:200])
            history = history + [
                resp.message,
                Message("user", f"Your previous answer was rejected: {exc}. "
                                "Answer again with only one JSON object that matches the schema."),
            ]
    raise LLMOutputError(f"model output still invalid after {repair_attempts + 1} attempts: {problem}")
