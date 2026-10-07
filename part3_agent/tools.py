"""The agent's tools. There is one: a calculator.

The tool is a pure function with no side effects (no network, no files), so even
if a document or a user tricks the model into calling it, nothing can be harmed.
Errors are returned to the model as data ({"error": ...}) so it can correct itself.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentkit.errors import ToolError
from agentkit.llm import ToolCall, ToolSpec
from agentkit.metrics import METRICS
from agentkit.obs import log_event

from .calculator import calculate

log = logging.getLogger("part3.tools")

CALCULATOR = ToolSpec(
    name="calculator",
    description=("Evaluate an arithmetic expression exactly. Supports + - * / // % ** and parentheses, and the "
                 "functions round(x, n), min(), max(), abs(), sqrt(). Write plain numbers without units or "
                 "thousands separators, for example '4 * 220' or 'round(236 * 0.62, 2)'."),
    parameters={"type": "object",
                "properties": {"expression": {"type": "string", "description": "The arithmetic expression."}},
                "required": ["expression"]},
)
TOOLS = [CALCULATOR]


def execute(call: ToolCall) -> dict[str, Any]:
    start = time.perf_counter()
    if call.error:
        outcome: dict[str, Any] = {"error": call.error}
    elif call.name != CALCULATOR.name:
        outcome = {"error": f"unknown tool {call.name!r}; the only tool is 'calculator'"}
    elif not isinstance(call.arguments.get("expression"), str):
        outcome = {"error": "the argument 'expression' (a string) is required"}
    else:
        try:
            outcome = {"result": calculate(call.arguments["expression"])}
        except ToolError as exc:
            outcome = {"error": str(exc)}
    METRICS.incr(f"tool.{call.name}.{'error' if 'error' in outcome else 'ok'}")
    log_event(log, "tool.call", tool=call.name, arguments=call.arguments, ok="error" not in outcome,
              outcome=outcome, duration_ms=round((time.perf_counter() - start) * 1000, 2))
    return outcome
