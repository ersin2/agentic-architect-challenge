"""The document Q&A agent: a small tool-calling loop (the ReAct pattern) over raw API calls.

One user question = one "turn":
  1. add the question to memory
  2. ask the model, with the document in the system prompt and the calculator offered
  3. if the model answers with text -> done
  4. if it asks for the calculator -> run it, add the result to memory, go to 2
The model decides whether to call the tool. The code decides how many steps are allowed.

Every way the loop can go wrong has a defined outcome:
  model outage   -> polite message; the failed turn is removed, so memory stays valid
  tool error     -> the error goes back to the model as data; it can try again
  same call twice-> stop (the model is going round in circles)
  too many steps -> stop with a clear message (no runaway loop, no runaway bill)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentkit.errors import LLMError
from agentkit.injection import untrusted_note, wrap_untrusted
from agentkit.llm import LLMClient, Message
from agentkit.metrics import METRICS
from agentkit.obs import log_event, trace

from . import tools
from .memory import ConversationMemory

log = logging.getLogger("part3.agent")
DOCUMENT_PATH = Path(__file__).resolve().parent / "data" / "sample_document.md"

SYSTEM = """You answer questions about one document: the AcmeSync Travel and Expense Policy below.
Rules:
- Answer only from the document. If it does not cover the question, say so.
- Always use the calculator tool for arithmetic (any multiplication, addition, percentage or total).
- Never use the calculator for a number that is already written in the document.
- When you used a calculation, show it briefly (for example: 4 nights x 220 USD = 880 USD).
- Keep answers short and friendly. Use the user's name if you know it.
{profile}
{note}

{document}"""

OUTAGE_REPLY = "Sorry, I cannot reach the language model right now. Please try again in a moment."
STEP_LIMIT_REPLY = "Sorry, I could not finish this answer. Please ask in a simpler way."


def load_document(path: Path = DOCUMENT_PATH) -> str:
    return path.read_text(encoding="utf-8")


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    stopped: str = "answer"  # answer | llm_error | repeated_call | max_steps
    trace_id: str = ""


class DocumentAgent:
    def __init__(self, client: LLMClient, document: str, memory: ConversationMemory | None = None,
                 max_steps: int = 5) -> None:
        self.client = client
        self.document = document
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps

    def system_prompt(self) -> str:
        return SYSTEM.format(profile=self.memory.profile_text(), note=untrusted_note("document"),
                             document=wrap_untrusted("document", self.document))

    def ask(self, question: str) -> AgentReply:
        with trace() as trace_id:
            start = len(self.memory.messages)
            self.memory.add_user(question)
            used: list[dict[str, Any]] = []
            seen: set[str] = set()
            for step in range(1, self.max_steps + 1):
                try:
                    resp = self.client.generate(self.memory.window(), system=self.system_prompt(),
                                                tools=tools.TOOLS, max_output_tokens=2048)
                except LLMError as exc:
                    self.memory.rollback(start)  # drop the whole turn: no half-finished history
                    return self._finish(AgentReply(OUTAGE_REPLY, used, step, "llm_error", trace_id), error=exc)
                self.memory.add(resp.message)
                if not resp.tool_calls:
                    return self._finish(AgentReply(resp.text.strip(), used, step, "answer", trace_id))
                for call in resp.tool_calls:
                    key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                    if key in seen:
                        return self._stop(start, AgentReply(STEP_LIMIT_REPLY, used, step, "repeated_call", trace_id))
                    seen.add(key)
                    outcome = tools.execute(call)
                    used.append({"name": call.name, "arguments": call.arguments, **outcome})
                    self.memory.add(Message("tool", json.dumps(outcome), tool_call_id=call.id, name=call.name))
            return self._stop(start, AgentReply(STEP_LIMIT_REPLY, used, self.max_steps, "max_steps", trace_id))

    def _stop(self, start: int, reply: AgentReply) -> AgentReply:
        """Abnormal stop: keep the question and the stop message, drop the unfinished tool exchange."""
        self.memory.rollback(start + 1)
        self.memory.add(Message("assistant", reply.text))
        return self._finish(reply)

    @staticmethod
    def _finish(reply: AgentReply, error: Exception | None = None) -> AgentReply:
        METRICS.incr(f"part3.turn.{reply.stopped}")
        log_event(log, "agent.turn", logging.WARNING if reply.stopped != "answer" else logging.INFO,
                  stopped=reply.stopped, steps=reply.steps, tools=[c["name"] for c in reply.tool_calls],
                  error=type(error).__name__ if error else None)
        return reply
