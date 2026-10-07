"""Conversation memory: short-term history plus a small pinned profile.

- History: every message of the conversation. The model sees the most recent
  whole turns that fit a token budget. A turn starts with a user message and
  holds everything until the next one, so a tool call is never separated from
  its result (both providers reject a tool result without its call).
- Profile: facts that must survive trimming, today only the user's name. It is
  found with simple rules ("my name is X", "call me X") and put in the system prompt.
  Weakness: other phrasings ("I'm Aigerim") are only remembered while that turn
  is still inside the history window.
- save()/load() keep a session in a JSON file between runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agentkit.llm import Message

_NAME = re.compile(r"\b(?:my name is|call me)\s+([^\W\d_][\w'\-]*)", re.IGNORECASE)
_NOT_NAMES = {"not", "a", "an", "the", "just", "really", "very", "no", "none", "unknown"}


def extract_name(text: str) -> str | None:
    match = _NAME.search(text)
    if not match or match.group(1).lower() in _NOT_NAMES:
        return None
    name = match.group(1)
    return name[0].upper() + name[1:]


def estimate_tokens(messages: list[Message]) -> int:
    """Rough count (about 4 characters per token). Good enough for a budget."""
    chars = sum(len(m.content) + sum(len(json.dumps(c.arguments)) for c in m.tool_calls) + 16 for m in messages)
    return chars // 4


class ConversationMemory:
    def __init__(self, max_tokens: int = 6_000) -> None:
        self.max_tokens = max_tokens
        self.messages: list[Message] = []
        self.profile: dict[str, str] = {}

    def add_user(self, text: str) -> None:
        name = extract_name(text)
        if name:
            self.profile["name"] = name
        self.messages.append(Message("user", text))

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def rollback(self, length: int) -> None:
        """Forget everything after the first `length` messages (used when a turn fails)."""
        del self.messages[length:]

    def turns(self) -> list[list[Message]]:
        turns: list[list[Message]] = []
        for message in self.messages:
            if message.role == "user" or not turns:
                turns.append([])
            turns[-1].append(message)
        return turns

    def window(self) -> list[Message]:
        """The most recent whole turns within the budget. The latest turn is always included."""
        selected: list[list[Message]] = []
        used = 0
        for turn in reversed(self.turns()):
            cost = estimate_tokens(turn)
            if selected and used + cost > self.max_tokens:
                break
            selected.insert(0, turn)
            used += cost
        return [m for turn in selected for m in turn]

    def profile_text(self) -> str:
        if not self.profile:
            return ""
        facts = "; ".join(f"{key}: {value}" for key, value in self.profile.items())
        return f"Known facts about the user from earlier in this conversation: {facts}."

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"profile": self.profile, "messages": [m.to_dict() for m in self.messages]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, max_tokens: int = 6_000) -> "ConversationMemory":
        memory = cls(max_tokens)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            memory.profile = dict(data.get("profile", {}))
            memory.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return memory
