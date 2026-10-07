"""Conciseness guardrail: the word limit is measured and enforced in code.

Fixes DEFECT 7. A model's own word count cannot be trusted, and max_output_tokens
is the wrong tool (it cuts mid-sentence and also counts hidden "thinking" tokens).
So:
1. clean the text (drop "Here is a summary:" preambles and repeated sentences);
2. count words in code;
3. if over the limit, ask the model ONCE to compress, telling it the measured count;
4. if still over, cut at the last sentence that fits, and flag `truncated`.

Guarantee: the returned text never has more than `max_words` words.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from agentkit.errors import LLMError
from agentkit.injection import untrusted_note, wrap_untrusted
from agentkit.llm import LLMClient, Message
from agentkit.obs import log_event

from .errors import SummaryError

log = logging.getLogger("part2.guardrail")

_WORD = re.compile(r"[^\W_]+(?:['’.-][^\W_]+)*")
_PREAMBLE = re.compile(r"^\s*(here(?:'s| is| are)|below is|sure|certainly|okay|of course)\b[^\n:]{0,80}:\s*",
                       re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN = re.compile(r"^\s*(#{1,6}\s+|[-*•]\s+)|\*\*|__", re.MULTILINE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def clean(text: str) -> str:
    text = _PREAMBLE.sub("", text.strip())
    text = _MARKDOWN.sub("", text)
    seen: set[str] = set()
    sentences = []
    for sentence in _SENTENCE.split(" ".join(text.split())):
        key = sentence.lower().strip()
        if key and key not in seen:
            seen.add(key)
            sentences.append(sentence.strip())
    return " ".join(sentences)


def truncate_to_words(text: str, max_words: int) -> str:
    """Keep whole sentences while they fit. If even the first one is too long, cut it by words."""
    kept: list[str] = []
    used = 0
    for sentence in _SENTENCE.split(text):
        n = count_words(sentence)
        if used + n > max_words:
            break
        kept.append(sentence)
        used += n
    if kept:
        return " ".join(kept)
    words = text.split()
    while count_words(" ".join(words)) > max_words - 1:  # leave room so the ellipsis never adds a word
        words.pop()
    return " ".join(words).rstrip(",;:") + " …"


@dataclass
class GuardrailResult:
    text: str
    words: int
    compressed: bool
    truncated: bool


COMPRESS_SYSTEM = ("You shorten summaries. Keep the most important facts, drop details. "
                   "Return only the shortened summary as plain prose. " + untrusted_note("summary"))


def enforce_word_limit(client: LLMClient | None, draft: str, max_words: int) -> GuardrailResult:
    text = clean(draft)
    if not text:
        raise SummaryError("the model returned an empty summary")
    words = count_words(text)
    compressed = False
    if words > max_words and client is not None:
        target = max(10, int(max_words * 0.85))  # aim below the limit: models overshoot
        log_event(log, "guardrail.compress", words=words, max_words=max_words, target=target)
        prompt = (f"This summary has {words} words. Rewrite it in at most {target} words.\n\n"
                  + wrap_untrusted("summary", text))
        try:
            candidate = clean(client.generate([Message("user", prompt)], system=COMPRESS_SYSTEM,
                                              max_output_tokens=2048).text)
        except LLMError as exc:  # the cut below still guarantees the limit
            log_event(log, "guardrail.compress_failed", logging.WARNING, error=type(exc).__name__)
            candidate = ""
        if candidate and count_words(candidate) < words:  # keep it only if it really got shorter
            text, compressed = candidate, True
            words = count_words(text)
    truncated = words > max_words
    if truncated:
        text = truncate_to_words(text, max_words)
        log_event(log, "guardrail.truncated", logging.WARNING, max_words=max_words, words_after=count_words(text))
    final_words = count_words(text)
    if final_words > max_words:  # cannot happen; checked so a future change cannot break the promise silently
        raise AssertionError(f"guardrail broke its guarantee: {final_words} > {max_words}")
    return GuardrailResult(text, final_words, compressed, truncated)
