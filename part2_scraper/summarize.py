"""Map-reduce summarisation: the fix for the bottleneck (DEFECT 4).

Short text: one call. Long text: each chunk is summarised to a few bullet notes
("map", in parallel), then the notes are combined into one summary ("reduce").
Every call has a bounded input size, whatever the page length.

Partial failure is tolerated: if a minority of chunks fail, the summary is built
from the rest and a warning says which part is missing. If most fail, we stop
with an error instead of presenting a summary of a few fragments as complete.
"""

from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agentkit.errors import LLMError
from agentkit.injection import untrusted_note, wrap_untrusted
from agentkit.llm import CountingClient, LLMClient, Message
from agentkit.obs import log_event

from .chunking import chunk_text
from .errors import SummaryError
from .guardrail import enforce_word_limit

log = logging.getLogger("part2.summarize")

MAP_SYSTEM = ("You read one part of a web page and write 3 to 6 short bullet points with its most important "
              "facts. Use only the given text. " + untrusted_note("page"))
REDUCE_SYSTEM = ("You combine notes taken from all parts of one web page into one summary of plain prose, "
                 "without a title or bullet points. Use only facts from the notes. " + untrusted_note("notes"))
SINGLE_SYSTEM = ("You summarise web pages in plain prose, without a title or bullet points. "
                 "Use only facts from the page. " + untrusted_note("page"))


@dataclass
class SummaryResult:
    summary: str
    words: int
    max_words: int
    compressed: bool
    truncated_by_guardrail: bool
    chunks: int
    chunks_failed: int
    content_truncated: bool
    llm_calls: int
    warnings: list[str] = field(default_factory=list)


def summarize(client: LLMClient, text: str, *, title: str = "", max_words: int = 120,
              chunk_chars: int = 12_000, max_chunks: int = 10, workers: int = 4) -> SummaryResult:
    llm = CountingClient(client)
    chunks, content_truncated = chunk_text(text, chunk_chars, max_chunks)
    if not chunks:
        raise SummaryError("there is no text to summarise")
    warnings = []
    if content_truncated:
        warnings.append(f"page is very long: only the first {len(chunks)} parts "
                        f"({sum(map(len, chunks)):,} characters) were summarised")
    heading = f"Page title: {title}\n\n" if title else ""
    failed = 0
    try:
        if len(chunks) == 1:
            prompt = (f"{heading}Summarise this page in at most {max_words} words.\n\n"
                      + wrap_untrusted("page", chunks[0]))
            draft = llm.generate([Message("user", prompt)], system=SINGLE_SYSTEM, max_output_tokens=2048).text
        else:
            notes = _map(llm, chunks, heading, workers)
            failed = sum(n is None for n in notes)
            if failed > len(chunks) // 2:
                raise SummaryError(f"{failed} of {len(chunks)} parts could not be summarised (model unavailable)")
            if failed:
                missing = ", ".join(str(i + 1) for i, n in enumerate(notes) if n is None)
                warnings.append(f"parts {missing} of {len(chunks)} could not be summarised and are missing")
            joined = "\n\n".join(f"Part {i + 1} notes:\n{n}" for i, n in enumerate(notes) if n)
            prompt = (f"{heading}Combine these notes into one summary of at most {max_words} words.\n\n"
                      + wrap_untrusted("notes", joined))
            draft = llm.generate([Message("user", prompt)], system=REDUCE_SYSTEM, max_output_tokens=2048).text
        guarded = enforce_word_limit(llm, draft, max_words)
    except LLMError as exc:
        raise SummaryError(f"model error: {exc}") from exc
    return SummaryResult(guarded.text, guarded.words, max_words, guarded.compressed, guarded.truncated,
                         len(chunks), failed, content_truncated, llm.calls, warnings)


def _map(llm: CountingClient, chunks: list[str], heading: str, workers: int) -> list[str | None]:
    def one(index: int) -> str | None:
        prompt = (f"{heading}This is part {index + 1} of {len(chunks)}.\n\n" + wrap_untrusted("page", chunks[index]))
        try:
            return llm.generate([Message("user", prompt)], system=MAP_SYSTEM, max_output_tokens=1024).text.strip() or None
        except LLMError as exc:
            log_event(log, "summarize.map_failed", logging.WARNING, part=index + 1, error=type(exc).__name__)
            return None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as pool:
        # copy_context() keeps the trace id in the worker threads, so their log lines join this request.
        futures = [pool.submit(contextvars.copy_context().run, one, i) for i in range(len(chunks))]
        return [f.result() for f in futures]
