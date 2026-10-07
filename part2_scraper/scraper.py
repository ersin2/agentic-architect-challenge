"""Orchestrates Part 2: fetch -> extract -> (render) -> summarise -> guardrail, under one trace id."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agentkit.injection import find_injection_signals
from agentkit.llm import LLMClient
from agentkit.obs import log_event, span, trace

from .errors import ExtractionError
from .extract import Extracted, extract
from .fetch import Resolver, fetch, resolve_host
from .render import render_html
from .summarize import SummaryResult, summarize

log = logging.getLogger("part2.scraper")
MIN_TEXT_CHARS = 50


@dataclass
class ScrapeConfig:
    max_words: int = 120
    render: str = "auto"  # auto | never | always
    respect_robots: bool = True
    allow_private: bool = False
    chunk_chars: int = 12_000
    max_chunks: int = 10
    workers: int = 4
    timeout_s: float = 15.0
    max_bytes: int = 5_000_000


@dataclass
class Report:
    source: str
    title: str
    summary: str
    trace_id: str
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_url(url: str, client: LLMClient, config: ScrapeConfig | None = None, *,
                  transport: httpx.BaseTransport | None = None, resolver: Resolver = resolve_host) -> Report:
    config = config or ScrapeConfig()
    with trace() as trace_id:
        warnings: list[str] = []
        with span(log, "step.fetch", url=url) as s:
            fetched = fetch(url, timeout_s=config.timeout_s, max_bytes=config.max_bytes,
                            respect_robots=config.respect_robots, allow_private=config.allow_private,
                            transport=transport, resolver=resolver)
            s.update(http_status=fetched.status, bytes=len(fetched.body), truncated=fetched.truncated)
        if fetched.truncated:
            warnings.append(f"page is larger than {config.max_bytes:,} bytes; only the first part was read")
        with span(log, "step.extract") as s:
            page = extract(fetched.body, fetched.encoding)
            s.update(method=page.method, raw_chars=page.raw_chars, text_chars=len(page.text), needs_js=page.needs_js)
        rendered = False
        if config.render == "always" or (config.render == "auto" and page.needs_js):
            with span(log, "step.render") as s:
                page = extract(render_html(fetched.url, allow_private=config.allow_private, resolver=resolver))
                rendered = True
                s.update(method=page.method, text_chars=len(page.text))
        stats = {"final_url": fetched.url, "html_bytes": len(fetched.body), "rendered_with_browser": rendered}
        return _summarise(page, client, config, url, trace_id, stats, warnings)


def summarize_file(path: Path, client: LLMClient, config: ScrapeConfig | None = None) -> Report:
    """Offline entry point: summarise a saved HTML file (no network)."""
    config = config or ScrapeConfig()
    with trace() as trace_id:
        body = path.read_bytes()
        with span(log, "step.extract", file=path.name) as s:
            page = extract(body)
            s.update(method=page.method, raw_chars=page.raw_chars, text_chars=len(page.text))
        return _summarise(page, client, config, str(path), trace_id, {"html_bytes": len(body)}, [])


def _summarise(page: Extracted, client: LLMClient, config: ScrapeConfig, source: str, trace_id: str,
               stats: dict[str, Any], warnings: list[str]) -> Report:
    if len(page.text) < MIN_TEXT_CHARS:
        hint = " The page seems to need JavaScript; try --render always." if page.needs_js else ""
        raise ExtractionError(f"no readable text found ({len(page.text)} characters).{hint}")
    signals = find_injection_signals(page.text)
    if signals:
        # The page text is wrapped as untrusted data and the length guardrail still applies,
        # so we continue, but we tell the user the page tried to give instructions.
        warnings.append(f"page contains instruction-like text ({', '.join(signals)}); it was treated as data")
    with span(log, "step.summarize") as s:
        result: SummaryResult = summarize(client, page.text, title=page.title, max_words=config.max_words,
                                          chunk_chars=config.chunk_chars, max_chunks=config.max_chunks,
                                          workers=config.workers)
        s.update(chunks=result.chunks, llm_calls=result.llm_calls, words=result.words,
                 compressed=result.compressed, truncated=result.truncated_by_guardrail)
    stats.update({
        "extraction_method": page.method,
        "raw_page_text_chars": page.raw_chars,  # what the old script would have sent to the model
        "main_text_chars": len(page.text),
        "chunks": result.chunks, "chunks_failed": result.chunks_failed, "llm_calls": result.llm_calls,
        "words": result.words, "max_words": result.max_words,
        "compressed_by_model": result.compressed, "truncated_by_guardrail": result.truncated_by_guardrail,
    })
    warnings += result.warnings
    log_event(log, "page.summarised", **{k: v for k, v in stats.items() if k != "final_url"})
    return Report(source, page.title, result.summary, trace_id, stats, warnings)
