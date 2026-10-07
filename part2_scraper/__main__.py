"""Summarise a web page (or a saved HTML file) in at most N words.

    python -m part2_scraper https://en.wikipedia.org/wiki/Web_scraping
    python -m part2_scraper https://quotes.toscrape.com/js/ --render auto
    python -m part2_scraper --file tests/fixtures/part2/article_with_boilerplate.html   # offline

Exit codes: 0 ok, 2 configuration/usage, 3 fetch failed, 4 no readable text or
browser unavailable, 5 model failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentkit.cli import prepare_cli
from agentkit.llm import get_client

from . import offline
from .errors import ScrapeError
from .scraper import ScrapeConfig, summarize_file, summarize_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m part2_scraper", description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", help="page to summarise")
    parser.add_argument("--file", type=Path, help="summarise a saved HTML file instead of a URL")
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--render", choices=["auto", "never", "always"], default="auto")
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--ignore-robots", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    if bool(args.url) == bool(args.file):
        parser.error("give either a URL or --file")
    if not 10 <= args.max_words <= 1000:
        parser.error("--max-words must be between 10 and 1000")

    settings = prepare_cli()
    if settings is None:
        return 2
    client = get_client(settings, fake_responder=offline.responder)
    config = ScrapeConfig(max_words=args.max_words, render=args.render, max_chunks=args.max_chunks,
                          respect_robots=not args.ignore_robots)
    try:
        report = summarize_file(args.file, client, config) if args.file else summarize_url(args.url, client, config)
    except ScrapeError as exc:
        print(f"Error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return exc.exit_code

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0
    s = report.stats
    print(f"{report.title or report.source}\n{report.source}\n")
    print(report.summary + "\n")
    print(f"words: {s['words']}/{s['max_words']}   compressed by model: {s['compressed_by_model']}   "
          f"cut by guardrail: {s['truncated_by_guardrail']}")
    print(f"page text: {s['raw_page_text_chars']:,} chars raw -> {s['main_text_chars']:,} chars main content "
          f"({s['extraction_method']})   chunks: {s['chunks']}   model calls: {s['llm_calls']}   "
          f"browser: {s.get('rendered_with_browser', False)}   trace: {report.trace_id}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
