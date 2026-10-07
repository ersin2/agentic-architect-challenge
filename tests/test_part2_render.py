"""Part 2 JavaScript rendering with Playwright, against a local web server (no internet).

Skipped automatically when Playwright or its Chromium build is not installed.
"""

from __future__ import annotations

import functools
import http.server
import sys
import threading
from pathlib import Path

import pytest

from agentkit.llm import FakeClient
from part2_scraper.errors import RenderUnavailable
from part2_scraper.extract import extract
from part2_scraper.render import render_html
from part2_scraper.scraper import ScrapeConfig, summarize_url

FIXTURES = Path(__file__).parent / "fixtures" / "part2"
LOCAL = ScrapeConfig(allow_private=True, respect_robots=False, max_words=50)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture(scope="module")
def local_site():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(_QuietHandler, directory=str(FIXTURES)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _render_or_skip(url: str) -> str:
    try:
        return render_html(url, allow_private=True)
    except RenderUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.render
def test_javascript_page_is_rendered_and_its_text_extracted(local_site):
    page = extract(_render_or_skip(f"{local_site}/js_page.html"))
    assert "Lighthouses guided ships" in page.text and not page.needs_js


@pytest.mark.render
def test_auto_mode_uses_the_browser_only_when_needed(local_site):
    _render_or_skip(f"{local_site}/js_page.html")
    js = summarize_url(f"{local_site}/js_page.html", FakeClient(script=["Lighthouses guided ships."]), LOCAL)
    plain = summarize_url(f"{local_site}/article_with_boilerplate.html", FakeClient(script=["Corals recover."]), LOCAL)
    assert js.stats["rendered_with_browser"] is True and "Lighthouses" in js.summary
    assert plain.stats["rendered_with_browser"] is False


def test_missing_playwright_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)  # makes the import fail
    with pytest.raises(RenderUnavailable, match="pip install playwright"):
        render_html("https://example.com/")
