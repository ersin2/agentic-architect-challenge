"""Render a JavaScript-only page in headless Chromium (Playwright). Optional.

Used only when extract() flags `needs_js` (or with --render always), because a
browser is about 100x slower and heavier than a plain HTTP request.
Images, fonts and media are blocked: we only need the text.
"""

from __future__ import annotations

import logging

from agentkit.obs import log_event

from .errors import FetchError, RenderUnavailable
from .fetch import USER_AGENT, Resolver, check_url, resolve_host

log = logging.getLogger("part2.render")
SKIP_RESOURCES = {"image", "media", "font"}


def render_html(url: str, *, timeout_s: float = 20.0, allow_private: bool = False,
                resolver: Resolver = resolve_host, max_chars: int = 5_000_000) -> str:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RenderUnavailable("this page needs JavaScript, but Playwright is not installed. "
                                "Run: pip install playwright && python -m playwright install chromium") from exc

    check_url(url, allow_private=allow_private, resolver=resolver)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError as exc:
            raise RenderUnavailable("Chromium could not start. Run: python -m playwright install chromium "
                                    f"({str(exc).splitlines()[0][:150]})") from exc
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.route("**/*", lambda route: route.abort() if route.request.resource_type in SKIP_RESOURCES
                       else route.continue_())
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
            try:  # best effort: some pages never go idle (polling, websockets)
                page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeout:
                log_event(log, "render.networkidle_timeout", logging.WARNING)
            html = page.content()
        except PlaywrightTimeout as exc:
            raise FetchError("timeout", f"browser did not load {url} within {timeout_s:.0f}s") from exc
        except PlaywrightError as exc:
            raise FetchError("network", f"browser error: {str(exc).splitlines()[0][:200]}") from exc
        finally:
            browser.close()
    log_event(log, "render.done", chars=len(html))
    return html[:max_chars]
