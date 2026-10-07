"""Download a page safely.

Fixes DEFECT 1 of the old script. Every way a download can go wrong ends in a
FetchError with a `kind`, never in a hang or a crash:
- slow server       -> connect/read timeouts plus an overall deadline (a server that
                       drips one byte per second never trips a read timeout alone)
- huge page         -> stop reading at max_bytes and flag the result as truncated
- PDF, image, video -> rejected by content type before the body is read
- 404 / 5xx / 429   -> 4xx fails at once; 5xx and 429 are retried with backoff
- redirect loops    -> at most `max_redirects` hops, each one checked again
- internal targets  -> loopback, private and link-local addresses are refused (SSRF),
                       also when a public URL redirects to one
- robots.txt        -> respected by default
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from agentkit.obs import log_event
from agentkit.resilience import RetryPolicy

from .errors import FetchError

log = logging.getLogger("part2.fetch")

ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
RETRYABLE = {429, 500, 502, 503, 504}

Resolver = Callable[[str], list[str]]


def user_agent() -> str:
    """Identify the bot with a contact. Sites such as Wikipedia reject requests without one (HTTP 403).

    Set SCRAPER_CONTACT in .env to a URL or email address that reaches you, for example the
    URL of this repository. Read at call time, so values from .env are picked up.
    """
    contact = os.environ.get("SCRAPER_CONTACT", "").strip() or "https://github.com/"
    return f"AgenticArchitectChallenge-Summarizer/1.0 (+{contact}; educational project) httpx"


def resolve_host(host: str) -> list[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        raise FetchError("network", f"cannot resolve host {host!r}") from exc


@dataclass
class FetchResult:
    url: str  # final URL after redirects
    status: int
    content_type: str
    body: bytes
    encoding: str | None
    truncated: bool
    elapsed_ms: float


def check_url(url: str, *, allow_private: bool = False, resolver: Resolver = resolve_host) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("bad_url", f"only http(s) URLs with a host are allowed, got {url!r}")
    if allow_private:
        return
    for address in resolver(parts.hostname):
        ip = ipaddress.ip_address(address.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchError("blocked_address", f"{parts.hostname} resolves to a non-public address ({ip})")


def _robots_allows(client: httpx.Client, url: str) -> bool:
    robots_url = urljoin(url, "/robots.txt")
    try:
        resp = client.get(robots_url, timeout=5.0)
    except httpx.HTTPError:
        return True  # robots.txt unreachable: treat as no rules
    if resp.status_code != 200:
        return True  # 4xx means "no rules"; we also do not block on 5xx
    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    return parser.can_fetch(user_agent(), url)


def fetch(url: str, *, timeout_s: float = 15.0, deadline_s: float = 30.0, max_bytes: int = 5_000_000,
          max_redirects: int = 5, respect_robots: bool = True, allow_private: bool = False,
          retry: RetryPolicy = RetryPolicy(max_attempts=2, base_delay_s=1.0),
          transport: httpx.BaseTransport | None = None, resolver: Resolver = resolve_host,
          sleep: Callable[[float], None] = time.sleep) -> FetchResult:
    started = time.monotonic()
    timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0))
    with httpx.Client(timeout=timeout, transport=transport, follow_redirects=False,
                      headers={"User-Agent": user_agent(), "Accept": "text/html,application/xhtml+xml,text/plain"}) as client:
        check_url(url, allow_private=allow_private, resolver=resolver)
        if respect_robots and not _robots_allows(client, url):
            raise FetchError("robots", f"robots.txt does not allow fetching {url}")
        for attempt in range(1, retry.max_attempts + 1):
            try:
                return _get(client, url, started, deadline_s, max_bytes, max_redirects, allow_private, resolver)
            except _Retryable as exc:
                if attempt == retry.max_attempts:
                    raise FetchError(exc.kind, str(exc)) from None
                delay = retry.delay(attempt, exc.retry_after)
                log_event(log, "fetch.retry", logging.WARNING, attempt=attempt, delay_s=round(delay, 2), reason=str(exc))
                sleep(delay)
    raise AssertionError("unreachable")


class _Retryable(Exception):
    def __init__(self, kind: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


def _get(client: httpx.Client, url: str, started: float, deadline_s: float, max_bytes: int,
         max_redirects: int, allow_private: bool, resolver: Resolver) -> FetchResult:
    current = url
    for _ in range(max_redirects + 1):
        try:
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    current = urljoin(current, location)
                    check_url(current, allow_private=allow_private, resolver=resolver)  # re-check every hop
                    continue
                if resp.status_code in RETRYABLE:
                    retry_after = resp.headers.get("retry-after")
                    raise _Retryable("http_status", f"HTTP {resp.status_code} from {current}",
                                     float(retry_after) if retry_after and retry_after.isdigit() else None)
                if resp.status_code >= 400:
                    raise FetchError("http_status", f"HTTP {resp.status_code} from {current}")
                content_type = resp.headers.get("content-type", "text/html").split(";")[0].strip().lower()
                if content_type not in ALLOWED_TYPES:
                    raise FetchError("unsupported_content", f"content type {content_type!r} is not HTML or text")
                body, truncated = bytearray(), False
                for chunk in resp.iter_bytes():
                    if time.monotonic() - started > deadline_s:  # overall deadline: not retried
                        raise FetchError("timeout", f"page took longer than {deadline_s:.0f}s to download")
                    body += chunk
                    if len(body) > max_bytes:
                        del body[max_bytes:]
                        truncated = True
                        break
                elapsed = round((time.monotonic() - started) * 1000, 1)
                log_event(log, "fetch.done", status=resp.status_code, content_type=content_type,
                          bytes=len(body), truncated=truncated, elapsed_ms=elapsed)
                return FetchResult(str(resp.url), resp.status_code, content_type, bytes(body),
                                   resp.charset_encoding, truncated, elapsed)
        except httpx.TimeoutException as exc:
            raise _Retryable("timeout", f"timed out ({type(exc).__name__}) fetching {current}") from None
        except httpx.TransportError as exc:
            raise _Retryable("network", f"network error ({type(exc).__name__}) fetching {current}") from None
    raise FetchError("too_many_redirects", f"more than {max_redirects} redirects starting at {url}")
