"""Part 2 fetching: every failure ends in a FetchError with a kind, never a hang."""

from __future__ import annotations

import time

import httpx
import pytest

from part2_scraper.errors import FetchError
from part2_scraper.fetch import fetch

PUBLIC = lambda host: ["93.184.216.34"]  # noqa: E731 - fake DNS: every host is public
HTML = {"content-type": "text/html; charset=utf-8"}


def run(handler, url="https://example.test/page", resolver=PUBLIC, **kwargs):
    kwargs.setdefault("respect_robots", False)
    return fetch(url, transport=httpx.MockTransport(handler), resolver=resolver, sleep=lambda s: None, **kwargs)


class Counter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, request):
        self.urls.append(str(request.url))
        return self.responses.pop(0)


def test_html_page_is_returned_with_encoding():
    result = run(lambda r: httpx.Response(200, headers=HTML, content=b"<p>hi</p>"))
    assert result.body == b"<p>hi</p>" and result.encoding == "utf-8" and not result.truncated


def test_pdf_is_rejected_before_reading_it():
    with pytest.raises(FetchError) as err:
        run(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7"))
    assert err.value.kind == "unsupported_content"


def test_404_fails_at_once_without_retry():
    handler = Counter(httpx.Response(404, headers=HTML))
    with pytest.raises(FetchError) as err:
        run(handler)
    assert err.value.kind == "http_status" and len(handler.urls) == 1


def test_503_is_retried_then_succeeds():
    handler = Counter(httpx.Response(503), httpx.Response(200, headers=HTML, content=b"<p>ok</p>"))
    assert run(handler).body == b"<p>ok</p>" and len(handler.urls) == 2


def test_timeouts_end_in_a_timeout_error():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(FetchError) as err:
        run(slow)
    assert err.value.kind == "timeout"


def test_server_that_drips_bytes_hits_the_overall_deadline():
    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(1000):
                time.sleep(0.01)
                yield b"<p>x</p>"
    with pytest.raises(FetchError) as err:
        run(lambda r: httpx.Response(200, headers=HTML, stream=Drip()), deadline_s=0.2)
    assert err.value.kind == "timeout"


def test_huge_page_is_cut_at_the_byte_limit_and_flagged():
    result = run(lambda r: httpx.Response(200, headers=HTML, content=b"a" * 50_000), max_bytes=10_000)
    assert len(result.body) == 10_000 and result.truncated


@pytest.mark.parametrize("url", ["ftp://example.test/file", "file:///etc/passwd", "javascript:alert(1)", "http://"])
def test_non_http_urls_are_refused(url):
    with pytest.raises(FetchError) as err:
        run(lambda r: httpx.Response(200), url=url)
    assert err.value.kind == "bad_url"


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"])
def test_internal_addresses_are_refused(address):
    with pytest.raises(FetchError) as err:
        run(lambda r: httpx.Response(200), resolver=lambda host: [address])
    assert err.value.kind == "blocked_address"


def test_redirect_to_an_internal_address_is_refused():
    resolver = lambda host: ["10.0.0.1"] if host == "internal.test" else ["93.184.216.34"]  # noqa: E731
    handler = Counter(httpx.Response(302, headers={"location": "http://internal.test/admin"}))
    with pytest.raises(FetchError) as err:
        run(handler, resolver=resolver)
    assert err.value.kind == "blocked_address"


def test_redirect_loop_stops():
    with pytest.raises(FetchError) as err:
        run(lambda r: httpx.Response(302, headers={"location": "/again"}), max_redirects=3)
    assert err.value.kind == "too_many_redirects"


def test_normal_redirect_is_followed():
    handler = Counter(httpx.Response(301, headers={"location": "/new"}),
                      httpx.Response(200, headers=HTML, content=b"<p>moved</p>"))
    result = run(handler)
    assert result.url == "https://example.test/new" and result.body == b"<p>moved</p>"


def test_robots_txt_disallow_is_respected():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private")
        return httpx.Response(200, headers=HTML, content=b"<p>secret</p>")
    with pytest.raises(FetchError) as err:
        run(handler, url="https://example.test/private/page", respect_robots=True)
    assert err.value.kind == "robots"
    assert run(handler, url="https://example.test/public", respect_robots=True).body == b"<p>secret</p>"
