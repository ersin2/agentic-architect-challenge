"""Part 2 failure types. Each one maps to a CLI exit code and a clear message."""

from __future__ import annotations


class ScrapeError(Exception):
    exit_code = 1


class FetchError(ScrapeError):
    """The page could not be downloaded safely. `kind` says why."""

    exit_code = 3

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind  # bad_url | blocked_address | robots | timeout | http_status | too_many_redirects
        #                   | unsupported_content | network


class ExtractionError(ScrapeError):
    """The page has no readable text (even after rendering, if that was possible)."""

    exit_code = 4


class RenderUnavailable(ScrapeError):
    """The page needs JavaScript, but Playwright or its browser is not installed."""

    exit_code = 4


class SummaryError(ScrapeError):
    """The model could not produce a usable summary."""

    exit_code = 5
