"""Structured logging and tracing.

- Every log line is one JSON object on stderr (or readable text with LOG_FORMAT=text).
- The current trace id lives in a ContextVar. Every line written while handling one
  request carries that id, even lines written deep inside the LLM client.
  So `grep <trace_id>` shows one request end to end.
- span() times one step and logs its outcome as a single line.
- hash_id() pseudonymises personal data (email addresses) before it reaches a log.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, TextIO

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def current_trace_id() -> str:
    return _trace_id.get()


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def trace(trace_id: str | None = None) -> Iterator[str]:
    """Run a block under one trace id (a new one unless given)."""
    tid = trace_id or new_trace_id()
    token = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


def hash_id(value: str) -> str:
    """Stable, non-reversible short id for personal data such as an email address."""
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:10]


class _TraceFilter(logging.Filter):
    """Stamp each record with the trace id of the thread/context that created it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{ts} {record.levelname:<7} [{getattr(record, 'trace_id', '-')}] {record.getMessage()} {extra}".rstrip()


def setup_logging(level: str = "INFO", fmt: str = "json", stream: TextIO | None = None) -> None:
    """Send all logs to one handler (stderr by default). Human output stays on stdout."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.addFilter(_TraceFilter())
    handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # httpx logs every request URL at INFO. Keep it quiet; our client logs its own calls.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


@contextmanager
def span(logger: logging.Logger, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a step and log one line when it ends.

    The yielded dict can be filled with outcome fields, which are added to the line.
    """
    info: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    try:
        yield info
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(logger, name, logging.WARNING, status="error", error=type(exc).__name__,
                  duration_ms=duration_ms, **info)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    log_event(logger, name, status="ok", duration_ms=duration_ms, **info)
