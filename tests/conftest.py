"""Shared test fixtures. Every test runs offline: no API key, no network."""

from __future__ import annotations

import io
import json
import logging

import pytest

from agentkit.metrics import METRICS
from agentkit.obs import setup_logging


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Make sure no real key or provider setting leaks into a test from the developer's shell."""
    for name in ("LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_RPM"):
        monkeypatch.delenv(name, raising=False)
    METRICS.reset()
    yield
    METRICS.reset()


@pytest.fixture
def json_logs():
    """Capture JSON log lines. Call the returned function to get them as dicts."""
    stream = io.StringIO()
    setup_logging("DEBUG", "json", stream)

    def lines() -> list[dict]:
        return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]

    yield lines
    logging.getLogger().handlers.clear()
