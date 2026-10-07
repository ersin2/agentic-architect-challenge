"""Settings loaded from environment variables and an optional .env file.

The API key is stored in a field excluded from repr(), so printing or logging
a Settings object never shows it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Checked against the providers' model lists on 2026-10-07. Override with LLM_MODEL.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-6-luna",
    "fake": "fake",
}
KEY_NAMES = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_s: float = 60.0
    max_attempts: int = 3
    rpm: int = 0
    thinking_level: str = ""
    log_format: str = "json"
    log_level: str = "INFO"

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (tests) or from os.environ plus the project .env file."""
    if env is None:
        # override=False: a variable already set in the shell wins over .env.
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ

    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        if env.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif env.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "fake"
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {provider!r}")

    api_key = ""
    if provider != "fake":
        key_name = KEY_NAMES[provider]
        api_key = env.get(key_name, "").strip()
        if not api_key:
            raise ConfigError(
                f"LLM_PROVIDER={provider} but {key_name} is not set. "
                "Copy .env.example to .env and add your key, or set LLM_PROVIDER=fake to run offline."
            )

    try:
        return Settings(
            provider=provider,
            model=env.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider],
            api_key=api_key,
            timeout_s=float(env.get("LLM_TIMEOUT_S", "60")),
            max_attempts=int(env.get("LLM_MAX_ATTEMPTS", "3")),
            rpm=int(env.get("LLM_RPM", "0") or 0),
            thinking_level=env.get("LLM_THINKING_LEVEL", "").strip(),
            log_format=env.get("LOG_FORMAT", "json").strip().lower(),
            log_level=env.get("LOG_LEVEL", "INFO").strip().upper(),
        )
    except ValueError as exc:
        raise ConfigError(f"invalid numeric setting: {exc}") from exc
