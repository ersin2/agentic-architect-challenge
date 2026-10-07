"""Small helpers shared by the three command-line entry points."""

from __future__ import annotations

import sys

from .config import Settings, load_settings
from .errors import ConfigError
from .obs import setup_logging


def prepare_cli(log_level: str | None = None) -> Settings | None:
    """UTF-8 console, settings, logging. Returns None (after printing why) if configuration is invalid."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # absent when output is redirected oddly
        if reconfigure is not None:  # Windows consoles are not UTF-8 by default
            reconfigure(encoding="utf-8", errors="replace")
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return None
    setup_logging(log_level or settings.log_level, settings.log_format)
    return settings
