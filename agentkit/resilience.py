"""Reliability building blocks: retry delays, a circuit breaker and a rate limiter.

All three take an injectable clock/sleep so tests run instantly.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Jitter spreads retries out, so many workers that failed together do not
    all retry at the same moment and overload the provider again.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    max_retry_after_s: float = 60.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait after failed attempt number `attempt` (1-based)."""
        if retry_after is not None:
            # The server told us when to come back. Trust it, within reason.
            return min(max(retry_after, 0.0), self.max_retry_after_s)
        ceiling = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return random.uniform(0, ceiling)


class CircuitBreaker:
    """Stop calling a provider that keeps failing.

    closed:    calls go through.
    open:      after `failure_threshold` failed calls in a row, calls fail fast
               for `reset_after_s` seconds. This protects us (no long timeouts)
               and the provider (no retry storm).
    half-open: after the cooldown, calls are let through again; one success
               closes the circuit, one failure opens it again.
    """

    def __init__(self, failure_threshold: int = 5, reset_after_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if self._clock() - self._opened_at >= self.reset_after_s:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = self._clock()


class RateLimiter:
    """Space calls at least 60/rpm seconds apart (rpm <= 0 disables it).

    A fixed spacing is simpler and more predictable than a token bucket.
    It keeps a free-tier key under its requests-per-minute quota, so we
    avoid 429 errors instead of reacting to them.
    """

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rpm = rpm
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> float:
        """Block until this caller may send. Returns the seconds waited."""
        if self.rpm <= 0:
            return 0.0
        with self._lock:  # reserve a slot under the lock, sleep outside it
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return wait
