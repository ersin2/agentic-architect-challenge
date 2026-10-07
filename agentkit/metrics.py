"""Thread-safe in-process metrics: counters and latency percentiles.

This is enough for an end-of-run report. In production the same calls would
feed Prometheus or OpenTelemetry instead of a dict.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from typing import Any


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile of `values` (p in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._timings: defaultdict[str, list[float]] = defaultdict(list)

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._timings[name].append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timings = {
                name: {
                    "count": len(vals),
                    "p50": round(percentile(vals, 50), 1),
                    "p95": round(percentile(vals, 95), 1),
                    "max": round(max(vals), 1),
                }
                for name, vals in self._timings.items()
                if vals
            }
            return {"counters": dict(sorted(self._counters.items())), "timings": timings}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()


# One process-wide registry. Tests call METRICS.reset().
METRICS = Metrics()
