"""Contact history: how many times a customer contacted support in a time window.

Stored in SQLite (standard library). Two properties matter:
- Idempotent: recording the same message_id twice counts once, so a retry or
  a duplicate delivery never inflates the count.
- Order-independent: we count by the email's own timestamp, not by the order
  we process emails in. So parallel workers get the same answer.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta

WINDOW = timedelta(days=7)


def _epoch(at: datetime) -> float:
    if at.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return at.timestamp()


class ContactHistory:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS contacts ("
                " message_id TEXT PRIMARY KEY, customer TEXT NOT NULL, ts REAL NOT NULL)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_customer_ts ON contacts (customer, ts)")

    def record(self, customer: str, message_id: str, at: datetime) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO contacts (message_id, customer, ts) VALUES (?, ?, ?)",
                               (message_id, customer.strip().lower(), _epoch(at)))

    def count_in_window(self, customer: str, at: datetime, window: timedelta = WINDOW) -> int:
        """Contacts with timestamp in (at - window, at]. The contact at `at` itself is included."""
        end = _epoch(at)
        start = end - window.total_seconds()
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM contacts WHERE customer = ? AND ts > ? AND ts <= ?",
                (customer.strip().lower(), start, end)).fetchone()
        return int(row[0])
