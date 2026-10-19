"""A disk cache for judge calls, so re-running the same answers costs no quota.

One SQLite file (stdlib, no server), easy to persist between CI runs with actions/cache. Keys
are hashes of the full request (model, messages, parameters, prompt version), never the API key.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol


def cache_key(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache(Protocol):
    hits: int
    misses: int

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...


class NoCache:
    def __init__(self) -> None:
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            self.misses += 1
        return None

    def set(self, key: str, value: str) -> None:
        pass


class DiskCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # one connection shared by worker threads; every use below holds the lock
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT, created REAL)"
        )
        self.hits = self.misses = 0

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            value: str = row[0]
        return value

    def set(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, value, time.time())
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()
