"""Persistent, version-aware cache port and SQLite implementation."""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from internship_scanner.exceptions import IntelligenceCacheError


class InferenceCache(ABC):
    """Storage boundary for deterministic inference artifacts."""

    @abstractmethod
    def get(self, namespace: str, key: str, version: str) -> Any | None:
        """Return an exact namespace/key/version match when present."""

    @abstractmethod
    def set(self, namespace: str, key: str, version: str, value: Any) -> None:
        """Persist a JSON-compatible value."""

    @abstractmethod
    def invalidate(self, namespace: str | None = None) -> int:
        """Delete cached values and return the number of removed rows."""


class SQLiteInferenceCache(InferenceCache):
    """Thread-safe persistent cache with explicit version invalidation."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        if path != Path(":memory:"):
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(path, check_same_thread=False)
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, cache_key, version)
                )
                """
            )
            self._connection.commit()
        except sqlite3.Error as error:
            raise IntelligenceCacheError(
                f"Cannot initialize AI cache: {path}"
            ) from error

    def get(self, namespace: str, key: str, version: str) -> Any | None:
        """Return the decoded value for an exact cache identity."""

        try:
            with self._lock:
                row = self._connection.execute(
                    """
                    SELECT payload FROM inference_cache
                    WHERE namespace = ? AND cache_key = ? AND version = ?
                    """,
                    (namespace, key, version),
                ).fetchone()
            return json.loads(str(row[0])) if row else None
        except (sqlite3.Error, json.JSONDecodeError) as error:
            raise IntelligenceCacheError("Cannot read the AI cache") from error

    def set(self, namespace: str, key: str, version: str, value: Any) -> None:
        """Atomically insert or replace one JSON-compatible cache value."""

        try:
            payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
            with self._lock:
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO inference_cache
                    (namespace, cache_key, version, payload)
                    VALUES (?, ?, ?, ?)
                    """,
                    (namespace, key, version, payload),
                )
                self._connection.commit()
        except (sqlite3.Error, TypeError, ValueError) as error:
            raise IntelligenceCacheError("Cannot write to the AI cache") from error

    def invalidate(self, namespace: str | None = None) -> int:
        """Delete one namespace or the complete cache."""

        try:
            with self._lock:
                if namespace is None:
                    cursor = self._connection.execute("DELETE FROM inference_cache")
                else:
                    cursor = self._connection.execute(
                        "DELETE FROM inference_cache WHERE namespace = ?", (namespace,)
                    )
                self._connection.commit()
                return cursor.rowcount
        except sqlite3.Error as error:
            raise IntelligenceCacheError("Cannot invalidate the AI cache") from error

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self._lock:
            self._connection.close()
