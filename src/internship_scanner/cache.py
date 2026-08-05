"""Persistent TTL cache and provider caching decorator."""

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from internship_scanner.models import Company, Job
from internship_scanner.providers.base import ATSProvider

LOGGER = logging.getLogger(__name__)


class JsonTTLCache:
    """Thread-safe JSON file cache with per-entry expiry timestamps."""

    def __init__(
        self,
        path: Path,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return a non-expired cached value or ``None``."""

        with self._lock:
            entries = self._read()
            entry = entries.get(key)
            if (
                not isinstance(entry, dict)
                or entry.get("expires_at", 0) <= self._clock()
            ):
                return None
            return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key`` until the configured expiry."""

        with self._lock:
            entries = self._read()
            entries[key] = {
                "expires_at": self._clock() + self._ttl_seconds,
                "value": value,
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
            temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            temporary.replace(self._path)

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Ignoring unreadable cache %s: %s", self._path, error)
            return {}


class CachedProvider(ATSProvider):
    """Provider-neutral decorator caching discovery and normalized jobs."""

    def __init__(self, provider: ATSProvider, cache: JsonTTLCache) -> None:
        self._provider = provider
        self._cache = cache

    @property
    def name(self) -> str:
        """Return the wrapped provider name."""

        return self._provider.name

    @property
    def discovery_cache_key(self) -> str:
        """Return the wrapped provider's discovery identity."""

        return self._provider.discovery_cache_key

    def discover_companies(self) -> list[Company]:
        """Return cached or freshly discovered companies."""

        key = f"v2:companies:{self._provider.discovery_cache_key}"
        cached = self._cache.get(key)
        if isinstance(cached, list):
            try:
                return [Company.from_dict(item) for item in cached]
            except (KeyError, TypeError, ValueError):
                LOGGER.warning("Ignoring invalid cached companies for %s", self.name)
        companies = self._provider.discover_companies()
        self._cache.set(key, [company.to_dict() for company in companies])
        return companies

    def fetch_jobs(self, company: Company) -> list[Job]:
        """Return cached or freshly fetched normalized jobs."""

        key = f"v2:jobs:{company.key}"
        cached = self._cache.get(key)
        if isinstance(cached, list):
            try:
                return [Job.from_dict(item) for item in cached]
            except (KeyError, TypeError, ValueError):
                LOGGER.warning("Ignoring invalid cached jobs for %s", company.key)
        jobs = self._provider.fetch_jobs(company)
        self._cache.set(key, [job.to_dict() for job in jobs])
        return jobs
