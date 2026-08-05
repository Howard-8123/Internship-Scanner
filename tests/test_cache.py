"""Tests for persistent caching and the provider decorator."""

from collections.abc import Callable
from pathlib import Path

from internship_scanner.cache import CachedProvider, JsonTTLCache
from internship_scanner.models import Company, Job
from internship_scanner.providers.base import ATSProvider


class CountingProvider(ATSProvider):
    def __init__(self, company: Company, jobs: list[Job]) -> None:
        self.company = company
        self.jobs = jobs

    @property
    def name(self) -> str:
        return "test"

    @property
    def discovery_cache_key(self) -> str:
        return f"test:{self.company.identifier}"

    def discover_companies(self) -> list[Company]:
        return [self.company]

    def fetch_jobs(self, company: Company) -> list[Job]:
        return self.jobs


def test_json_cache_persists_and_expires(tmp_path: Path) -> None:
    now = 100.0
    cache = JsonTTLCache(tmp_path / "cache.json", 10, lambda: now)
    assert cache.get("missing") is None
    cache.set("key", {"value": 1})
    assert cache.get("key") == {"value": 1}
    expired = JsonTTLCache(tmp_path / "cache.json", 10, lambda: 111.0)
    assert expired.get("key") is None


def test_cache_recovers_from_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("not json", encoding="utf-8")
    cache = JsonTTLCache(path, 10)
    assert cache.get("key") is None


def test_cached_provider_avoids_repeated_provider_calls(
    tmp_path: Path, job_factory: Callable[..., Job]
) -> None:
    company = Company("Example", "example", "test")
    provider = CountingProvider(company, [job_factory(provider="test")])
    cached = CachedProvider(provider, JsonTTLCache(tmp_path / "cache.json", 60))
    assert cached.discover_companies() == [company]
    assert cached.discover_companies() == [company]
    assert cached.fetch_jobs(company) == provider.jobs
    provider.jobs = []
    assert len(cached.fetch_jobs(company)) == 1


def test_cached_provider_recovers_from_invalid_typed_entries(
    tmp_path: Path, job_factory: Callable[..., Job]
) -> None:
    company = Company("Example", "example", "test")
    provider = CountingProvider(company, [job_factory(provider="test")])
    cache = JsonTTLCache(tmp_path / "cache.json", 60)
    cache.set("v2:companies:test:example", [{"invalid": True}])
    cache.set("v2:jobs:test:example", [{"invalid": True}])
    cached = CachedProvider(provider, cache)
    assert cached.discover_companies() == [company]
    assert cached.fetch_jobs(company) == provider.jobs
