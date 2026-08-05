"""Tests for registry, discovery, aggregation, and deduplication."""

from collections.abc import Callable, Sequence

import pytest

from internship_scanner.aggregation import JobAggregationEngine
from internship_scanner.exceptions import JobSourceError
from internship_scanner.intelligence.models import Recommendation, SemanticMatch
from internship_scanner.intelligence.pipeline import JobRecommender
from internship_scanner.models import Company, InternshipType, Job
from internship_scanner.providers.base import ATSProvider
from internship_scanner.registry import CompanyCatalog, ProviderRegistry


class StubProvider(ATSProvider):
    """Configurable provider test double."""

    def __init__(
        self, name: str, companies: list[Company], jobs: list[Job] | None = None
    ) -> None:
        self._name = name
        self.companies = companies
        self.jobs = jobs or []
        self.failure: JobSourceError | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def discovery_cache_key(self) -> str:
        return self._name

    def discover_companies(self) -> list[Company]:
        return self.companies

    def fetch_jobs(self, company: Company) -> list[Job]:
        if self.failure:
            raise self.failure
        return self.jobs


class AcceptingRecommender(JobRecommender):
    """Recommender proving aggregation no longer prefilters internship keywords."""

    def recommend(self, jobs: Sequence[Job]) -> tuple[Recommendation, ...]:
        semantic = SemanticMatch((("Backend", 1.0),), 1.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        return tuple(
            Recommendation(job, 1.0, semantic, None, ("Backend",)) for job in jobs
        )


def test_registry_rejects_duplicates_and_unknown_provider() -> None:
    provider = StubProvider("test", [])
    registry = ProviderRegistry([provider])
    assert registry.get("test") is provider
    with pytest.raises(ValueError, match="already registered"):
        registry.register(provider)
    with pytest.raises(ValueError, match="not registered"):
        registry.get("missing")


def test_catalog_deduplicates_provider_scoped_companies() -> None:
    company = Company("Example", "example", "one")
    other = Company("Example", "example", "two")
    registry = ProviderRegistry(
        [StubProvider("one", [company, company]), StubProvider("two", [other])]
    )
    assert CompanyCatalog(registry).discover() == (company, other)


def test_engine_filters_locations_internships_and_deduplicates(
    job_factory: Callable[..., Job],
) -> None:
    company_one = Company("Example", "one", "one")
    company_two = Company("Example", "two", "two")
    less_complete = job_factory(
        provider="one",
        company_identifier="one",
        description="",
        location="London, UK",
    )
    more_complete = job_factory(
        provider="two",
        company_identifier="two",
        company="Example Ltd",
        location="London",
    )
    hong_kong = job_factory(
        id="2",
        title="Data Placement",
        provider="one",
        company_identifier="one",
        location="Hong Kong",
        country="HK",
        internship_type=InternshipType.PLACEMENT,
    )
    rejected_country = job_factory(
        id="3", location="New York", country="US", provider="one"
    )
    rejected_permanent = job_factory(
        id="4",
        title="Software Engineer",
        internship_type=InternshipType.UNKNOWN,
        provider="one",
    )
    providers = [
        StubProvider(
            "one",
            [company_one],
            [less_complete, hong_kong, rejected_country, rejected_permanent],
        ),
        StubProvider("two", [company_two], [more_complete]),
    ]
    registry = ProviderRegistry(providers)
    result = JobAggregationEngine(registry, CompanyCatalog(registry)).aggregate()
    assert result.downloaded_job_count == 5
    assert len(result.internships) == 2
    assert any(job.provider == "two" for job in result.internships)
    assert {job.country for job in result.internships} == {"United Kingdom", "HK"}


def test_engine_isolates_provider_failures() -> None:
    company = Company("Broken", "broken", "test")
    provider = StubProvider("test", [company])
    provider.failure = JobSourceError("unavailable")
    registry = ProviderRegistry([provider])
    result = JobAggregationEngine(registry, CompanyCatalog(registry)).aggregate()
    assert result.downloaded_job_count == 0
    assert result.failures[0].company == company


def test_engine_delegates_unknown_job_types_to_injected_intelligence(
    job_factory: Callable[..., Job],
) -> None:
    company = Company("Example", "example", "test")
    job = job_factory(
        title="Student Technology Programme",
        internship_type=InternshipType.UNKNOWN,
        provider="test",
    )
    provider = StubProvider("test", [company], [job])
    registry = ProviderRegistry([provider])
    result = JobAggregationEngine(
        registry, CompanyCatalog(registry), AcceptingRecommender()
    ).aggregate()
    assert result.internships == (job,)
    assert result.recommendations[0].labels == ("Backend",)
