"""Provider-neutral job aggregation application service."""

import logging
from dataclasses import dataclass

from internship_scanner.deduplication import deduplicate_jobs
from internship_scanner.exceptions import JobSourceError
from internship_scanner.intelligence.models import Recommendation
from internship_scanner.intelligence.pipeline import (
    JobRecommender,
    KeywordFallbackRecommender,
)
from internship_scanner.models import Company, Job
from internship_scanner.normalization import normalize_country
from internship_scanner.registry import CompanyCatalog, ProviderRegistry

LOGGER = logging.getLogger(__name__)
TARGET_COUNTRIES = frozenset({"United Kingdom", "Hong Kong"})


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A provider/company failure isolated during aggregation."""

    company: Company
    message: str


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Aggregate counts, normalized internships, and isolated failures."""

    discovered_company_count: int
    downloaded_job_count: int
    internships: tuple[Job, ...]
    failures: tuple[ProviderFailure, ...]
    recommendations: tuple[Recommendation, ...] = ()


class JobAggregationEngine:
    """Aggregate, filter, classify, and deduplicate jobs from registered providers."""

    def __init__(
        self,
        registry: ProviderRegistry,
        catalog: CompanyCatalog,
        recommender: JobRecommender | None = None,
    ) -> None:
        self._registry = registry
        self._catalog = catalog
        self._recommender = recommender or KeywordFallbackRecommender()

    def aggregate(self) -> AggregationResult:
        """Run a scan while isolating failures to individual company boards."""

        companies = self._catalog.discover()
        downloaded: list[Job] = []
        failures: list[ProviderFailure] = []
        for company in companies:
            try:
                downloaded.extend(
                    self._registry.get(company.provider).fetch_jobs(company)
                )
            except JobSourceError as error:
                LOGGER.warning(
                    "%s failed for %s: %s", company.provider, company.name, error
                )
                failures.append(ProviderFailure(company, str(error)))

        target_jobs = [job for job in downloaded if _is_target_location(job)]
        unique = deduplicate_jobs(target_jobs)
        recommendations = self._recommender.recommend(unique)
        return AggregationResult(
            discovered_company_count=len(companies),
            downloaded_job_count=len(downloaded),
            internships=tuple(item.job for item in recommendations),
            failures=tuple(failures),
            recommendations=recommendations,
        )


def _is_target_location(job: Job) -> bool:
    country = normalize_country(job.country, job.location)
    return country in TARGET_COUNTRIES
