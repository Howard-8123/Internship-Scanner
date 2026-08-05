"""Contract shared by every independent ATS provider."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from internship_scanner.models import Company, Job


class ATSProvider(ABC):
    """Provider plugin capable of company discovery and normalized job retrieval."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable provider identifier."""

    @property
    @abstractmethod
    def discovery_cache_key(self) -> str:
        """Return a key that changes when discovery inputs change."""

    @abstractmethod
    def discover_companies(self) -> list[Company]:
        """Discover configured public company boards supported by this provider."""

    @abstractmethod
    def fetch_jobs(self, company: Company) -> list[Job]:
        """Fetch and normalize published jobs for ``company``."""


class ConfiguredProvider(ATSProvider):
    """Base for ATSs that address boards using public company identifiers."""

    def __init__(self, companies: Iterable[Company]) -> None:
        self._companies = tuple(
            company for company in companies if company.provider == self.name
        )

    def discover_companies(self) -> list[Company]:
        """Return configured boards, deduplicated by provider and identifier."""

        return list({company.key: company for company in self._companies}.values())

    @property
    def discovery_cache_key(self) -> str:
        """Include configured board identifiers in discovery cache identity."""

        identifiers = ",".join(sorted(company.key for company in self._companies))
        return f"{self.name}:{identifiers}"
