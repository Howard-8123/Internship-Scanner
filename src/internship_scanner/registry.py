"""Provider registry and company discovery catalog."""

from collections.abc import Iterable

from internship_scanner.models import Company
from internship_scanner.providers.base import ATSProvider


class ProviderRegistry:
    """Lookup of provider plugins without provider-specific branching."""

    def __init__(self, providers: Iterable[ATSProvider] = ()) -> None:
        self._providers: dict[str, ATSProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ATSProvider) -> None:
        """Register a uniquely named provider plugin."""

        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ATSProvider:
        """Return a provider by stable name."""

        try:
            return self._providers[name]
        except KeyError as error:
            raise ValueError(f"Provider is not registered: {name}") from error

    def all(self) -> tuple[ATSProvider, ...]:
        """Return registered providers in deterministic registration order."""

        return tuple(self._providers.values())


class CompanyCatalog:
    """Discover and deduplicate companies across provider plugins."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def discover(self) -> tuple[Company, ...]:
        """Return provider-scoped unique discovered companies."""

        companies = (
            company
            for provider in self._registry.all()
            for company in provider.discover_companies()
        )
        return tuple({company.key: company for company in companies}.values())
