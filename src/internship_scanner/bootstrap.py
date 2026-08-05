"""Application composition root for built-in provider plugins."""

from collections.abc import Callable

from internship_scanner.aggregation import JobAggregationEngine
from internship_scanner.cache import CachedProvider, JsonTTLCache
from internship_scanner.config import Settings
from internship_scanner.exceptions import ConfigurationError
from internship_scanner.http import HttpClient
from internship_scanner.intelligence.cache import SQLiteInferenceCache
from internship_scanner.intelligence.embeddings import (
    BGEEmbeddingProvider,
    CachedEmbeddingProvider,
)
from internship_scanner.intelligence.evaluation import LiveLLMQualityEvaluator
from internship_scanner.intelligence.llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    StructuredJobAnalyzer,
)
from internship_scanner.intelligence.llm.http import RequestsLLMTransport
from internship_scanner.intelligence.pipeline import (
    HybridIntelligencePipeline,
    JobRecommender,
    KeywordFallbackRecommender,
)
from internship_scanner.intelligence.ranking import (
    RankingWeights,
    WeightedRankingEngine,
)
from internship_scanner.intelligence.semantic import (
    CosineSimilaritySearch,
    profiles_from_labels,
)
from internship_scanner.models import Company
from internship_scanner.providers import (
    AshbyProvider,
    GreenhouseProvider,
    LeverProvider,
    PersonioProvider,
    RecruiteeProvider,
    SmartRecruitersProvider,
    WorkableProvider,
)
from internship_scanner.providers.base import ATSProvider
from internship_scanner.registry import CompanyCatalog, ProviderRegistry

ProviderFactory = Callable[[list[Company], HttpClient], ATSProvider]
PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "greenhouse": GreenhouseProvider,
    "lever": LeverProvider,
    "ashby": AshbyProvider,
    "smartrecruiters": SmartRecruitersProvider,
    "workable": WorkableProvider,
    "recruitee": RecruiteeProvider,
    "personio": PersonioProvider,
}


def build_engine(settings: Settings) -> JobAggregationEngine:
    """Compose configured providers, persistent caching, registry, and engine."""

    supported = set(PROVIDER_FACTORIES)
    configured = {company.provider for company in settings.companies}
    unsupported = configured - supported
    if unsupported:
        raise ConfigurationError(
            f"Unsupported configured providers: {', '.join(sorted(unsupported))}"
        )

    http = HttpClient(settings.request_timeout_seconds)
    cache = JsonTTLCache(settings.cache_path, settings.cache_ttl_seconds)
    providers = [
        CachedProvider(factory(list(settings.companies), http), cache)
        for name, factory in PROVIDER_FACTORIES.items()
        if name in configured
    ]
    registry = ProviderRegistry(providers)
    return JobAggregationEngine(
        registry,
        CompanyCatalog(registry),
        recommender=build_recommender(settings),
    )


def build_recommender(settings: Settings) -> JobRecommender:
    """Compose the model-independent intelligence pipeline from configuration."""

    fallback = KeywordFallbackRecommender()
    if not settings.ai_enabled:
        return fallback

    cache, semantic_search = _build_semantic_search(settings)
    analyzer = None
    if settings.llm_provider != "none":
        analyzer = StructuredJobAnalyzer(
            _build_llm_provider(settings),
            cache,
            settings.ai_cache_version,
            settings.ai_labels,
        )
    ranking = WeightedRankingEngine(
        RankingWeights(
            semantic_similarity=settings.ranking_semantic_weight,
            llm_confidence=settings.ranking_confidence_weight,
            ai_relevance=settings.ranking_ai_weight,
            internship_relevance=settings.ranking_internship_weight,
        )
    )
    return HybridIntelligencePipeline(
        semantic_search,
        ranking,
        analyzer=analyzer,
        semantic_threshold=settings.semantic_threshold,
        internship_margin_threshold=settings.internship_margin_threshold,
        technical_margin_threshold=settings.technical_margin_threshold,
        label_threshold=settings.label_threshold,
        max_labels=settings.max_labels,
        llm_confidence_threshold=settings.llm_confidence_threshold,
        llm_internship_threshold=settings.llm_internship_threshold,
        recommendation_threshold=settings.recommendation_threshold,
        fallback=fallback,
    )


def build_quality_evaluator(settings: Settings) -> LiveLLMQualityEvaluator:
    """Compose a live evaluator and reject semantic-only configurations."""

    if not settings.ai_enabled or settings.llm_provider == "none":
        raise ConfigurationError(
            "Live quality evaluation requires AI_ENABLED=true and an LLM_PROVIDER"
        )
    cache, semantic_search = _build_semantic_search(settings)
    analyzer = StructuredJobAnalyzer(
        _build_llm_provider(settings),
        cache,
        settings.ai_cache_version,
        settings.ai_labels,
    )
    return LiveLLMQualityEvaluator(
        semantic_search,
        analyzer,
        confidence_threshold=settings.llm_confidence_threshold,
        internship_threshold=settings.llm_internship_threshold,
    )


def _build_semantic_search(
    settings: Settings,
) -> tuple[SQLiteInferenceCache, CosineSimilaritySearch]:
    cache = SQLiteInferenceCache(settings.ai_cache_path)
    embeddings = CachedEmbeddingProvider(
        BGEEmbeddingProvider(settings.embedding_model),
        cache,
        settings.ai_cache_version,
    )
    semantic_search = CosineSimilaritySearch(
        embeddings,
        profiles_from_labels(settings.ai_labels),
        cache,
        settings.ai_cache_version,
    )
    return cache, semantic_search


def _build_llm_provider(settings: Settings) -> LLMProvider:
    transport = RequestsLLMTransport(settings.request_timeout_seconds)
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key or ""
    if settings.llm_provider == "openai":
        return OpenAIProvider(
            settings.llm_model,
            api_key,
            transport,
            base_url or "https://api.openai.com/v1",
        )
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            settings.llm_model,
            api_key,
            transport,
            base_url or "https://api.anthropic.com/v1",
        )
    if settings.llm_provider == "gemini":
        return GeminiProvider(
            settings.llm_model,
            api_key,
            transport,
            base_url or "https://generativelanguage.googleapis.com/v1beta",
        )
    if settings.llm_provider == "openrouter":
        return OpenRouterProvider(
            settings.llm_model,
            api_key,
            transport,
            base_url or "https://openrouter.ai/api/v1",
        )
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            settings.llm_model,
            transport,
            base_url or "http://localhost:11434",
        )
    raise ConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")
