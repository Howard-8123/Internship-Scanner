"""Tests for configuration-only AI provider composition."""

from pathlib import Path

import pytest

from internship_scanner.bootstrap import (
    _build_llm_provider,
    build_quality_evaluator,
    build_recommender,
)
from internship_scanner.config import Settings
from internship_scanner.exceptions import ConfigurationError
from internship_scanner.intelligence.llm import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from internship_scanner.intelligence.pipeline import (
    HybridIntelligencePipeline,
    KeywordFallbackRecommender,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("gemini", GeminiProvider),
        ("openrouter", OpenRouterProvider),
        ("ollama", OllamaProvider),
    ],
)
def test_llm_provider_is_selected_only_by_configuration(
    name: str, expected: type[object]
) -> None:
    settings = Settings(
        llm_provider=name,
        llm_model="model-id",
        llm_api_key="key",
        llm_base_url="https://compatible.example/v1",
    )
    provider = _build_llm_provider(settings)
    assert isinstance(provider, expected)
    assert "model-id" in provider.identity


def test_recommender_switches_between_hybrid_and_fallback(tmp_path: Path) -> None:
    disabled = build_recommender(Settings(ai_enabled=False))
    assert isinstance(disabled, KeywordFallbackRecommender)

    enabled = build_recommender(
        Settings(ai_enabled=True, ai_cache_path=tmp_path / "ai.sqlite3")
    )
    assert isinstance(enabled, HybridIntelligencePipeline)


def test_llm_factory_rejects_unknown_manual_settings() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
        _build_llm_provider(
            Settings(llm_provider="unknown", llm_model="model", llm_api_key="key")
        )


def test_live_evaluator_rejects_disabled_llm() -> None:
    with pytest.raises(ConfigurationError, match="requires"):
        build_quality_evaluator(Settings(llm_provider="none"))
