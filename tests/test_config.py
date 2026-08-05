"""Tests for Phase 2 environment configuration."""

from pathlib import Path

import pytest

from internship_scanner.bootstrap import build_engine
from internship_scanner.config import Settings
from internship_scanner.exceptions import ConfigurationError
from internship_scanner.models import Company


def clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ATS_COMPANIES",
        "BOARD_TOKEN",
        "COMPANY_NAME",
        "REQUEST_TIMEOUT_SECONDS",
        "CACHE_TTL_SECONDS",
        "CACHE_PATH",
        "LOG_LEVEL",
        "AI_ENABLED",
        "AI_CACHE_PATH",
        "AI_CACHE_VERSION",
        "EMBEDDING_MODEL",
        "SEMANTIC_THRESHOLD",
        "INTERNSHIP_MARGIN_THRESHOLD",
        "TECHNICAL_MARGIN_THRESHOLD",
        "LABEL_THRESHOLD",
        "MAX_LABELS",
        "LLM_CONFIDENCE_THRESHOLD",
        "LLM_INTERNSHIP_THRESHOLD",
        "RECOMMENDATION_THRESHOLD",
        "AI_LABELS",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "RANKING_SEMANTIC_WEIGHT",
        "RANKING_CONFIDENCE_WEIGHT",
        "RANKING_AI_WEIGHT",
        "RANKING_INTERNSHIP_WEIGHT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_preserve_cloudflare_greenhouse_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    assert Settings.from_env() == Settings()


def test_parses_and_deduplicates_multiple_provider_companies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv(
        "ATS_COMPANIES",
        "lever|one|One Ltd;ashby|two|Two Ltd;lever|one|One Ltd",
    )
    settings = Settings.from_env()
    assert settings.companies == (
        Company("One Ltd", "one", "lever"),
        Company("Two Ltd", "two", "ashby"),
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("REQUEST_TIMEOUT_SECONDS", "never", "must be a number"),
        ("CACHE_TTL_SECONDS", "0", "must be positive"),
        ("LOG_LEVEL", "verbose", "LOG_LEVEL is not valid"),
        ("ATS_COMPANIES", "invalid", "must use"),
    ],
)
def test_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env()


def test_build_engine_rejects_unimplemented_provider() -> None:
    settings = Settings(
        companies=(Company("Unknown", "unknown", "future-ats"),),
        cache_path=Path("unused"),
    )
    with pytest.raises(ConfigurationError, match="Unsupported configured providers"):
        build_engine(settings)


def test_rejects_blank_legacy_board_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv("BOARD_TOKEN", " ")
    with pytest.raises(ConfigurationError, match="requires provider"):
        Settings.from_env()


def test_build_engine_composes_supported_plugins(tmp_path: Path) -> None:
    engine = build_engine(
        Settings(
            cache_path=tmp_path / "cache.json",
            ai_cache_path=tmp_path / "ai.sqlite3",
        )
    )
    assert engine.__class__.__name__ == "JobAggregationEngine"


def test_parses_ai_and_openai_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv("AI_LABELS", "AI,Custom Label,AI")
    monkeypatch.setenv("SEMANTIC_THRESHOLD", "0.6")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    settings = Settings.from_env()
    assert settings.ai_labels == ("AI", "Custom Label")
    assert settings.semantic_threshold == 0.6
    assert settings.llm_model == "gpt-5.6-sol"
    assert settings.llm_api_key == "secret"
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("AI_ENABLED", "perhaps", "true or false"),
        ("SEMANTIC_THRESHOLD", "1.5", "between zero and one"),
        ("INTERNSHIP_MARGIN_THRESHOLD", "-2", "between -1.0 and 1.0"),
        ("MAX_LABELS", "0", "must be positive"),
        ("RANKING_AI_WEIGHT", "-1", "must not be negative"),
        ("LLM_PROVIDER", "unknown", "must be one of"),
    ],
)
def test_rejects_invalid_ai_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env()


def test_requires_model_and_key_for_remote_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ConfigurationError, match="LLM_MODEL"):
        Settings.from_env()
    monkeypatch.setenv("LLM_MODEL", "claude-model")
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        Settings.from_env()
