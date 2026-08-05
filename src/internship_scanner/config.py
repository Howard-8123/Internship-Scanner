"""Environment-backed application configuration."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from internship_scanner.exceptions import ConfigurationError
from internship_scanner.models import Company

DEFAULT_AI_LABELS = (
    "AI",
    "Machine Learning",
    "Deep Learning",
    "LLMs",
    "NLP",
    "Computer Vision",
    "Robotics",
    "Data Engineering",
    "Backend",
    "Frontend",
    "Full Stack",
    "Cloud",
    "Infrastructure",
    "Security",
    "DevOps",
    "Embedded",
    "Mobile",
    "Research",
    "General Software Engineering",
)
LLM_PROVIDERS = frozenset(
    {"none", "openai", "anthropic", "gemini", "openrouter", "ollama"}
)
API_KEY_ENVIRONMENTS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings loaded from ``.env`` and the environment."""

    companies: tuple[Company, ...] = (
        Company(name="Cloudflare", identifier="cloudflare", provider="greenhouse"),
    )
    request_timeout_seconds: float = 10.0
    cache_ttl_seconds: float = 3600.0
    cache_path: Path = Path(".cache/internship-scanner.json")
    log_level: str = "INFO"
    ai_enabled: bool = True
    ai_cache_path: Path = Path(".cache/intelligence.sqlite3")
    ai_cache_version: str = "2"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    semantic_threshold: float = 0.45
    internship_margin_threshold: float = -0.03
    technical_margin_threshold: float = 0.02
    label_threshold: float = 0.45
    max_labels: int = 3
    llm_confidence_threshold: float = 0.65
    llm_internship_threshold: float = 0.75
    recommendation_threshold: float = 0.40
    ai_labels: tuple[str, ...] = DEFAULT_AI_LABELS
    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str | None = None
    ranking_semantic_weight: float = 0.35
    ranking_confidence_weight: float = 0.20
    ranking_ai_weight: float = 0.20
    ranking_internship_weight: float = 0.25

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate settings, preserving Cloudflare as the default board."""

        load_dotenv()
        timeout = _positive_float("REQUEST_TIMEOUT_SECONDS", "10")
        cache_ttl = _positive_float("CACHE_TTL_SECONDS", "3600")
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigurationError(f"LOG_LEVEL is not valid: {log_level}")

        companies_value = os.getenv("ATS_COMPANIES", "").strip()
        companies = (
            _parse_companies(companies_value)
            if companies_value
            else (
                Company(
                    name=os.getenv("COMPANY_NAME", "Cloudflare").strip(),
                    identifier=os.getenv("BOARD_TOKEN", "cloudflare").strip(),
                    provider="greenhouse",
                ),
            )
        )
        if not companies or any(
            not company.name or not company.identifier or not company.provider
            for company in companies
        ):
            raise ConfigurationError(
                "Every ATS company requires provider, identifier, and name"
            )

        ai_enabled = _boolean("AI_ENABLED", "true")
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5").strip()
        ai_cache_version = os.getenv("AI_CACHE_VERSION", "2").strip()
        if ai_enabled and (not embedding_model or not ai_cache_version):
            raise ConfigurationError(
                "EMBEDDING_MODEL and AI_CACHE_VERSION must not be blank"
            )

        labels = _parse_labels(os.getenv("AI_LABELS", ""))
        llm_provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
        if llm_provider not in LLM_PROVIDERS:
            raise ConfigurationError(
                f"LLM_PROVIDER must be one of: {', '.join(sorted(LLM_PROVIDERS))}"
            )
        configured_model = os.getenv("LLM_MODEL", "").strip()
        llm_model = configured_model or (
            "gpt-5.6-sol" if llm_provider == "openai" else ""
        )
        if llm_provider != "none" and not llm_model:
            raise ConfigurationError("LLM_MODEL is required for the selected provider")
        key_environment = API_KEY_ENVIRONMENTS.get(llm_provider)
        llm_api_key = os.getenv("LLM_API_KEY", "").strip() or (
            os.getenv(key_environment, "").strip() if key_environment else ""
        )
        if key_environment and not llm_api_key:
            raise ConfigurationError(
                f"LLM_API_KEY or {key_environment} is required for {llm_provider}"
            )

        ranking_weights = (
            _nonnegative_float("RANKING_SEMANTIC_WEIGHT", "0.35"),
            _nonnegative_float("RANKING_CONFIDENCE_WEIGHT", "0.20"),
            _nonnegative_float("RANKING_AI_WEIGHT", "0.20"),
            _nonnegative_float("RANKING_INTERNSHIP_WEIGHT", "0.25"),
        )
        if sum(ranking_weights) <= 0.0:
            raise ConfigurationError("At least one ranking weight must be positive")

        return cls(
            companies=companies,
            request_timeout_seconds=timeout,
            cache_ttl_seconds=cache_ttl,
            cache_path=Path(os.getenv("CACHE_PATH", ".cache/internship-scanner.json")),
            log_level=log_level,
            ai_enabled=ai_enabled,
            ai_cache_path=Path(
                os.getenv("AI_CACHE_PATH", ".cache/intelligence.sqlite3")
            ),
            ai_cache_version=ai_cache_version,
            embedding_model=embedding_model,
            semantic_threshold=_unit_float("SEMANTIC_THRESHOLD", "0.45"),
            internship_margin_threshold=_bounded_float(
                "INTERNSHIP_MARGIN_THRESHOLD", "-0.03", -1.0, 1.0
            ),
            technical_margin_threshold=_bounded_float(
                "TECHNICAL_MARGIN_THRESHOLD", "0.02", -1.0, 1.0
            ),
            label_threshold=_unit_float("LABEL_THRESHOLD", "0.45"),
            max_labels=_positive_int("MAX_LABELS", "3"),
            llm_confidence_threshold=_unit_float("LLM_CONFIDENCE_THRESHOLD", "0.65"),
            llm_internship_threshold=_unit_float("LLM_INTERNSHIP_THRESHOLD", "0.75"),
            recommendation_threshold=_unit_float("RECOMMENDATION_THRESHOLD", "0.40"),
            ai_labels=labels,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key or None,
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            ranking_semantic_weight=ranking_weights[0],
            ranking_confidence_weight=ranking_weights[1],
            ranking_ai_weight=ranking_weights[2],
            ranking_internship_weight=ranking_weights[3],
        )


def _positive_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _nonnegative_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if value < 0:
        raise ConfigurationError(f"{name} must not be negative")
    return value


def _unit_float(name: str, default: str) -> float:
    value = _nonnegative_float(name, default)
    if value > 1:
        raise ConfigurationError(f"{name} must be between zero and one")
    return value


def _bounded_float(name: str, default: str, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _positive_int(name: str, default: str) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _boolean(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    return labels or DEFAULT_AI_LABELS


def _parse_companies(value: str) -> tuple[Company, ...]:
    companies: list[Company] = []
    for entry in value.split(";"):
        parts = [part.strip() for part in entry.split("|", 2)]
        if len(parts) != 3 or not all(parts):
            raise ConfigurationError(
                "ATS_COMPANIES entries must use provider|identifier|company name"
            )
        companies.append(Company(name=parts[2], identifier=parts[1], provider=parts[0]))
    return tuple({company.key: company for company in companies}.values())
