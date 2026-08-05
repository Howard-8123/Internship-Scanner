"""Validated models passed between intelligence pipeline stages."""

from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from internship_scanner.models import Job


class JobAnalysis(BaseModel):
    """Strict, provider-neutral structured analysis returned by an LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    technologies: tuple[str, ...]
    programming_languages: tuple[str, ...]
    responsibilities: tuple[str, ...]
    required_experience: tuple[str, ...]
    preferred_experience: tuple[str, ...]
    is_internship: bool
    is_technical: bool
    internship_relevance: float = Field(ge=0.0, le=1.0)
    ai_relevance_score: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "labels",
        "technologies",
        "programming_languages",
        "responsibilities",
        "required_experience",
        "preferred_experience",
        mode="after",
    )
    @classmethod
    def normalize_string_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Trim and deduplicate string arrays while retaining their order."""

        normalized = tuple(
            dict.fromkeys(value.strip() for value in values if value.strip())
        )
        if not normalized and values:
            raise ValueError("string arrays must not contain only blank values")
        return normalized

    @field_validator("reasoning", mode="after")
    @classmethod
    def normalize_reasoning(cls, value: str) -> str:
        """Reject a reasoning value containing whitespace only."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("reasoning must not be blank")
        return normalized


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """Cosine-similarity output for one normalized job document."""

    label_scores: tuple[tuple[str, float], ...]
    semantic_similarity: float
    internship_similarity: float
    permanent_similarity: float
    technical_similarity: float
    nontechnical_similarity: float
    ai_similarity: float

    @property
    def internship_margin(self) -> float:
        """Return positive internship evidence relative to permanent-role evidence."""

        return self.internship_similarity - self.permanent_similarity

    @property
    def technical_margin(self) -> float:
        """Return hands-on technical evidence relative to business-role evidence."""

        return self.technical_similarity - self.nontechnical_similarity

    def top_labels(self, threshold: float, limit: int) -> tuple[str, ...]:
        """Return at most ``limit`` labels meeting the absolute score threshold."""

        return tuple(label for label, score in self.label_scores if score >= threshold)[
            :limit
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for persistent caching."""

        return {
            "label_scores": [list(item) for item in self.label_scores],
            "semantic_similarity": self.semantic_similarity,
            "internship_similarity": self.internship_similarity,
            "permanent_similarity": self.permanent_similarity,
            "technical_similarity": self.technical_similarity,
            "nontechnical_similarity": self.nontechnical_similarity,
            "ai_similarity": self.ai_similarity,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Restore and validate a cached semantic result."""

        scores_value = value["label_scores"]
        if not isinstance(scores_value, list):
            raise ValueError("label_scores must be a list")
        scores: list[tuple[str, float]] = []
        for item in scores_value:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("invalid cached label score")
            scores.append((str(item[0]), _score(item[1])))
        return cls(
            label_scores=tuple(scores),
            semantic_similarity=_score(value["semantic_similarity"]),
            internship_similarity=_score(value["internship_similarity"]),
            permanent_similarity=_score(value["permanent_similarity"]),
            technical_similarity=_score(value["technical_similarity"]),
            nontechnical_similarity=_score(value["nontechnical_similarity"]),
            ai_similarity=_score(value["ai_similarity"]),
        )


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Ranked job recommendation emitted by the complete AI pipeline."""

    job: Job
    score: float
    semantic: SemanticMatch
    analysis: JobAnalysis | None
    labels: tuple[str, ...]
    used_keyword_fallback: bool = False


def _score(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise ValueError("score must be numeric")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between zero and one")
    return score
