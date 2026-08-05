"""Independent, configurable recommendation ranking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from internship_scanner.intelligence.models import JobAnalysis, SemanticMatch


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Relative contribution of each normalized ranking signal."""

    semantic_similarity: float = 0.35
    llm_confidence: float = 0.20
    ai_relevance: float = 0.20
    internship_relevance: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.semantic_similarity,
            self.llm_confidence,
            self.ai_relevance,
            self.internship_relevance,
        )
        if any(value < 0.0 for value in values) or sum(values) <= 0.0:
            raise ValueError("ranking weights must be non-negative with a positive sum")


class RankingEngine(ABC):
    """Port for independently replacing recommendation scoring."""

    @abstractmethod
    def score(self, semantic: SemanticMatch, analysis: JobAnalysis | None) -> float:
        """Return a normalized recommendation score."""


class WeightedRankingEngine(RankingEngine):
    """Blend semantic and structured-analysis signals using injected weights."""

    def __init__(self, weights: RankingWeights) -> None:
        self._weights = weights

    def score(self, semantic: SemanticMatch, analysis: JobAnalysis | None) -> float:
        """Calculate a weighted average, using semantic signals without an LLM."""

        confidence = analysis.confidence if analysis else 0.0
        ai_relevance = (
            analysis.ai_relevance_score if analysis else semantic.ai_similarity
        )
        internship_relevance = (
            analysis.internship_relevance
            if analysis
            else semantic.internship_similarity
        )
        numerator = (
            semantic.semantic_similarity * self._weights.semantic_similarity
            + confidence * self._weights.llm_confidence
            + ai_relevance * self._weights.ai_relevance
            + internship_relevance * self._weights.internship_relevance
        )
        denominator = (
            self._weights.semantic_similarity
            + self._weights.llm_confidence
            + self._weights.ai_relevance
            + self._weights.internship_relevance
        )
        return min(1.0, max(0.0, numerator / denominator))
