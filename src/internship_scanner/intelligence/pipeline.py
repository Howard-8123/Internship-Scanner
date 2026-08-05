"""Provider-independent orchestration of the hybrid AI stages."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from internship_scanner.classification import classify_role, is_internship
from internship_scanner.exceptions import IntelligenceError
from internship_scanner.intelligence.llm.analyzer import JobAnalyzer
from internship_scanner.intelligence.models import (
    JobAnalysis,
    Recommendation,
    SemanticMatch,
)
from internship_scanner.intelligence.ranking import RankingEngine
from internship_scanner.intelligence.semantic import VectorSimilaritySearch
from internship_scanner.intelligence.text import build_job_document
from internship_scanner.models import (
    TECHNICAL_CATEGORIES,
    InternshipType,
    Job,
    RoleCategory,
)

LOGGER = logging.getLogger(__name__)


class JobRecommender(ABC):
    """Application-facing port for any job intelligence implementation."""

    @abstractmethod
    def recommend(self, jobs: Sequence[Job]) -> tuple[Recommendation, ...]:
        """Return ranked recommendations for normalized jobs."""


class HybridIntelligencePipeline(JobRecommender):
    """Compose semantic retrieval, optional LLM analysis, and ranking."""

    def __init__(
        self,
        semantic_search: VectorSimilaritySearch,
        ranking: RankingEngine,
        *,
        analyzer: JobAnalyzer | None,
        semantic_threshold: float,
        internship_margin_threshold: float,
        technical_margin_threshold: float,
        label_threshold: float,
        max_labels: int,
        llm_confidence_threshold: float,
        llm_internship_threshold: float,
        recommendation_threshold: float,
        fallback: JobRecommender,
    ) -> None:
        for name, value in (
            ("semantic_threshold", semantic_threshold),
            ("label_threshold", label_threshold),
            ("recommendation_threshold", recommendation_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        self._semantic_search = semantic_search
        self._ranking = ranking
        self._analyzer = analyzer
        self._semantic_threshold = semantic_threshold
        self._internship_margin_threshold = internship_margin_threshold
        self._technical_margin_threshold = technical_margin_threshold
        self._label_threshold = label_threshold
        if max_labels <= 0:
            raise ValueError("max_labels must be positive")
        self._max_labels = max_labels
        self._llm_confidence_threshold = llm_confidence_threshold
        self._llm_internship_threshold = llm_internship_threshold
        self._recommendation_threshold = recommendation_threshold
        self._fallback = fallback

    def recommend(self, jobs: Sequence[Job]) -> tuple[Recommendation, ...]:
        """Run every job through independent stages and sort accepted results."""

        recommendations: list[Recommendation] = []
        for job in jobs:
            try:
                recommendation = self._recommend_one(job)
            except IntelligenceError as error:
                LOGGER.warning(
                    "AI inference failed for %s; using keyword fallback: %s",
                    job.url,
                    error,
                )
                recommendations.extend(self._fallback.recommend([job]))
                continue
            if recommendation is not None:
                recommendations.append(recommendation)
        recommendations.sort(key=lambda item: item.score, reverse=True)
        return tuple(recommendations)

    def _recommend_one(self, job: Job) -> Recommendation | None:
        document = build_job_document(job)
        semantic = self._semantic_search.search(document)
        if (
            semantic.internship_similarity < self._semantic_threshold
            or semantic.internship_margin < self._internship_margin_threshold
            or semantic.technical_margin < self._technical_margin_threshold
        ):
            return None

        analysis: JobAnalysis | None = None
        if self._analyzer is not None:
            analysis = self._analyzer.analyze(document, semantic)
            if (
                not analysis.is_internship
                or not analysis.is_technical
                or analysis.confidence < self._llm_confidence_threshold
                or analysis.internship_relevance < self._llm_internship_threshold
            ):
                return None
        score = self._ranking.score(semantic, analysis)
        if score < self._recommendation_threshold:
            return None

        labels = (
            analysis.labels[: self._max_labels]
            if analysis is not None
            else semantic.top_labels(self._label_threshold, self._max_labels)
        )
        if not labels:
            return None
        category = _legacy_category(labels)
        return Recommendation(
            job=job.with_category(category),
            score=score,
            semantic=semantic,
            analysis=analysis,
            labels=labels,
        )


class KeywordFallbackRecommender(JobRecommender):
    """Final recovery path retaining the prototype's deterministic title rules."""

    def recommend(self, jobs: Sequence[Job]) -> tuple[Recommendation, ...]:
        """Recommend known technical internships when AI inference is unavailable."""

        recommendations: list[Recommendation] = []
        for job in jobs:
            if job.internship_type is InternshipType.UNKNOWN and not is_internship(job):
                continue
            category = classify_role(job)
            if category not in TECHNICAL_CATEGORIES:
                continue
            labels = _fallback_labels(category)
            semantic = SemanticMatch(
                label_scores=tuple((label, 0.5) for label in labels),
                semantic_similarity=0.5,
                internship_similarity=1.0,
                permanent_similarity=0.0,
                technical_similarity=1.0,
                nontechnical_similarity=0.0,
                ai_similarity=0.5 if category is RoleCategory.AI_ML else 0.0,
            )
            recommendations.append(
                Recommendation(
                    job=job.with_category(category),
                    score=0.5,
                    semantic=semantic,
                    analysis=None,
                    labels=labels,
                    used_keyword_fallback=True,
                )
            )
        return tuple(recommendations)


def _legacy_category(labels: tuple[str, ...]) -> RoleCategory:
    label_set = set(labels)
    if label_set & {
        "AI",
        "Machine Learning",
        "Deep Learning",
        "LLMs",
        "NLP",
        "Computer Vision",
        "Robotics",
    }:
        return RoleCategory.AI_ML
    if "Research" in label_set:
        return RoleCategory.RESEARCH_ENGINEERING
    if "Data Engineering" in label_set:
        return RoleCategory.DATA
    if label_set & {"Cloud", "Infrastructure", "Security", "DevOps", "Embedded"}:
        return RoleCategory.INFRASTRUCTURE_SECURITY
    if label_set:
        return RoleCategory.SOFTWARE_ENGINEERING
    return RoleCategory.OTHER


def _fallback_labels(category: RoleCategory) -> tuple[str, ...]:
    return {
        RoleCategory.AI_ML: ("AI", "Machine Learning"),
        RoleCategory.RESEARCH_ENGINEERING: ("Research",),
        RoleCategory.SOFTWARE_ENGINEERING: ("General Software Engineering",),
        RoleCategory.INFRASTRUCTURE_SECURITY: ("Infrastructure", "Security"),
        RoleCategory.DATA: ("Data Engineering",),
    }.get(category, ())
