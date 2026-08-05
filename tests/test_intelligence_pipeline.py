"""Tests for hybrid orchestration and failure-only keyword fallback."""

from collections.abc import Callable, Sequence

from internship_scanner.exceptions import EmbeddingError, MalformedAnalysisError
from internship_scanner.intelligence.llm.analyzer import JobAnalyzer
from internship_scanner.intelligence.models import (
    JobAnalysis,
    Recommendation,
    SemanticMatch,
)
from internship_scanner.intelligence.pipeline import (
    HybridIntelligencePipeline,
    JobRecommender,
    KeywordFallbackRecommender,
)
from internship_scanner.intelligence.ranking import RankingEngine
from internship_scanner.intelligence.semantic import VectorSimilaritySearch
from internship_scanner.intelligence.text import JobDocument
from internship_scanner.models import InternshipType, Job, RoleCategory


class StubSearch(VectorSimilaritySearch):
    """Configurable semantic-search double."""

    def __init__(self, match: SemanticMatch) -> None:
        self.match = match
        self.failure: EmbeddingError | None = None

    def search(self, document: JobDocument) -> SemanticMatch:
        if self.failure:
            raise self.failure
        return self.match


class StubAnalyzer(JobAnalyzer):
    """Configurable structured-analysis double."""

    def __init__(self, result: JobAnalysis) -> None:
        self.result = result
        self.calls = 0
        self.failure: MalformedAnalysisError | None = None

    def analyze(self, document: JobDocument, semantic: SemanticMatch) -> JobAnalysis:
        self.calls += 1
        if self.failure:
            raise self.failure
        return self.result


class StubRanking(RankingEngine):
    """Ranking double returning an injected score."""

    def __init__(self, score: float) -> None:
        self.result = score

    def score(self, semantic: SemanticMatch, analysis: JobAnalysis | None) -> float:
        return self.result


class RecordingFallback(JobRecommender):
    """Fallback double that records jobs and returns a sentinel recommendation."""

    def __init__(self, recommendation: Recommendation) -> None:
        self.recommendation = recommendation
        self.calls: list[Job] = []

    def recommend(self, jobs: Sequence[Job]) -> tuple[Recommendation, ...]:
        self.calls.extend(jobs)
        return (self.recommendation,)


def make_analysis() -> JobAnalysis:
    return JobAnalysis(
        labels=("AI", "Backend"),
        confidence=0.9,
        reasoning="Build AI services",
        technologies=("PyTorch",),
        programming_languages=("Python",),
        responsibilities=("Build models",),
        required_experience=(),
        preferred_experience=(),
        is_internship=True,
        is_technical=True,
        internship_relevance=0.95,
        ai_relevance_score=0.9,
    )


def make_pipeline(
    search: StubSearch,
    ranking: StubRanking,
    analyzer: StubAnalyzer | None,
    fallback: JobRecommender,
) -> HybridIntelligencePipeline:
    return HybridIntelligencePipeline(
        search,
        ranking,
        analyzer=analyzer,
        semantic_threshold=0.5,
        internship_margin_threshold=0.0,
        technical_margin_threshold=0.0,
        label_threshold=0.6,
        max_labels=3,
        llm_confidence_threshold=0.6,
        llm_internship_threshold=0.7,
        recommendation_threshold=0.5,
        fallback=fallback,
    )


def test_hybrid_pipeline_uses_analysis_labels_and_sorts(
    job_factory: Callable[..., Job],
) -> None:
    semantic = SemanticMatch(
        (("Backend", 0.8), ("AI", 0.7)), 0.8, 0.9, 0.1, 0.9, 0.1, 0.7
    )
    search = StubSearch(semantic)
    analyzer = StubAnalyzer(make_analysis())
    fallback = KeywordFallbackRecommender()
    pipeline = make_pipeline(search, StubRanking(0.75), analyzer, fallback)
    one = job_factory(
        id="1", title="Student Engineer", internship_type=InternshipType.UNKNOWN
    )
    two = job_factory(
        id="2", title="Graduate Developer", internship_type=InternshipType.UNKNOWN
    )
    recommendations = pipeline.recommend([one, two])
    assert [item.score for item in recommendations] == [0.75, 0.75]
    assert recommendations[0].labels == ("AI", "Backend")
    assert recommendations[0].job.category is RoleCategory.AI_ML
    assert not recommendations[0].used_keyword_fallback
    assert analyzer.calls == 2


def test_hybrid_pipeline_skips_below_semantic_and_ranking_thresholds(
    job_factory: Callable[..., Job],
) -> None:
    job = job_factory()
    below_semantic = StubSearch(
        SemanticMatch((("AI", 0.9),), 0.9, 0.4, 0.1, 0.9, 0.1, 0.9)
    )
    analyzer = StubAnalyzer(make_analysis())
    pipeline = make_pipeline(
        below_semantic, StubRanking(1.0), analyzer, KeywordFallbackRecommender()
    )
    assert pipeline.recommend([job]) == ()
    assert analyzer.calls == 0

    accepted = StubSearch(SemanticMatch((("AI", 0.9),), 0.9, 0.9, 0.1, 0.9, 0.1, 0.9))
    pipeline = make_pipeline(
        accepted, StubRanking(0.4), analyzer, KeywordFallbackRecommender()
    )
    assert pipeline.recommend([job]) == ()


def test_pipeline_rejects_weak_relative_evidence_and_llm_vetoes(
    job_factory: Callable[..., Job],
) -> None:
    job = job_factory()
    weak_internship = SemanticMatch((("Backend", 0.9),), 0.9, 0.7, 0.8, 0.9, 0.1, 0.1)
    pipeline = make_pipeline(
        StubSearch(weak_internship),
        StubRanking(1.0),
        None,
        KeywordFallbackRecommender(),
    )
    assert pipeline.recommend([job]) == ()

    valid = SemanticMatch((("Backend", 0.9),), 0.9, 0.9, 0.1, 0.9, 0.1, 0.1)
    analyzer = StubAnalyzer(make_analysis().model_copy(update={"is_technical": False}))
    pipeline = make_pipeline(
        StubSearch(valid), StubRanking(1.0), analyzer, KeywordFallbackRecommender()
    )
    assert pipeline.recommend([job]) == ()
    assert analyzer.calls == 1


def test_pipeline_without_llm_uses_semantic_labels(
    job_factory: Callable[..., Job],
) -> None:
    semantic = SemanticMatch(
        (("Backend", 0.8), ("Mobile", 0.5)), 0.8, 0.9, 0.1, 0.9, 0.1, 0.1
    )
    pipeline = make_pipeline(
        StubSearch(semantic),
        StubRanking(0.8),
        None,
        KeywordFallbackRecommender(),
    )
    recommendation = pipeline.recommend([job_factory()])[0]
    assert recommendation.labels == ("Backend",)
    assert recommendation.job.category is RoleCategory.SOFTWARE_ENGINEERING


def test_pipeline_only_invokes_keyword_fallback_on_ai_failure(
    job_factory: Callable[..., Job],
) -> None:
    job = job_factory()
    semantic = SemanticMatch((("AI", 0.9),), 0.9, 0.9, 0.1, 0.9, 0.1, 0.9)
    sentinel = KeywordFallbackRecommender().recommend([job])[0]
    fallback = RecordingFallback(sentinel)
    search = StubSearch(semantic)
    search.failure = EmbeddingError("model unavailable")
    pipeline = make_pipeline(search, StubRanking(0.9), None, fallback)
    assert pipeline.recommend([job]) == (sentinel,)
    assert fallback.calls == [job]

    search.failure = None
    analyzer = StubAnalyzer(make_analysis())
    analyzer.failure = MalformedAnalysisError("bad JSON")
    pipeline = make_pipeline(search, StubRanking(0.9), analyzer, fallback)
    assert pipeline.recommend([job]) == (sentinel,)
    assert fallback.calls == [job, job]


def test_keyword_fallback_retains_only_technical_internships(
    job_factory: Callable[..., Job],
) -> None:
    recommender = KeywordFallbackRecommender()
    technical = job_factory(title="Software Engineering Intern")
    nontechnical = job_factory(id="2", title="Sales Intern")
    permanent = job_factory(
        id="3", title="Software Engineer", internship_type=InternshipType.UNKNOWN
    )
    result = recommender.recommend([technical, nontechnical, permanent])
    assert len(result) == 1
    assert result[0].job == technical.with_category(RoleCategory.SOFTWARE_ENGINEERING)
    assert result[0].used_keyword_fallback
