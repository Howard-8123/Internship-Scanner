"""Tests for the auditable live-LLM quality evaluation harness."""

from internship_scanner.intelligence.evaluation import (
    EvaluationCase,
    LiveLLMQualityEvaluator,
    load_evaluation_cases,
    validate_corpus,
)
from internship_scanner.intelligence.llm.analyzer import JobAnalyzer
from internship_scanner.intelligence.models import JobAnalysis, SemanticMatch
from internship_scanner.intelligence.semantic import VectorSimilaritySearch
from internship_scanner.intelligence.text import JobDocument


class EvaluationSearch(VectorSimilaritySearch):
    """Network-free semantic stage for evaluator unit tests."""

    def search(self, document: JobDocument) -> SemanticMatch:
        return SemanticMatch(
            (("Backend", 0.8),),
            0.8,
            0.8,
            0.2,
            0.8,
            0.2,
            0.1,
        )


class EvaluationAnalyzer(JobAnalyzer):
    """Predict cases based on their explicit test title prefix."""

    def analyze(self, document: JobDocument, semantic: SemanticMatch) -> JobAnalysis:
        related = "Positive" in document.text
        return JobAnalysis(
            labels=("Backend",) if related else (),
            confidence=0.99,
            reasoning="Deterministic evaluator test decision.",
            technologies=(),
            programming_languages=(),
            responsibilities=(),
            required_experience=(),
            preferred_experience=(),
            is_internship=related,
            is_technical=related,
            internship_relevance=0.99 if related else 0.01,
            ai_relevance_score=0.0,
        )


def cases() -> tuple[EvaluationCase, ...]:
    """Return a balanced corpus meeting anti-gaming size constraints."""

    positive = tuple(
        EvaluationCase(
            id=f"positive-{index}",
            title=f"Positive role {index}",
            description="A technical student internship writing backend software.",
            expected_is_internship=True,
            expected_is_technical=True,
            expected_labels=("Backend",),
        )
        for index in range(13)
    )
    negative = tuple(
        EvaluationCase(
            id=f"negative-{index}",
            title=f"Negative role {index}",
            description="A permanent nontechnical sales position.",
            expected_is_internship=False,
            expected_is_technical=False,
            expected_labels=(),
        )
        for index in range(12)
    )
    return positive + negative


def test_default_corpus_is_balanced_and_large_enough() -> None:
    loaded = load_evaluation_cases()
    validate_corpus(loaded)
    assert len(loaded) == 32
    assert sum(case.expected_related for case in loaded) == 16


def test_evaluator_calculates_accuracy_and_label_f1() -> None:
    evaluator = LiveLLMQualityEvaluator(
        EvaluationSearch(),
        EvaluationAnalyzer(),
        confidence_threshold=0.65,
        internship_threshold=0.75,
    )
    report = evaluator.evaluate(cases())
    assert report.correct == 25
    assert report.accuracy == 1.0
    assert report.label_precision == 1.0
    assert report.label_recall == 1.0
    assert report.label_f1 == 1.0
    assert report.failures == ()


def test_corpus_validation_rejects_small_or_duplicate_sets() -> None:
    try:
        validate_corpus(cases()[:5])
    except ValueError as error:
        assert "at least 25" in str(error)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("small corpus was accepted")

    duplicate = (*cases()[:-1], cases()[0])
    try:
        validate_corpus(duplicate)
    except ValueError as error:
        assert "unique" in str(error)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("duplicate ids were accepted")
