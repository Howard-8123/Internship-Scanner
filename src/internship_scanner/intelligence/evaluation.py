"""Reproducible quality evaluation for a real configured LLM analyzer."""

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from internship_scanner.intelligence.llm.analyzer import JobAnalyzer
from internship_scanner.intelligence.semantic import VectorSimilaritySearch
from internship_scanner.intelligence.text import JobDocument

MINIMUM_CORPUS_SIZE = 25
MINIMUM_CLASS_SIZE = 10


class EvaluationCase(BaseModel):
    """One human-labeled role used to measure live LLM classification quality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expected_is_internship: bool
    expected_is_technical: bool
    expected_labels: tuple[str, ...]

    @property
    def expected_related(self) -> bool:
        """Return whether this is a technical internship recommendation."""

        return self.expected_is_internship and self.expected_is_technical

    @property
    def document(self) -> JobDocument:
        """Return the same normalized shape consumed by live analysis."""

        text = (
            f"Title: {self.title}\n"
            f"Role title: {self.title}\n"
            f"Role-specific description: {self.description}"
        )
        return JobDocument.from_text(text)


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """A misclassified benchmark case and its live model decision."""

    case_id: str
    expected_related: bool
    predicted_related: bool
    reasoning: str


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Auditable binary accuracy and multi-label F1 metrics."""

    total: int
    correct: int
    accuracy: float
    label_precision: float
    label_recall: float
    label_f1: float
    failures: tuple[EvaluationFailure, ...]


class LiveLLMQualityEvaluator:
    """Measure a real analyzer using semantic candidates from the production model."""

    def __init__(
        self,
        semantic_search: VectorSimilaritySearch,
        analyzer: JobAnalyzer,
        *,
        confidence_threshold: float,
        internship_threshold: float,
    ) -> None:
        self._semantic_search = semantic_search
        self._analyzer = analyzer
        self._confidence_threshold = confidence_threshold
        self._internship_threshold = internship_threshold

    def evaluate(self, cases: tuple[EvaluationCase, ...]) -> EvaluationReport:
        """Run uncensored live LLM analysis over every positive and negative case."""

        validate_corpus(cases)
        correct = 0
        true_positive_labels = 0
        false_positive_labels = 0
        false_negative_labels = 0
        failures: list[EvaluationFailure] = []
        for case in cases:
            document = case.document
            semantic = self._semantic_search.search(document)
            analysis = self._analyzer.analyze(document, semantic)
            predicted_related = (
                analysis.is_internship
                and analysis.is_technical
                and analysis.confidence >= self._confidence_threshold
                and analysis.internship_relevance >= self._internship_threshold
            )
            if predicted_related == case.expected_related:
                correct += 1
            else:
                failures.append(
                    EvaluationFailure(
                        case.id,
                        case.expected_related,
                        predicted_related,
                        analysis.reasoning,
                    )
                )

            expected = set(case.expected_labels) if case.expected_related else set()
            predicted = set(analysis.labels) if predicted_related else set()
            true_positive_labels += len(expected & predicted)
            false_positive_labels += len(predicted - expected)
            false_negative_labels += len(expected - predicted)

        precision = _ratio(
            true_positive_labels, true_positive_labels + false_positive_labels
        )
        recall = _ratio(
            true_positive_labels, true_positive_labels + false_negative_labels
        )
        label_f1 = _ratio(2 * precision * recall, precision + recall)
        return EvaluationReport(
            total=len(cases),
            correct=correct,
            accuracy=correct / len(cases),
            label_precision=precision,
            label_recall=recall,
            label_f1=label_f1,
            failures=tuple(failures),
        )


def load_evaluation_cases(path: Path | None = None) -> tuple[EvaluationCase, ...]:
    """Load a custom corpus or the version-controlled default benchmark."""

    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else resources.files("internship_scanner")
        .joinpath("data/llm_quality.json")
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    return tuple(TypeAdapter(list[EvaluationCase]).validate_python(value))


def validate_corpus(cases: tuple[EvaluationCase, ...]) -> None:
    """Prevent small or imbalanced datasets from producing misleading accuracy."""

    positive = sum(case.expected_related for case in cases)
    negative = len(cases) - positive
    if len(cases) < MINIMUM_CORPUS_SIZE:
        raise ValueError(
            f"evaluation corpus requires at least {MINIMUM_CORPUS_SIZE} cases"
        )
    if positive < MINIMUM_CLASS_SIZE or negative < MINIMUM_CLASS_SIZE:
        raise ValueError(
            f"evaluation corpus requires at least {MINIMUM_CLASS_SIZE} cases per class"
        )
    identifiers = {case.id for case in cases}
    if len(identifiers) != len(cases):
        raise ValueError("evaluation case ids must be unique")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
