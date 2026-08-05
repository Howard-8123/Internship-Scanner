"""Tests for normalized documents, cosine search, models, and ranking."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from internship_scanner.intelligence.cache import SQLiteInferenceCache
from internship_scanner.intelligence.embeddings.base import EmbeddingProvider
from internship_scanner.intelligence.models import JobAnalysis, SemanticMatch
from internship_scanner.intelligence.ranking import (
    RankingWeights,
    WeightedRankingEngine,
)
from internship_scanner.intelligence.semantic import (
    CosineSimilaritySearch,
    SemanticProfile,
    profiles_from_labels,
)
from internship_scanner.intelligence.text import JobDocument, build_job_document
from internship_scanner.models import Job


class SemanticEmbeddings(EmbeddingProvider):
    """Embedding double that makes internship and AI exactly similar."""

    def __init__(self) -> None:
        self.document_calls = 0

    @property
    def identity(self) -> str:
        return "semantic-test"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0] if "Backend" in text else [1.0, 0.0]


def analysis(**overrides: object) -> JobAnalysis:
    """Build a complete strict analysis."""

    values: dict[str, object] = {
        "labels": ["AI", "Machine Learning"],
        "confidence": 0.8,
        "reasoning": "The role builds ML systems.",
        "technologies": ["PyTorch"],
        "programming_languages": ["Python"],
        "responsibilities": ["Train models"],
        "required_experience": [],
        "preferred_experience": ["Research"],
        "is_internship": True,
        "is_technical": True,
        "internship_relevance": 0.9,
        "ai_relevance_score": 1.0,
    }
    values.update(overrides)
    return JobAnalysis.model_validate(values)


def test_job_document_is_stable_and_removes_markup(
    job_factory: Callable[..., Job],
) -> None:
    job = job_factory(description="<p>Build&nbsp; models</p>", tags=("AI", "Python"))
    one = build_job_document(job)
    two = build_job_document(job)
    assert one == two
    assert "<p>" not in one.text
    assert "Build models" in one.text
    assert "AI, Python" in one.text


def test_job_document_decodes_nested_html_and_removes_employer_boilerplate(
    job_factory: Callable[..., Job],
) -> None:
    job = job_factory(
        description=(
            "&lt;div&gt;About Us: We are an AI cloud company.&lt;/div&gt;"
            "&lt;h2&gt;About the Role&lt;/h2&gt;"
            "&lt;p&gt;Implement backend APIs and database tests.&lt;/p&gt;"
            "&lt;h2&gt;Equal Opportunity Employer&lt;/h2&gt;"
            "&lt;p&gt;Boilerplate text.&lt;/p&gt;"
        )
    )
    document = build_job_document(job)
    assert "<div>" not in document.text
    assert "About Us" not in document.text
    assert "AI cloud company" not in document.text
    assert "Implement backend APIs" in document.text
    assert "Boilerplate text" not in document.text
    assert document.text.count(job.title) == 2


def test_job_analysis_normalizes_arrays_and_rejects_extra_fields() -> None:
    value = analysis(labels=[" AI ", "AI", "Research"], technologies=[])
    assert value.labels == ("AI", "Research")
    with pytest.raises(ValidationError):
        analysis(confidence=2.0)
    with pytest.raises(ValidationError):
        analysis(unexpected=True)
    with pytest.raises(ValidationError):
        analysis(reasoning="   ")


def test_cosine_search_scores_profiles_and_caches_results(tmp_path: Path) -> None:
    cache = SQLiteInferenceCache(tmp_path / "cache.sqlite3")
    embeddings = SemanticEmbeddings()
    search = CosineSimilaritySearch(
        embeddings,
        (
            SemanticProfile("AI", "AI role"),
            SemanticProfile("Backend", "Backend role"),
        ),
        cache,
        "1",
    )
    document = JobDocument("AI student role", "digest")
    first = search.search(document)
    second = search.search(document)
    assert first == second
    assert first.label_scores == (("AI", 1.0), ("Backend", 0.0))
    assert first.internship_similarity == 1.0
    assert first.permanent_similarity == 1.0
    assert first.internship_margin == 0.0
    assert first.technical_margin == 0.0
    assert first.ai_similarity == 1.0
    assert first.top_labels(0.5, 1) == ("AI",)
    assert embeddings.document_calls == 1
    cache.close()


def test_profiles_are_open_ended_and_require_at_least_one(tmp_path: Path) -> None:
    assert profiles_from_labels(("Custom", "Custom", "")) == (
        SemanticProfile(
            "Custom",
            "Hands-on technical responsibilities specifically centered on Custom.",
        ),
    )
    cache = SQLiteInferenceCache(tmp_path / "cache.sqlite3")
    with pytest.raises(ValueError, match="at least one"):
        CosineSimilaritySearch(SemanticEmbeddings(), (), cache, "1")
    cache.close()


def test_semantic_match_rejects_invalid_cached_scores() -> None:
    with pytest.raises(ValueError, match="between"):
        SemanticMatch.from_dict(
            {
                "label_scores": [["AI", 2]],
                "semantic_similarity": 1,
                "internship_similarity": 1,
                "permanent_similarity": 0,
                "technical_similarity": 1,
                "nontechnical_similarity": 0,
                "ai_similarity": 1,
            }
        )


def test_weighted_ranking_combines_all_signals() -> None:
    semantic = SemanticMatch(
        label_scores=(("AI", 0.8),),
        semantic_similarity=0.8,
        internship_similarity=0.6,
        permanent_similarity=0.2,
        technical_similarity=0.8,
        nontechnical_similarity=0.2,
        ai_similarity=0.7,
    )
    engine = WeightedRankingEngine(RankingWeights(1, 1, 1, 1))
    assert engine.score(semantic, analysis()) == pytest.approx(0.875)
    assert engine.score(semantic, None) == pytest.approx(0.525)
    with pytest.raises(ValueError, match="positive sum"):
        RankingWeights(0, 0, 0, 0)
