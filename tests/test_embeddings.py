"""Tests for BGE embedding behavior and persistent cache decoration."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from internship_scanner.exceptions import EmbeddingError
from internship_scanner.intelligence.cache import SQLiteInferenceCache
from internship_scanner.intelligence.embeddings.base import EmbeddingProvider
from internship_scanner.intelligence.embeddings.bge import BGEEmbeddingProvider
from internship_scanner.intelligence.embeddings.cached import CachedEmbeddingProvider


class ArrayResult:
    """Small NumPy-like result used by the BGE adapter test."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class FakeModel:
    """Recording Sentence Transformers model double."""

    def __init__(self, values: list[list[float]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._values = values

    def encode(self, texts: list[str], **kwargs: Any) -> ArrayResult:
        self.calls.append(texts)
        values = self._values or [[1.0, float(index)] for index, _ in enumerate(texts)]
        return ArrayResult(values)


class RecordingEmbeddings(EmbeddingProvider):
    """Deterministic provider with call counters."""

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def identity(self) -> str:
        return "fake:v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(len(text)), 1.0]


def test_bge_is_lazy_and_applies_query_instruction() -> None:
    model = FakeModel()
    created: list[str] = []

    def factory(name: str) -> FakeModel:
        created.append(name)
        return model

    provider = BGEEmbeddingProvider("example/model", model_factory=factory)
    assert created == []
    assert provider.embed_documents(["job one", "job two"]) == [
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    provider.embed_query("internship")
    assert created == ["example/model"]
    assert model.calls[-1][0].endswith("internship")
    assert model.calls[-1][0].startswith("Represent this sentence")
    assert provider.identity.startswith("bge:example/model")


def test_bge_rejects_invalid_or_failed_model_output() -> None:
    invalid = BGEEmbeddingProvider(model_factory=lambda _: FakeModel([[float("nan")]]))
    with pytest.raises(EmbeddingError, match="malformed"):
        invalid.embed_documents(["job"])

    class BrokenModel:
        def encode(self, texts: list[str], **kwargs: Any) -> None:
            raise RuntimeError("offline")

    broken = BGEEmbeddingProvider(model_factory=lambda _: BrokenModel())
    with pytest.raises(EmbeddingError, match="failed"):
        broken.embed_query("job")


def test_cached_embeddings_only_generate_missing_unique_texts(tmp_path: Path) -> None:
    cache = SQLiteInferenceCache(tmp_path / "cache.sqlite3")
    underlying = RecordingEmbeddings()
    provider = CachedEmbeddingProvider(underlying, cache, "1")
    first = provider.embed_documents(["same", "same", "different"])
    second = provider.embed_documents(["same", "different"])
    assert first[0] == first[1]
    assert second == [first[0], first[2]]
    assert underlying.document_calls == [["same", "different"]]

    assert provider.embed_query("student") == provider.embed_query("student")
    assert underlying.query_calls == ["student"]
    cache.close()


def test_cached_embeddings_reject_wrong_vector_count(tmp_path: Path) -> None:
    class EmptyEmbeddings(RecordingEmbeddings):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return []

    cache = SQLiteInferenceCache(tmp_path / "cache.sqlite3")
    provider = CachedEmbeddingProvider(EmptyEmbeddings(), cache, "1")
    with pytest.raises(EmbeddingError, match="wrong vector count"):
        provider.embed_documents(["job"])
    cache.close()
