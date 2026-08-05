"""Persistent cache decorator for arbitrary embedding providers."""

import hashlib
import math
from collections.abc import Sequence

from internship_scanner.exceptions import EmbeddingError, IntelligenceCacheError
from internship_scanner.intelligence.cache import InferenceCache
from internship_scanner.intelligence.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """Never recompute an embedding for the same text/model/cache version."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: InferenceCache,
        cache_version: str,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._version = f"{cache_version}:{provider.identity}"

    @property
    def identity(self) -> str:
        """Expose the wrapped embedding identity."""

        return self._provider.identity

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Read cached document vectors and batch only cache misses."""

        return self._embed_many("document_embeddings", texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        """Read or create one cached query vector."""

        return self._embed_many("query_embeddings", [text], query=True)[0]

    def _embed_many(
        self, namespace: str, texts: Sequence[str], *, query: bool
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}
        for index, text in enumerate(texts):
            key = hashlib.sha256(text.encode()).hexdigest()
            try:
                cached = _cached_vector(self._cache.get(namespace, key, self._version))
            except (TypeError, ValueError):
                cached = None
            if cached is not None:
                vectors[index] = cached
                continue
            if key in missing:
                missing[key][1].append(index)
            else:
                missing[key] = (text, [index])

        if missing:
            missing_items = list(missing.items())
            missing_texts = [item[1][0] for item in missing_items]
            generated = (
                [self._provider.embed_query(missing_texts[0])]
                if query
                else self._provider.embed_documents(missing_texts)
            )
            if len(generated) != len(missing_items):
                raise EmbeddingError(
                    "Embedding provider returned the wrong vector count"
                )
            for (key, (_, indexes)), vector in zip(
                missing_items, generated, strict=True
            ):
                _validate_vector(vector)
                self._cache.set(namespace, key, self._version, {"vector": vector})
                for index in indexes:
                    vectors[index] = vector

        if any(vector is None for vector in vectors):  # pragma: no cover - invariant
            raise IntelligenceCacheError("Embedding cache left unresolved values")
        return [vector for vector in vectors if vector is not None]


def _cached_vector(value: object) -> list[float] | None:
    if not isinstance(value, dict) or "vector" not in value:
        return None
    raw = value["vector"]
    if not isinstance(raw, list):
        raise TypeError("cached vector must be a list")
    vector = [float(item) for item in raw]
    _validate_vector(vector)
    return vector


def _validate_vector(vector: list[float]) -> None:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise EmbeddingError("Embedding provider returned a malformed vector")
