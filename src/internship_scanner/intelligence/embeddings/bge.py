"""Local BAAI BGE embedding implementation."""

import math
from collections.abc import Callable, Sequence
from typing import Any

from internship_scanner.exceptions import EmbeddingError
from internship_scanner.intelligence.embeddings.base import EmbeddingProvider

DEFAULT_BGE_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbeddingProvider(EmbeddingProvider):
    """Generate normalized BGE vectors through Sentence Transformers.

    The model is loaded lazily so configuration, CLI help, and unit tests do not
    trigger a model download. ``model_factory`` is an injection seam for tests and
    alternate local runtimes.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_MODEL,
        *,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def identity(self) -> str:
        """Return the model and query encoding strategy."""

        return f"bge:{self._model_name}:query-instruction-v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed job documents without a retrieval instruction."""

        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a query using the instruction recommended for BGE v1.5."""

        return self._encode([f"{self._query_instruction}{text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self._get_model().encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            raw = result.tolist() if hasattr(result, "tolist") else result
            vectors = [[float(value) for value in vector] for vector in raw]
            _validate_vectors(vectors, len(texts))
            return vectors
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(
                f"Embedding model {self._model_name} failed"
            ) from error

    def _get_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self._model_name)
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise EmbeddingError(
                        "sentence-transformers is required for BGE embeddings"
                    ) from error
                self._model = SentenceTransformer(self._model_name)
        return self._model


def _validate_vectors(vectors: list[list[float]], expected_count: int) -> None:
    if len(vectors) != expected_count or not vectors or not vectors[0]:
        raise EmbeddingError("Embedding provider returned an invalid vector count")
    dimension = len(vectors[0])
    if any(
        len(vector) != dimension or any(not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise EmbeddingError("Embedding provider returned malformed vectors")
