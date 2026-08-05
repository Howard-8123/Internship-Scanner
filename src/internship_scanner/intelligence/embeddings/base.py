"""Model-independent embedding contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Convert documents and search queries to vectors through one interface."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable provider/model identity for cache versioning."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passage-like job documents in input order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query."""
