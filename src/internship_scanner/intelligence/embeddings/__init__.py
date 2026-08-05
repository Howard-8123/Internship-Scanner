"""Embedding provider contracts and implementations."""

from internship_scanner.intelligence.embeddings.base import EmbeddingProvider
from internship_scanner.intelligence.embeddings.bge import BGEEmbeddingProvider
from internship_scanner.intelligence.embeddings.cached import CachedEmbeddingProvider

__all__ = ["BGEEmbeddingProvider", "CachedEmbeddingProvider", "EmbeddingProvider"]
