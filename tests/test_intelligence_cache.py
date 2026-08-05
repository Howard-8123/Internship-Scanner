"""Tests for persistent inference caching and invalidation."""

from pathlib import Path

from internship_scanner.intelligence.cache import SQLiteInferenceCache


def test_sqlite_cache_persists_exact_versions_and_invalidates(tmp_path: Path) -> None:
    path = tmp_path / "ai.sqlite3"
    cache = SQLiteInferenceCache(path)
    cache.set("embeddings", "job", "model-v1", {"vector": [1.0, 0.0]})
    assert cache.get("embeddings", "job", "model-v1") == {"vector": [1.0, 0.0]}
    assert cache.get("embeddings", "job", "model-v2") is None
    cache.close()

    reopened = SQLiteInferenceCache(path)
    assert reopened.get("embeddings", "job", "model-v1") is not None
    reopened.set("analysis", "job", "prompt-v1", {"status": "valid"})
    assert reopened.invalidate("embeddings") == 1
    assert reopened.get("analysis", "job", "prompt-v1") is not None
    assert reopened.invalidate() == 1
    reopened.close()


def test_sqlite_cache_supports_in_memory_storage() -> None:
    cache = SQLiteInferenceCache(Path(":memory:"))
    cache.set("semantic", "key", "v1", 0.75)
    assert cache.get("semantic", "key", "v1") == 0.75
    assert cache.invalidate("missing") == 0
    cache.close()
