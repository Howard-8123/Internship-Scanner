"""Configurable cosine-similarity search over semantic label profiles."""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from internship_scanner.exceptions import EmbeddingError
from internship_scanner.intelligence.cache import InferenceCache
from internship_scanner.intelligence.embeddings.base import EmbeddingProvider
from internship_scanner.intelligence.models import SemanticMatch
from internship_scanner.intelligence.text import JobDocument

DEFAULT_LABELS = (
    "AI",
    "Machine Learning",
    "Deep Learning",
    "LLMs",
    "NLP",
    "Computer Vision",
    "Robotics",
    "Data Engineering",
    "Backend",
    "Frontend",
    "Full Stack",
    "Cloud",
    "Infrastructure",
    "Security",
    "DevOps",
    "Embedded",
    "Mobile",
    "Research",
    "General Software Engineering",
)
AI_LABELS = frozenset(
    {"AI", "Machine Learning", "Deep Learning", "LLMs", "NLP", "Computer Vision"}
)
INTERNSHIP_QUERY = (
    "A temporary internship, placement, co-op, or student work programme explicitly "
    "intended for a currently enrolled university student."
)
PERMANENT_ROLE_QUERY = (
    "A permanent full-time professional position for an experienced hire, manager, "
    "representative, counsel, architect, or account owner; not a student internship."
)
TECHNICAL_ROLE_QUERY = (
    "A hands-on technical engineering role whose core responsibilities include "
    "programming, software development, data engineering, infrastructure, security "
    "engineering, or scientific research."
)
NONTECHNICAL_ROLE_QUERY = (
    "A commercial or business role in sales, business development, account "
    "management, professional services, customer success, marketing, legal, finance, "
    "recruiting, or operations, without hands-on software engineering responsibilities."
)
LABEL_DESCRIPTIONS = {
    "AI": (
        "Design and implement artificial intelligence systems, intelligent agents, "
        "generative AI features, or applied AI products as a core responsibility."
    ),
    "Machine Learning": (
        "Train, evaluate, optimize, and deploy statistical machine learning models "
        "and production ML systems."
    ),
    "Deep Learning": (
        "Develop neural network architectures and train deep learning models using "
        "PyTorch, TensorFlow, or similar frameworks."
    ),
    "LLMs": (
        "Build, fine-tune, evaluate, or deploy large language models, retrieval "
        "augmented generation, or language model infrastructure."
    ),
    "NLP": (
        "Develop natural language processing systems for text understanding, "
        "generation, classification, search, or speech-language tasks."
    ),
    "Computer Vision": (
        "Develop image or video understanding, visual recognition, detection, "
        "segmentation, or multimodal vision models."
    ),
    "Robotics": (
        "Build robotic perception, planning, control, autonomy, or physical "
        "automation systems."
    ),
    "Data Engineering": (
        "Build data pipelines, streaming systems, warehouses, ETL platforms, or "
        "large-scale data processing infrastructure."
    ),
    "Backend": (
        "Write server-side application code, APIs, distributed services, databases, "
        "and backend business logic."
    ),
    "Frontend": (
        "Build web user interfaces and browser applications using JavaScript, "
        "TypeScript, React, HTML, or CSS."
    ),
    "Full Stack": (
        "Build both browser user interfaces and server-side application services "
        "as a full-stack software engineer."
    ),
    "Cloud": (
        "Engineer cloud computing platforms, cloud services, distributed systems, "
        "virtualization, or cloud-native software."
    ),
    "Infrastructure": (
        "Build and operate systems infrastructure, networking platforms, operating "
        "systems, reliability, or distributed platforms."
    ),
    "Security": (
        "Engineer cybersecurity, application security, network security, threat "
        "detection, cryptography, or security tooling."
    ),
    "DevOps": (
        "Build CI/CD, deployment automation, observability, site reliability, and "
        "developer operations platforms."
    ),
    "Embedded": (
        "Develop firmware, device drivers, real-time systems, or software for "
        "resource-constrained hardware."
    ),
    "Mobile": "Develop native or cross-platform iOS and Android mobile applications.",
    "Research": (
        "Conduct novel computer science or engineering research, design experiments, "
        "publish findings, and prototype algorithms."
    ),
    "General Software Engineering": (
        "Design, implement, test, and maintain production software as a hands-on "
        "software engineer."
    ),
}


@dataclass(frozen=True, slots=True)
class SemanticProfile:
    """A configurable label and natural-language retrieval description."""

    label: str
    description: str


class VectorSimilaritySearch(ABC):
    """Port for replacing the in-process search with a future vector database."""

    @abstractmethod
    def search(self, document: JobDocument) -> SemanticMatch:
        """Return semantic scores for one normalized document."""


class CosineSimilaritySearch(VectorSimilaritySearch):
    """Compute and persist cosine scores using injected embeddings."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        profiles: tuple[SemanticProfile, ...],
        cache: InferenceCache,
        cache_version: str,
    ) -> None:
        if not profiles:
            raise ValueError("at least one semantic profile is required")
        self._embeddings = embeddings
        self._profiles = profiles
        profile_value = [(item.label, item.description) for item in profiles]
        self._profile_digest = hashlib.sha256(
            json.dumps(profile_value, separators=(",", ":")).encode()
        ).hexdigest()
        self._cache = cache
        self._version = (
            f"{cache_version}:{embeddings.identity}:{self._profile_digest}:cosine-v2"
        )

    def search(self, document: JobDocument) -> SemanticMatch:
        """Return cached scores or compute them with cosine similarity."""

        key = document.digest
        cached = self._cache.get("semantic_similarity", key, self._version)
        if isinstance(cached, dict):
            try:
                return SemanticMatch.from_dict(cached)
            except (KeyError, TypeError, ValueError):
                pass

        document_vector = self._embeddings.embed_documents([document.text])[0]
        queries = [profile.description for profile in self._profiles]
        query_vectors = [self._embeddings.embed_query(query) for query in queries]
        scores = tuple(
            sorted(
                (
                    (profile.label, _cosine(document_vector, vector))
                    for profile, vector in zip(
                        self._profiles, query_vectors, strict=True
                    )
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        internship_score = _cosine(
            document_vector, self._embeddings.embed_query(INTERNSHIP_QUERY)
        )
        permanent_score = _cosine(
            document_vector, self._embeddings.embed_query(PERMANENT_ROLE_QUERY)
        )
        technical_score = _cosine(
            document_vector, self._embeddings.embed_query(TECHNICAL_ROLE_QUERY)
        )
        nontechnical_score = _cosine(
            document_vector, self._embeddings.embed_query(NONTECHNICAL_ROLE_QUERY)
        )
        ai_score = max(
            (score for label, score in scores if label in AI_LABELS), default=0.0
        )
        result = SemanticMatch(
            label_scores=scores,
            semantic_similarity=max(score for _, score in scores),
            internship_similarity=internship_score,
            permanent_similarity=permanent_score,
            technical_similarity=technical_score,
            nontechnical_similarity=nontechnical_score,
            ai_similarity=ai_score,
        )
        self._cache.set("semantic_similarity", key, self._version, result.to_dict())
        return result


def profiles_from_labels(labels: tuple[str, ...]) -> tuple[SemanticProfile, ...]:
    """Build open-ended profiles so adding a label requires configuration only."""

    return tuple(
        SemanticProfile(
            label=label,
            description=LABEL_DESCRIPTIONS.get(
                label,
                (
                    "Hands-on technical responsibilities specifically centered on "
                    f"{label}."
                ),
            ),
        )
        for label in dict.fromkeys(item.strip() for item in labels if item.strip())
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise EmbeddingError("cosine vectors must have matching non-zero dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    # Embedding cosine can be negative; downstream relevance scores are normalized.
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))
