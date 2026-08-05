"""Cached prompting and strict validation for structured job analysis."""

import hashlib
import json
from abc import ABC, abstractmethod

from pydantic import ValidationError

from internship_scanner.exceptions import MalformedAnalysisError
from internship_scanner.intelligence.cache import InferenceCache
from internship_scanner.intelligence.llm.base import LLMProvider, StructuredRequest
from internship_scanner.intelligence.models import JobAnalysis, SemanticMatch
from internship_scanner.intelligence.text import JobDocument

PROMPT_VERSION = "job-analysis-v1"


class JobAnalyzer(ABC):
    """Port for a structured analysis stage."""

    @abstractmethod
    def analyze(self, document: JobDocument, semantic: SemanticMatch) -> JobAnalysis:
        """Analyze a normalized job document."""


class StructuredJobAnalyzer(JobAnalyzer):
    """Validate and cache JSON emitted by any configured LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        cache: InferenceCache,
        cache_version: str,
        allowed_labels: tuple[str, ...],
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._version = f"{cache_version}:{provider.identity}:{PROMPT_VERSION}"
        self._allowed_labels = allowed_labels
        self._schema = JobAnalysis.model_json_schema()

    def analyze(self, document: JobDocument, semantic: SemanticMatch) -> JobAnalysis:
        """Return a validated cached result or perform one LLM inference."""

        prompt = self._prompt(document, semantic)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        cached = self._cache.get("llm_analysis", key, self._version)
        if isinstance(cached, dict):
            if cached.get("status") == "invalid":
                raise MalformedAnalysisError("Cached LLM output was malformed")
            try:
                analysis = JobAnalysis.model_validate(cached["analysis"])
                self._validate_labels(analysis)
                return analysis
            except (KeyError, TypeError, ValueError, ValidationError):
                pass

        raw = self._provider.generate(
            StructuredRequest(
                prompt=prompt,
                schema_name="job_analysis",
                schema=self._schema,
            )
        )
        try:
            decoded = json.loads(raw)
            analysis = JobAnalysis.model_validate(decoded)
            self._validate_labels(analysis)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
            self._cache.set(
                "llm_analysis",
                key,
                self._version,
                {
                    "status": "invalid",
                    "output_digest": hashlib.sha256(raw.encode()).hexdigest(),
                },
            )
            raise MalformedAnalysisError(
                "LLM returned malformed structured analysis"
            ) from error

        self._cache.set(
            "llm_analysis",
            key,
            self._version,
            {"status": "valid", "analysis": analysis.model_dump(mode="json")},
        )
        return analysis

    def _validate_labels(self, analysis: JobAnalysis) -> None:
        if any(label not in self._allowed_labels for label in analysis.labels):
            raise ValueError("analysis contains a label outside the catalog")

    def _prompt(self, document: JobDocument, semantic: SemanticMatch) -> str:
        candidate_labels = [label for label, _ in semantic.label_scores[:5]]
        labels = ", ".join(self._allowed_labels)
        candidates = ", ".join(candidate_labels)
        return (
            "Analyze the job below. Return only JSON matching the supplied schema. "
            "Decide whether this is both a genuine student internship and a hands-on "
            "technical role. Ignore employer boilerplate and technologies mentioned "
            "only as company context. Select no more than three labels, based on core "
            "responsibilities rather than incidental words. Scores must be numbers "
            "from 0 to 1. Base every field only on the supplied job; use empty arrays "
            "when the posting does not contain an item.\n\n"
            f"Allowed labels (choose only from these): {labels}\n"
            f"Semantic candidates: {candidates}\n\n"
            f"{document.text}"
        )
