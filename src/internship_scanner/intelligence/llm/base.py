"""Model- and vendor-independent LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """Prompt and JSON Schema supplied identically to every LLM adapter."""

    prompt: str
    schema_name: str
    schema: dict[str, Any]


class LLMProvider(ABC):
    """Generate schema-constrained JSON through an interchangeable backend."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable vendor/model identity used for caching."""

    @abstractmethod
    def generate(self, request: StructuredRequest) -> str:
        """Return the provider's JSON response as text."""
