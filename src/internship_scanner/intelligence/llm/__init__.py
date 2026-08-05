"""Structured LLM provider contracts and built-in adapters."""

from internship_scanner.intelligence.llm.analyzer import StructuredJobAnalyzer
from internship_scanner.intelligence.llm.base import LLMProvider, StructuredRequest
from internship_scanner.intelligence.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "StructuredJobAnalyzer",
    "StructuredRequest",
]
