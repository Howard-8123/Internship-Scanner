"""Network-free tests for structured LLM adapters and analysis validation."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from internship_scanner.exceptions import LLMError, MalformedAnalysisError
from internship_scanner.intelligence.cache import SQLiteInferenceCache
from internship_scanner.intelligence.llm.analyzer import StructuredJobAnalyzer
from internship_scanner.intelligence.llm.base import LLMProvider, StructuredRequest
from internship_scanner.intelligence.llm.http import RequestsLLMTransport
from internship_scanner.intelligence.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from internship_scanner.intelligence.models import SemanticMatch
from internship_scanner.intelligence.text import JobDocument


class RecordingTransport:
    """Record provider requests and return an injected response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers or {}),
                "params": dict(params or {}),
            }
        )
        return self.response


class RecordingLLM(LLMProvider):
    """LLM provider double returning exact JSON text."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[StructuredRequest] = []

    @property
    def identity(self) -> str:
        return "fake:model:v1"

    def generate(self, request: StructuredRequest) -> str:
        self.calls.append(request)
        return self.output


def analysis_json(**overrides: object) -> str:
    """Return a complete structured analysis JSON response."""

    value: dict[str, object] = {
        "labels": ["AI", "Machine Learning"],
        "confidence": 0.9,
        "reasoning": "The posting describes model development.",
        "technologies": ["PyTorch"],
        "programming_languages": ["Python"],
        "responsibilities": ["Build models"],
        "required_experience": [],
        "preferred_experience": ["ML coursework"],
        "is_internship": True,
        "is_technical": True,
        "internship_relevance": 0.95,
        "ai_relevance_score": 0.98,
    }
    value.update(overrides)
    return json.dumps(value)


def request() -> StructuredRequest:
    return StructuredRequest("Analyze", "job_analysis", {"type": "object"})


def test_openai_uses_responses_api_strict_text_format() -> None:
    transport = RecordingTransport(
        {"output": [{"content": [{"type": "output_text", "text": "{}"}]}]}
    )
    provider = OpenAIProvider("gpt", "secret", transport)
    assert provider.generate(request()) == "{}"
    call = transport.calls[0]
    assert call["url"].endswith("/responses")
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["payload"]["store"] is False
    assert call["payload"]["text"]["format"]["strict"] is True
    assert provider.identity == "openai:gpt:responses-v1"


def test_anthropic_uses_messages_output_config() -> None:
    transport = RecordingTransport({"content": [{"text": "{}"}]})
    provider = AnthropicProvider("claude", "secret", transport)
    assert provider.generate(request()) == "{}"
    call = transport.calls[0]
    assert call["url"].endswith("/messages")
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["payload"]["output_config"]["format"]["type"] == "json_schema"
    assert provider.identity.startswith("anthropic:claude")


def test_gemini_uses_generate_content_json_schema() -> None:
    transport = RecordingTransport(
        {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
    )
    provider = GeminiProvider("gemini/test", "secret", transport)
    assert provider.generate(request()) == "{}"
    call = transport.calls[0]
    assert call["url"].endswith("/models/gemini%2Ftest:generateContent")
    assert call["params"] == {"key": "secret"}
    assert call["payload"]["generationConfig"]["responseMimeType"] == (
        "application/json"
    )
    assert provider.identity.startswith("gemini:gemini/test")


def test_openrouter_uses_chat_completions_response_format() -> None:
    transport = RecordingTransport({"choices": [{"message": {"content": "{}"}}]})
    provider = OpenRouterProvider("vendor/model", "secret", transport)
    assert provider.generate(request()) == "{}"
    call = transport.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["payload"]["response_format"]["json_schema"]["strict"] is True
    assert provider.identity.startswith("openrouter:vendor/model")


def test_ollama_uses_local_chat_format_schema() -> None:
    transport = RecordingTransport({"message": {"content": "{}"}})
    provider = OllamaProvider("local-model", transport)
    assert provider.generate(request()) == "{}"
    call = transport.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["payload"]["stream"] is False
    assert call["payload"]["format"] == {"type": "object"}
    assert provider.identity.startswith("ollama:local-model")


@pytest.mark.parametrize(
    "provider",
    [
        OpenAIProvider("gpt", "key", RecordingTransport({"output": []})),
        AnthropicProvider("claude", "key", RecordingTransport({"content": []})),
        GeminiProvider("gemini", "key", RecordingTransport({"candidates": []})),
        OpenRouterProvider("model", "key", RecordingTransport({"choices": []})),
        OllamaProvider("model", RecordingTransport({"message": {}})),
    ],
)
def test_providers_reject_missing_structured_output(provider: LLMProvider) -> None:
    with pytest.raises(LLMError, match="no structured output"):
        provider.generate(request())


def test_requests_transport_posts_and_sanitizes_failures() -> None:
    response = Mock()
    response.json.return_value = {"ok": True}
    session = Mock()
    session.headers = {}
    session.post.return_value = response
    transport = RequestsLLMTransport(3.0, session)
    assert transport.post_json("https://llm.test", payload={"x": 1}) == {"ok": True}
    session.post.assert_called_once_with(
        "https://llm.test",
        json={"x": 1},
        headers=None,
        params=None,
        timeout=3.0,
    )

    session.post.side_effect = requests.Timeout("secret details")
    with pytest.raises(LLMError, match=r"https://llm\.test"):
        transport.post_json("https://llm.test", payload={})


def test_structured_analyzer_validates_and_caches_output(tmp_path: Path) -> None:
    cache = SQLiteInferenceCache(tmp_path / "ai.sqlite3")
    provider = RecordingLLM(analysis_json())
    analyzer = StructuredJobAnalyzer(provider, cache, "1", ("AI", "Machine Learning"))
    document = JobDocument("Title: ML Intern", "digest")
    semantic = SemanticMatch((("AI", 0.9),), 0.9, 0.9, 0.1, 0.9, 0.1, 0.9)
    first = analyzer.analyze(document, semantic)
    second = analyzer.analyze(document, semantic)
    assert first == second
    assert first.labels == ("AI", "Machine Learning")
    assert len(provider.calls) == 1
    assert provider.calls[0].schema["additionalProperties"] is False
    assert "Allowed labels" in provider.calls[0].prompt
    cache.close()


@pytest.mark.parametrize(
    "output",
    ["not json", analysis_json(confidence=4), analysis_json(labels=["Unknown"])],
)
def test_structured_analyzer_caches_malformed_output(
    tmp_path: Path, output: str
) -> None:
    cache = SQLiteInferenceCache(tmp_path / "ai.sqlite3")
    provider = RecordingLLM(output)
    analyzer = StructuredJobAnalyzer(provider, cache, "1", ("AI",))
    document = JobDocument("job", "digest")
    semantic = SemanticMatch((("AI", 1.0),), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(MalformedAnalysisError, match="malformed"):
        analyzer.analyze(document, semantic)
    with pytest.raises(MalformedAnalysisError, match="Cached"):
        analyzer.analyze(document, semantic)
    assert len(provider.calls) == 1
    cache.close()
