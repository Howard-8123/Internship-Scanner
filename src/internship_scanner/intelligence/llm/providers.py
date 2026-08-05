"""Independent adapters for supported structured-output LLM APIs."""

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from internship_scanner.exceptions import LLMError
from internship_scanner.intelligence.llm.base import LLMProvider, StructuredRequest
from internship_scanner.intelligence.llm.http import LLMTransport


class OpenAIProvider(LLMProvider):
    """OpenAI Responses API adapter using strict ``text.format`` schemas."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: LLMTransport,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")

    @property
    def identity(self) -> str:
        """Return the configured OpenAI model identity."""

        return f"openai:{self._model}:responses-v1"

    def generate(self, request: StructuredRequest) -> str:
        """Generate strict JSON through the Responses API."""

        response = self._transport.post_json(
            f"{self._base_url}/responses",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload={
                "model": self._model,
                "input": request.prompt,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": request.schema_name,
                        "strict": True,
                        "schema": request.schema,
                    }
                },
            },
        )
        return _openai_output_text(response)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API adapter using ``output_config.format``."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: LLMTransport,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens

    @property
    def identity(self) -> str:
        """Return the configured Anthropic model identity."""

        return f"anthropic:{self._model}:messages-v1"

    def generate(self, request: StructuredRequest) -> str:
        """Generate schema-constrained JSON through the Messages API."""

        response = self._transport.post_json(
            f"{self._base_url}/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            payload={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": request.schema}
                },
            },
        )
        content = _list(response, "content")
        return _string(_mapping(content[0]), "text") if content else _missing()


class GeminiProvider(LLMProvider):
    """Google Gemini generateContent adapter with a response JSON Schema."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: LLMTransport,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")

    @property
    def identity(self) -> str:
        """Return the configured Gemini model identity."""

        return f"gemini:{self._model}:generate-content-v1beta"

    def generate(self, request: StructuredRequest) -> str:
        """Generate JSON through Gemini's public REST API."""

        model = quote(self._model, safe="-_.")
        response = self._transport.post_json(
            f"{self._base_url}/models/{model}:generateContent",
            params={"key": self._api_key},
            payload={
                "contents": [{"parts": [{"text": request.prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": request.schema,
                },
            },
        )
        candidates = _list(response, "candidates")
        if not candidates:
            return _missing()
        content = _mapping(_mapping(candidates[0]).get("content"))
        parts = _list(content, "parts")
        return _string(_mapping(parts[0]), "text") if parts else _missing()


class OpenRouterProvider(LLMProvider):
    """OpenRouter chat-completions adapter using strict structured outputs."""

    def __init__(
        self,
        model: str,
        api_key: str,
        transport: LLMTransport,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")

    @property
    def identity(self) -> str:
        """Return the configured OpenRouter model identity."""

        return f"openrouter:{self._model}:chat-completions-v1"

    def generate(self, request: StructuredRequest) -> str:
        """Generate strict JSON through OpenRouter."""

        response = self._transport.post_json(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload={
                "model": self._model,
                "messages": [{"role": "user", "content": request.prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.schema_name,
                        "strict": True,
                        "schema": request.schema,
                    },
                },
            },
        )
        choices = _list(response, "choices")
        if not choices:
            return _missing()
        message = _mapping(_mapping(choices[0]).get("message"))
        return _string(message, "content")


class OllamaProvider(LLMProvider):
    """Local Ollama chat adapter using a JSON Schema in ``format``."""

    def __init__(
        self,
        model: str,
        transport: LLMTransport,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._transport = transport
        self._base_url = base_url.rstrip("/")

    @property
    def identity(self) -> str:
        """Return the configured local Ollama model identity."""

        return f"ollama:{self._model}:chat-v1"

    def generate(self, request: StructuredRequest) -> str:
        """Generate schema-constrained JSON using the local Ollama API."""

        response = self._transport.post_json(
            f"{self._base_url}/api/chat",
            payload={
                "model": self._model,
                "messages": [{"role": "user", "content": request.prompt}],
                "stream": False,
                "format": request.schema,
            },
        )
        return _string(_mapping(response.get("message")), "content")


def _openai_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    for output in _list(response, "output"):
        for content in _list(_mapping(output), "content"):
            item = _mapping(content)
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                return str(item["text"])
    return _missing()


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LLMError("LLM provider returned an unexpected response shape")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise LLMError("LLM provider returned an unexpected response shape")
    return item


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        return _missing()
    return item


def _missing() -> str:
    raise LLMError("LLM provider returned no structured output")
