"""Injectable JSON HTTP transport used only by remote LLM adapters."""

from collections.abc import Mapping
from typing import Any, Protocol

import requests

from internship_scanner import __version__
from internship_scanner.exceptions import LLMError


class LLMTransport(Protocol):
    """Minimal HTTP port shared by independent provider adapters."""

    def post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST JSON and return an object response."""


class RequestsLLMTransport:
    """Requests-based transport with timeouts and sanitized errors."""

    def __init__(
        self, timeout_seconds: float, session: requests.Session | None = None
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": f"internship-scanner/{__version__}"}
        )

    def post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST one inference without retrying potentially billable work."""

        try:
            response = self._session.post(
                url,
                json=dict(payload),
                headers=headers,
                params=params,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("LLM response must be an object")
            return value
        except (requests.RequestException, ValueError) as error:
            raise LLMError(f"LLM request failed: {url}") from error
