"""Provider-neutral HTTP transport with consistent errors."""

from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from internship_scanner import __version__
from internship_scanner.exceptions import JobSourceError


class HttpClient:
    """Small injectable wrapper around a requests session."""

    def __init__(
        self, timeout_seconds: float, session: requests.Session | None = None
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._session = session or self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"User-Agent": f"internship-scanner/{__version__}"})
        return session

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float | bool] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """GET and decode JSON, translating transport failures."""

        try:
            response = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            raise JobSourceError(f"Request failed: {url}") from error

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float | bool] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        """GET text, translating transport failures."""

        try:
            response = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            raise JobSourceError(f"Request failed: {url}") from error
