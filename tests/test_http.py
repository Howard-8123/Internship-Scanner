"""Tests for provider-neutral HTTP transport behavior."""

from unittest.mock import Mock

import pytest
import requests
from requests.adapters import HTTPAdapter

from internship_scanner import __version__
from internship_scanner.exceptions import JobSourceError
from internship_scanner.http import HttpClient


def test_get_json_sends_options_and_decodes() -> None:
    response = Mock()
    response.json.return_value = {"ok": True}
    session = Mock(spec=requests.Session)
    session.get.return_value = response
    client = HttpClient(5, session)
    assert client.get_json(
        "https://example.com", params={"limit": 1}, headers={"Accept": "json"}
    ) == {"ok": True}
    session.get.assert_called_once_with(
        "https://example.com",
        params={"limit": 1},
        headers={"Accept": "json"},
        timeout=5,
    )
    response.raise_for_status.assert_called_once()


def test_get_text_returns_response_text() -> None:
    response = Mock(text="content")
    session = Mock(spec=requests.Session)
    session.get.return_value = response
    assert HttpClient(5, session).get_text("https://example.com") == "content"


@pytest.mark.parametrize("method", ["get_json", "get_text"])
def test_transport_wraps_request_errors(method: str) -> None:
    session = Mock(spec=requests.Session)
    session.get.side_effect = requests.Timeout("slow")
    with pytest.raises(JobSourceError, match="Request failed"):
        getattr(HttpClient(5, session), method)("https://example.com")


def test_json_decode_error_is_wrapped() -> None:
    response = Mock()
    response.json.side_effect = ValueError("invalid")
    session = Mock(spec=requests.Session)
    session.get.return_value = response
    with pytest.raises(JobSourceError, match="Request failed"):
        HttpClient(5, session).get_json("https://example.com")


def test_default_session_has_identity_and_retry_adapter() -> None:
    client = HttpClient(5)
    session = client._session
    assert session.headers["User-Agent"] == f"internship-scanner/{__version__}"
    adapter = session.get_adapter("https://")
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 2
