"""Tests for shared field normalization primitives."""

from internship_scanner.models import InternshipType, RemoteStatus
from internship_scanner.normalization import (
    normalize_country,
    normalize_internship_type,
    normalize_remote_status,
    parse_datetime,
    string_value,
)


def test_country_normalization_uses_codes_and_location_markers() -> None:
    assert normalize_country("GB") == "United Kingdom"
    assert normalize_country(None, "Cambridge") == "United Kingdom"
    assert normalize_country(None, "Hong Kong SAR") == "Hong Kong"
    assert normalize_country("US", "New York") == "US"


def test_remote_status_normalization() -> None:
    assert normalize_remote_status("hybrid") is RemoteStatus.HYBRID
    assert normalize_remote_status(True) is RemoteStatus.REMOTE
    assert normalize_remote_status("on_site") is RemoteStatus.OFFICE
    assert normalize_remote_status(None, "") is RemoteStatus.UNKNOWN
    assert normalize_remote_status(None, "London") is RemoteStatus.OFFICE


def test_internship_type_normalization() -> None:
    assert normalize_internship_type("Developer Co-op") is InternshipType.CO_OP
    assert normalize_internship_type("Year Placement") is InternshipType.PLACEMENT
    assert (
        normalize_internship_type("Working Student") is InternshipType.WORKING_STUDENT
    )
    assert normalize_internship_type("Engineer", "Intern") is InternshipType.INTERNSHIP
    assert normalize_internship_type("Engineer") is InternshipType.UNKNOWN


def test_tolerant_datetime_and_string_parsing() -> None:
    assert parse_datetime("2026-01-01T00:00:00Z") is not None
    assert parse_datetime("2026-01-01") is not None
    assert parse_datetime("not-a-date") is None
    assert parse_datetime(None) is None
    assert string_value(None, "missing") == "missing"
