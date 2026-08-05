"""Shared normalization primitives used independently by provider adapters."""

import re
from datetime import datetime

from internship_scanner.models import InternshipType, RemoteStatus

COUNTRY_ALIASES = {
    "gb": "United Kingdom",
    "gbr": "United Kingdom",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "hk": "Hong Kong",
    "hkg": "Hong Kong",
    "hong kong": "Hong Kong",
    "hong kong sar": "Hong Kong",
}

UK_LOCATION_MARKERS = (
    "belfast",
    "birmingham",
    "bristol",
    "cambridge",
    "cardiff",
    "edinburgh",
    "glasgow",
    "london",
    "manchester",
    "oxford",
    "reading",
)


def normalize_country(country: object, location: str = "") -> str | None:
    """Normalize explicit country codes or infer supported countries from location."""

    raw = str(country or "").strip().lower()
    if raw in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[raw]
    haystack = location.casefold()
    for alias, normalized in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", haystack):
            return normalized
    if any(re.search(rf"\b{marker}\b", haystack) for marker in UK_LOCATION_MARKERS):
        return "United Kingdom"
    return str(country).strip() if country else None


def normalize_remote_status(value: object, location: str = "") -> RemoteStatus:
    """Normalize common ATS workplace labels."""

    text = f"{value or ''} {location}".casefold().replace("_", "-")
    if "hybrid" in text:
        return RemoteStatus.HYBRID
    if "remote" in text or value is True:
        return RemoteStatus.REMOTE
    if any(marker in text for marker in ("on-site", "onsite", "on site", "office")):
        return RemoteStatus.OFFICE
    if location.strip() and location.casefold() != "unknown":
        return RemoteStatus.OFFICE
    return RemoteStatus.UNKNOWN


def normalize_internship_type(
    title: str, employment_type: object = None
) -> InternshipType:
    """Normalize internship terminology from title and employment type."""

    text = f"{title} {employment_type or ''}".casefold()
    if re.search(r"\bco-?op\b", text):
        return InternshipType.CO_OP
    if "placement" in text:
        return InternshipType.PLACEMENT
    if "working student" in text:
        return InternshipType.WORKING_STUDENT
    if re.search(r"\bintern(ship)?s?\b", text):
        return InternshipType.INTERNSHIP
    return InternshipType.UNKNOWN


def parse_datetime(value: object) -> datetime | None:
    """Parse common ISO timestamps without failing an entire provider response."""

    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None


def string_value(value: object, default: str = "") -> str:
    """Return a string without leaking JSON null as the literal ``None``."""

    return str(value) if value is not None else default
