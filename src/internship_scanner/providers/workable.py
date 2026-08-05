"""Workable public careers endpoint provider."""

from collections.abc import Mapping
from typing import Any

from internship_scanner.exceptions import JobSourceError
from internship_scanner.http import HttpClient
from internship_scanner.models import Company, Job
from internship_scanner.normalization import (
    normalize_country,
    normalize_internship_type,
    normalize_remote_status,
    parse_datetime,
    string_value,
)
from internship_scanner.providers.base import ConfiguredProvider


class WorkableProvider(ConfiguredProvider):
    """Fetch public Workable account jobs without private API credentials."""

    name = "workable"
    API_URL = "https://www.workable.com/api/accounts/{identifier}"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        payload = self._http.get_json(
            self.API_URL.format(identifier=company.identifier),
            params={"details": "true"},
        )
        raw_jobs = payload.get("jobs") if isinstance(payload, Mapping) else None
        if not isinstance(raw_jobs, list):
            raise JobSourceError("Workable returned an invalid jobs payload")
        discovered_name = string_value(payload.get("name"), company.name)
        discovered = Company(discovered_name, company.identifier, self.name)
        return [
            self._normalize(raw, discovered)
            for raw in raw_jobs
            if isinstance(raw, Mapping)
        ]

    def _normalize(self, raw: Mapping[str, Any], company: Company) -> Job:
        location = (
            ", ".join(
                string_value(raw.get(key))
                for key in ("city", "state", "country")
                if raw.get(key)
            )
            or "Unknown"
        )
        title = string_value(raw.get("title"))
        employment = string_value(raw.get("employment_type")) or None
        tags = tuple(
            string_value(raw.get(key))
            for key in ("department", "function", "industry")
            if raw.get(key)
        )
        return Job(
            id=string_value(raw.get("shortcode") or raw.get("code")),
            title=title,
            company=company.name,
            description=string_value(raw.get("description")),
            location=location,
            country=normalize_country(raw.get("country"), location),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(
                raw.get("workplace_type") or raw.get("telecommuting"), location
            ),
            salary=None,
            url=string_value(raw.get("application_url") or raw.get("url")),
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=parse_datetime(
                raw.get("published_on") or raw.get("created_at")
            ),
            deadline=None,
            tags=tags,
        )
