"""Recruitee public Careers Site API provider."""

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


class RecruiteeProvider(ConfiguredProvider):
    """Fetch published Recruitee offers from the public careers API."""

    name = "recruitee"
    API_URL = "https://{identifier}.recruitee.com/api/offers/"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        payload = self._http.get_json(
            self.API_URL.format(identifier=company.identifier),
            headers={"Accept": "application/json"},
        )
        raw_jobs = payload.get("offers") if isinstance(payload, Mapping) else None
        if not isinstance(raw_jobs, list):
            raise JobSourceError("Recruitee returned an invalid jobs payload")
        return [
            self._normalize(raw, company)
            for raw in raw_jobs
            if isinstance(raw, Mapping)
        ]

    def _normalize(self, raw: Mapping[str, Any], company: Company) -> Job:
        location_data = raw.get("location")
        location_data = location_data if isinstance(location_data, Mapping) else {}
        location = string_value(
            raw.get("location_name") or location_data.get("name") or raw.get("city"),
            "Unknown",
        )
        employment = string_value(raw.get("employment_type")) or None
        title = string_value(raw.get("title"))
        department = raw.get("department")
        tags = tuple(
            string_value(tag.get("name") if isinstance(tag, Mapping) else tag)
            for tag in raw.get("tags") or []
        )
        if department:
            tags += (
                string_value(
                    department.get("name")
                    if isinstance(department, Mapping)
                    else department
                ),
            )
        return Job(
            id=string_value(raw.get("id") or raw.get("slug")),
            title=title,
            company=company.name,
            description=string_value(
                raw.get("description") or raw.get("description_html")
            ),
            location=location,
            country=normalize_country(
                raw.get("country") or location_data.get("country"), location
            ),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(
                raw.get("remote") or raw.get("workplace_type"), location
            ),
            salary=string_value(raw.get("salary")) or None,
            url=string_value(raw.get("careers_url") or raw.get("url")),
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=parse_datetime(
                raw.get("created_at") or raw.get("published_at")
            ),
            deadline=parse_datetime(raw.get("expires_at")),
            tags=tuple(tag for tag in tags if tag),
        )
