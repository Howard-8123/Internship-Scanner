"""Lever public Postings API provider."""

from collections.abc import Mapping
from typing import Any

from internship_scanner.exceptions import JobSourceError
from internship_scanner.http import HttpClient
from internship_scanner.models import Company, Job
from internship_scanner.normalization import (
    normalize_country,
    normalize_internship_type,
    normalize_remote_status,
    string_value,
)
from internship_scanner.providers.base import ConfiguredProvider


class LeverProvider(ConfiguredProvider):
    """Fetch published Lever postings without authentication."""

    name = "lever"
    API_URL = "https://api.lever.co/v0/postings/{identifier}"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        payload = self._http.get_json(
            self.API_URL.format(identifier=company.identifier), params={"mode": "json"}
        )
        if not isinstance(payload, list):
            raise JobSourceError("Lever returned an invalid jobs payload")
        try:
            return [self._normalize(raw, company) for raw in payload]
        except (AttributeError, TypeError) as error:
            raise JobSourceError("Lever returned an invalid job record") from error

    def _normalize(self, raw: Mapping[str, Any], company: Company) -> Job:
        categories = raw.get("categories")
        categories = categories if isinstance(categories, Mapping) else {}
        location = string_value(categories.get("location"), "Unknown")
        employment = string_value(categories.get("commitment")) or None
        title = string_value(raw.get("text"))
        salary = raw.get("salaryDescriptionPlain")
        if not salary and isinstance(raw.get("salaryRange"), Mapping):
            salary_range = raw["salaryRange"]
            salary = "{currency} {minimum}-{maximum} {interval}".format(
                currency=string_value(salary_range.get("currency")),
                minimum=string_value(salary_range.get("min")),
                maximum=string_value(salary_range.get("max")),
                interval=string_value(salary_range.get("interval")),
            ).strip()
        tags = tuple(
            string_value(categories.get(key))
            for key in ("team", "department")
            if categories.get(key)
        )
        return Job(
            id=string_value(raw.get("id")),
            title=title,
            company=company.name,
            description=string_value(
                raw.get("descriptionPlain") or raw.get("description")
            ),
            location=location,
            country=normalize_country(raw.get("country"), location),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(raw.get("workplaceType"), location),
            salary=string_value(salary) or None,
            url=string_value(raw.get("hostedUrl") or raw.get("applyUrl")),
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=None,
            deadline=None,
            tags=tags,
        )
