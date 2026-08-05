"""Ashby public Job Postings API provider."""

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


class AshbyProvider(ConfiguredProvider):
    """Fetch listed postings from Ashby's unauthenticated job-board API."""

    name = "ashby"
    API_URL = "https://api.ashbyhq.com/posting-api/job-board/{identifier}"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        payload = self._http.get_json(
            self.API_URL.format(identifier=company.identifier),
            params={"includeCompensation": "true"},
        )
        raw_jobs = payload.get("jobs") if isinstance(payload, Mapping) else None
        if not isinstance(raw_jobs, list):
            raise JobSourceError("Ashby returned an invalid jobs payload")
        return [
            self._normalize(raw, company)
            for raw in raw_jobs
            if isinstance(raw, Mapping) and raw.get("isListed", True)
        ]

    def _normalize(self, raw: Mapping[str, Any], company: Company) -> Job:
        title = string_value(raw.get("title"))
        location = string_value(raw.get("location"), "Unknown")
        employment = string_value(raw.get("employmentType")) or None
        postal = raw.get("address")
        postal = postal.get("postalAddress", {}) if isinstance(postal, Mapping) else {}
        country = postal.get("addressCountry") if isinstance(postal, Mapping) else None
        compensation = raw.get("compensation")
        salary = (
            compensation.get("scrapeableCompensationSalarySummary")
            if isinstance(compensation, Mapping)
            else None
        )
        url = string_value(raw.get("jobUrl") or raw.get("applyUrl"))
        return Job(
            id=url.rstrip("/").rsplit("/", 1)[-1],
            title=title,
            company=company.name,
            description=string_value(
                raw.get("descriptionPlain") or raw.get("descriptionHtml")
            ),
            location=location,
            country=normalize_country(country, location),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(raw.get("workplaceType"), location),
            salary=string_value(salary) or None,
            url=url,
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=parse_datetime(raw.get("publishedAt")),
            deadline=None,
            tags=tuple(
                string_value(raw.get(key))
                for key in ("department", "team")
                if raw.get(key)
            ),
        )
