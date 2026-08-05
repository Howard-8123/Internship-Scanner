"""SmartRecruiters public Posting API provider."""

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


class SmartRecruitersProvider(ConfiguredProvider):
    """Fetch public SmartRecruiters posting summaries."""

    name = "smartrecruiters"
    API_URL = "https://api.smartrecruiters.com/v1/companies/{identifier}/postings"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        url = self.API_URL.format(identifier=company.identifier)
        jobs: list[Job] = []
        offset = 0
        while True:
            payload = self._http.get_json(url, params={"limit": 100, "offset": offset})
            raw_jobs = payload.get("content") if isinstance(payload, Mapping) else None
            if not isinstance(raw_jobs, list):
                raise JobSourceError("SmartRecruiters returned an invalid jobs payload")
            jobs.extend(
                self._normalize(raw, company)
                for raw in raw_jobs
                if isinstance(raw, Mapping)
            )
            total = int(payload.get("totalFound", len(jobs)))
            if not raw_jobs or len(jobs) >= total:
                return jobs
            offset += len(raw_jobs)

    def _normalize(self, raw: Mapping[str, Any], company: Company) -> Job:
        location_data = raw.get("location")
        location_data = location_data if isinstance(location_data, Mapping) else {}
        location = (
            ", ".join(
                string_value(location_data.get(key))
                for key in ("city", "region", "country")
                if location_data.get(key)
            )
            or "Unknown"
        )
        employment_data = raw.get("typeOfEmployment")
        employment = (
            string_value(employment_data.get("label"))
            if isinstance(employment_data, Mapping)
            else None
        )
        title = string_value(raw.get("name"))
        department = raw.get("department")
        tags = (
            (string_value(department.get("label")),)
            if isinstance(department, Mapping) and department.get("label")
            else ()
        )
        identifier = string_value(raw.get("id") or raw.get("uuid"))
        return Job(
            id=identifier,
            title=title,
            company=company.name,
            description="",
            location=location,
            country=normalize_country(location_data.get("country"), location),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(
                location_data.get("remote"), location
            ),
            salary=None,
            url=string_value(
                raw.get("ref")
                or f"https://careers.smartrecruiters.com/{company.identifier}/{identifier}"
            ),
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=parse_datetime(raw.get("releasedDate")),
            deadline=None,
            tags=tags,
        )
