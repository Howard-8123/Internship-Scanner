"""Greenhouse public Job Board API provider."""

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


class GreenhouseProvider(ConfiguredProvider):
    """Fetch jobs from Greenhouse's unauthenticated public endpoint."""

    name = "greenhouse"
    API_URL = "https://boards-api.greenhouse.io/v1/boards/{identifier}/jobs"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        """Fetch and normalize all published jobs for a Greenhouse board."""

        payload = self._http.get_json(
            self.API_URL.format(identifier=company.identifier),
            params={"content": "true"},
        )
        raw_jobs = payload.get("jobs") if isinstance(payload, Mapping) else None
        if not isinstance(raw_jobs, list):
            raise JobSourceError("Greenhouse returned an invalid jobs payload")
        try:
            return [self._normalize(job, company) for job in raw_jobs]
        except (KeyError, TypeError) as error:
            raise JobSourceError("Greenhouse returned an invalid job record") from error

    def _normalize(self, raw: Mapping[str, Any], company: Company) -> Job:
        location = self._extract_location(raw)
        employment = self._metadata_value(raw, "employment")
        title = string_value(raw.get("title"))
        return Job(
            id=string_value(raw["id"]),
            title=title,
            company=company.name,
            description=string_value(raw.get("content")),
            location=location,
            country=normalize_country(None, location),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(None, location),
            salary=self._metadata_value(raw, "salary"),
            url=string_value(raw.get("absolute_url")),
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=parse_datetime(raw.get("updated_at")),
            deadline=parse_datetime(self._metadata_value(raw, "deadline")),
            tags=self._tags(raw),
        )

    @classmethod
    def _extract_location(cls, raw: Mapping[str, Any]) -> str:
        offices = raw.get("offices") or []
        locations = list(
            dict.fromkeys(
                location
                for office in offices
                if isinstance(office, Mapping)
                if isinstance(
                    location := office.get("location") or office.get("name"), str
                )
                and location
            )
        )
        if locations:
            return ", ".join(locations)
        metadata_location = cls._metadata_value(raw, "location")
        if metadata_location:
            return metadata_location
        location = raw.get("location")
        return (
            string_value(location.get("name"), "Unknown")
            if isinstance(location, Mapping)
            else "Unknown"
        )

    @staticmethod
    def _metadata_value(raw: Mapping[str, Any], name: str) -> str | None:
        for field in raw.get("metadata") or []:
            if (
                not isinstance(field, Mapping)
                or name not in string_value(field.get("name")).casefold()
            ):
                continue
            value = field.get("value")
            if isinstance(value, list):
                return ", ".join(string_value(item) for item in value)
            return string_value(value) or None
        return None

    @staticmethod
    def _tags(raw: Mapping[str, Any]) -> tuple[str, ...]:
        values: list[str] = []
        for key in ("departments", "offices"):
            values.extend(
                string_value(item.get("name"))
                for item in raw.get(key) or []
                if isinstance(item, Mapping) and item.get("name")
            )
        return tuple(dict.fromkeys(values))
