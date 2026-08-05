"""Personio public XML careers-feed provider."""

import xml.etree.ElementTree as ET

from internship_scanner.exceptions import JobSourceError
from internship_scanner.http import HttpClient
from internship_scanner.models import Company, Job
from internship_scanner.normalization import (
    normalize_country,
    normalize_internship_type,
    normalize_remote_status,
    parse_datetime,
)
from internship_scanner.providers.base import ConfiguredProvider


class PersonioProvider(ConfiguredProvider):
    """Fetch public positions from a Personio career-site XML feed."""

    name = "personio"
    API_URL = "https://{identifier}.jobs.personio.de/xml"

    def __init__(self, companies: list[Company], http: HttpClient) -> None:
        self._http = http
        super().__init__(companies)

    def fetch_jobs(self, company: Company) -> list[Job]:
        content = self._http.get_text(
            self.API_URL.format(identifier=company.identifier),
            params={"language": "en"},
        )
        try:
            root = ET.fromstring(content)
        except ET.ParseError as error:
            raise JobSourceError("Personio returned invalid XML") from error
        return [
            self._normalize(position, company)
            for position in root.findall(".//position")
        ]

    def _normalize(self, position: ET.Element, company: Company) -> Job:
        title = self._text(position, "name")
        location = self._text(position, "office", "Unknown")
        employment = self._text(position, "employmentType") or None
        description = "\n".join(
            text.strip()
            for text in position.itertext()
            if text.strip() and text.strip() != title
        )
        identifier = self._text(position, "id")
        return Job(
            id=identifier,
            title=title,
            company=company.name,
            description=description,
            location=location,
            country=normalize_country(None, location),
            employment_type=employment,
            internship_type=normalize_internship_type(title, employment),
            remote_status=normalize_remote_status(None, location),
            salary=None,
            url=f"https://{company.identifier}.jobs.personio.de/?language=en",
            provider=self.name,
            company_identifier=company.identifier,
            posted_date=parse_datetime(self._text(position, "createdAt")),
            deadline=None,
            tags=tuple(
                value
                for value in (
                    self._text(position, "department"),
                    self._text(position, "recruitingCategory"),
                )
                if value
            ),
        )

    @staticmethod
    def _text(element: ET.Element, name: str, default: str = "") -> str:
        child = element.find(name)
        return child.text.strip() if child is not None and child.text else default
