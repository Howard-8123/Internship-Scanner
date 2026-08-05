"""Contract and normalization tests for every public ATS provider."""

from typing import Any
from unittest.mock import Mock

import pytest

from internship_scanner.exceptions import JobSourceError
from internship_scanner.http import HttpClient
from internship_scanner.models import Company, InternshipType, RemoteStatus
from internship_scanner.providers import (
    AshbyProvider,
    GreenhouseProvider,
    LeverProvider,
    PersonioProvider,
    RecruiteeProvider,
    SmartRecruitersProvider,
    WorkableProvider,
)


def http_mock() -> Mock:
    return Mock(spec=HttpClient)


def provider_company(provider: str) -> Company:
    return Company("Example", "example", provider)


@pytest.mark.parametrize(
    ("provider_type", "name"),
    [
        (GreenhouseProvider, "greenhouse"),
        (LeverProvider, "lever"),
        (AshbyProvider, "ashby"),
        (SmartRecruitersProvider, "smartrecruiters"),
        (WorkableProvider, "workable"),
        (RecruiteeProvider, "recruitee"),
        (PersonioProvider, "personio"),
    ],
)
def test_provider_contract_discovers_deduplicated_configured_companies(
    provider_type: Any, name: str
) -> None:
    company = provider_company(name)
    provider = provider_type([company, company, provider_company("other")], http_mock())
    assert provider.name == name
    assert provider.discover_companies() == [company]


def test_greenhouse_normalization_preserves_location_fallbacks() -> None:
    http = http_mock()
    http.get_json.return_value = {
        "jobs": [
            {
                "id": 1,
                "title": "Security Internship",
                "content": "Description",
                "absolute_url": "https://example.com/1",
                "updated_at": "2026-01-02T00:00:00Z",
                "offices": [
                    {"location": "London, UK", "name": "HQ"},
                    {"location": "London, UK"},
                ],
                "departments": [{"name": "Engineering"}],
                "metadata": [{"name": "Employment Type", "value": "Intern"}],
            }
        ]
    }
    company = provider_company("greenhouse")
    job = GreenhouseProvider([company], http).fetch_jobs(company)[0]
    assert job.location == "London, UK"
    assert job.country == "United Kingdom"
    assert job.internship_type is InternshipType.INTERNSHIP
    assert job.tags == ("Engineering", "HQ")


def test_greenhouse_rejects_invalid_payload() -> None:
    http = http_mock()
    http.get_json.return_value = {}
    company = provider_company("greenhouse")
    with pytest.raises(JobSourceError, match="invalid jobs payload"):
        GreenhouseProvider([company], http).fetch_jobs(company)


def test_lever_normalizes_salary_country_and_workplace() -> None:
    http = http_mock()
    http.get_json.return_value = [
        {
            "id": "lever-1",
            "text": "Developer Co-op",
            "descriptionPlain": "Description",
            "categories": {
                "location": "Hong Kong",
                "commitment": "Intern",
                "team": "Platform",
            },
            "country": "HK",
            "workplaceType": "hybrid",
            "salaryRange": {
                "currency": "HKD",
                "min": 100,
                "max": 200,
                "interval": "month",
            },
            "hostedUrl": "https://jobs.lever.co/example/1",
        }
    ]
    company = provider_company("lever")
    job = LeverProvider([company], http).fetch_jobs(company)[0]
    assert job.country == "Hong Kong"
    assert job.remote_status is RemoteStatus.HYBRID
    assert job.internship_type is InternshipType.CO_OP
    assert job.salary == "HKD 100-200 month"


def test_ashby_omits_unlisted_jobs_and_normalizes_compensation() -> None:
    http = http_mock()
    http.get_json.return_value = {
        "jobs": [
            {"title": "Hidden Intern", "isListed": False},
            {
                "title": "Data Placement",
                "location": "London",
                "employmentType": "Intern",
                "workplaceType": "Remote",
                "jobUrl": "https://jobs.ashbyhq.com/example/abc",
                "publishedAt": "2026-02-01T00:00:00Z",
                "address": {"postalAddress": {"addressCountry": "GB"}},
                "compensation": {"scrapeableCompensationSalarySummary": "£20,000"},
                "department": "Data",
            },
        ]
    }
    company = provider_company("ashby")
    jobs = AshbyProvider([company], http).fetch_jobs(company)
    assert len(jobs) == 1
    assert jobs[0].id == "abc"
    assert jobs[0].salary == "£20,000"
    assert jobs[0].remote_status is RemoteStatus.REMOTE


def test_smartrecruiters_paginates_and_normalizes() -> None:
    http = http_mock()
    http.get_json.side_effect = [
        {
            "totalFound": 2,
            "content": [
                {
                    "id": "1",
                    "name": "Software Intern",
                    "location": {"city": "London", "country": "gb", "remote": True},
                    "typeOfEmployment": {"label": "Intern"},
                    "department": {"label": "Engineering"},
                    "releasedDate": "2026-01-01",
                }
            ],
        },
        {
            "totalFound": 2,
            "content": [
                {"id": "2", "name": "Marketing Intern", "location": {"country": "hk"}}
            ],
        },
    ]
    company = provider_company("smartrecruiters")
    jobs = SmartRecruitersProvider([company], http).fetch_jobs(company)
    assert [job.id for job in jobs] == ["1", "2"]
    assert jobs[0].remote_status is RemoteStatus.REMOTE
    assert jobs[1].country == "Hong Kong"


def test_workable_uses_discovered_account_name() -> None:
    http = http_mock()
    http.get_json.return_value = {
        "name": "Actual Company",
        "jobs": [
            {
                "shortcode": "ABC",
                "title": "Engineering Intern",
                "city": "London",
                "country": "United Kingdom",
                "workplace_type": "on_site",
                "employment_type": "Intern",
                "application_url": "https://example.com/ABC",
                "published_on": "2026-01-01",
            }
        ],
    }
    company = provider_company("workable")
    job = WorkableProvider([company], http).fetch_jobs(company)[0]
    assert job.company == "Actual Company"
    assert job.remote_status is RemoteStatus.OFFICE


def test_recruitee_normalizes_flexible_public_offer_shape() -> None:
    http = http_mock()
    http.get_json.return_value = {
        "offers": [
            {
                "id": 12,
                "title": "Working Student Developer",
                "location": {"name": "Hong Kong", "country": "HK"},
                "remote": True,
                "tags": [{"name": "Python"}],
                "department": {"name": "Engineering"},
                "careers_url": "https://example.recruitee.com/o/job",
            }
        ]
    }
    company = provider_company("recruitee")
    job = RecruiteeProvider([company], http).fetch_jobs(company)[0]
    assert job.internship_type is InternshipType.WORKING_STUDENT
    assert job.tags == ("Python", "Engineering")
    assert job.remote_status is RemoteStatus.REMOTE


def test_personio_parses_public_xml_feed() -> None:
    http = http_mock()
    http.get_text.return_value = (
        "<workzag-jobs><position><id>9</id><name>Research Intern</name>"
        "<office>London, UK</office><department>R&amp;D</department>"
        "<employmentType>Intern</employmentType><createdAt>2026-01-01</createdAt>"
        "<jobDescriptions><jobDescription><value>Research things</value>"
        "</jobDescription></jobDescriptions></position></workzag-jobs>"
    )
    company = provider_company("personio")
    job = PersonioProvider([company], http).fetch_jobs(company)[0]
    assert job.id == "9"
    assert job.country == "United Kingdom"
    assert "Research things" in job.description
    assert job.tags == ("R&D",)


def test_personio_rejects_invalid_xml() -> None:
    http = http_mock()
    http.get_text.return_value = "<invalid"
    company = provider_company("personio")
    with pytest.raises(JobSourceError, match="invalid XML"):
        PersonioProvider([company], http).fetch_jobs(company)


@pytest.mark.parametrize(
    ("provider_type", "name", "payload"),
    [
        (LeverProvider, "lever", {}),
        (AshbyProvider, "ashby", {}),
        (SmartRecruitersProvider, "smartrecruiters", {}),
        (WorkableProvider, "workable", {}),
        (RecruiteeProvider, "recruitee", {}),
    ],
)
def test_json_providers_reject_invalid_top_level_payloads(
    provider_type: Any, name: str, payload: object
) -> None:
    http = http_mock()
    http.get_json.return_value = payload
    company = provider_company(name)
    with pytest.raises(JobSourceError, match="invalid jobs payload"):
        provider_type([company], http).fetch_jobs(company)
