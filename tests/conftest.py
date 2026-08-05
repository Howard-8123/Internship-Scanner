"""Shared fixtures for provider and engine tests."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from internship_scanner.models import (
    Company,
    InternshipType,
    Job,
    RemoteStatus,
)


@pytest.fixture
def company() -> Company:
    """Return a representative configured board."""

    return Company("Example", "example", "greenhouse")


@pytest.fixture
def job_factory() -> Callable[..., Job]:
    """Create universal jobs with concise overrides."""

    def create(title: str = "Software Engineer Intern", **overrides: Any) -> Job:
        values: dict[str, Any] = {
            "id": "1",
            "title": title,
            "company": "Example",
            "description": "Build software",
            "location": "London, United Kingdom",
            "country": "United Kingdom",
            "employment_type": "Intern",
            "internship_type": InternshipType.INTERNSHIP,
            "remote_status": RemoteStatus.HYBRID,
            "salary": None,
            "url": "https://example.com/jobs/1",
            "provider": "greenhouse",
            "company_identifier": "example",
            "posted_date": datetime(2026, 1, 1, tzinfo=UTC),
            "deadline": None,
            "tags": (),
        }
        values.update(overrides)
        return Job(**values)

    return create
