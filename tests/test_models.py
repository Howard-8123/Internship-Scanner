"""Tests for universal model serialization and compatibility properties."""

from collections.abc import Callable

from internship_scanner.models import Company, Job, RoleCategory


def test_company_cache_round_trip() -> None:
    company = Company("Example", "Board", "Greenhouse")
    assert company.key == "greenhouse:board"
    assert Company.from_dict(company.to_dict()) == company


def test_job_cache_round_trip_and_compatibility(
    job_factory: Callable[..., Job],
) -> None:
    job = job_factory(tags=("Engineering",))
    restored = Job.from_dict(job.to_dict())
    assert restored == job
    assert job.source == job.provider
    assert job.source_job_id == job.id
    assert job.description_html == job.description
    assert job.application_url == job.url
    assert job.posted_date is not None
    assert job.updated_at == job.posted_date.isoformat()
    assert job.with_category(RoleCategory.DATA).category is RoleCategory.DATA
