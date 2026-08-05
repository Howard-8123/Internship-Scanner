"""Tests for internship detection and role classification."""

from collections.abc import Callable

import pytest

from internship_scanner.classification import classify_role, is_internship
from internship_scanner.models import Job, RoleCategory


@pytest.mark.parametrize(
    "title",
    [
        "Engineering Intern",
        "Summer Internships",
        "Industrial Placement",
        "Co-op Developer",
        "Working Student",
    ],
)
def test_recognizes_internship_titles(
    job_factory: Callable[..., Job], title: str
) -> None:
    assert is_internship(job_factory(title))


def test_does_not_match_internal(job_factory: Callable[..., Job]) -> None:
    assert not is_internship(job_factory("Internal Tools Engineer"))


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Machine Learning Intern", RoleCategory.AI_ML),
        ("Research Engineer Intern", RoleCategory.RESEARCH_ENGINEERING),
        ("Backend Developer Intern", RoleCategory.SOFTWARE_ENGINEERING),
        ("Security Intern", RoleCategory.INFRASTRUCTURE_SECURITY),
        ("Analytics Intern", RoleCategory.DATA),
        ("Marketing Intern", RoleCategory.NON_TECHNICAL),
        ("Product Intern", RoleCategory.OTHER),
    ],
)
def test_classifies_roles(
    job_factory: Callable[..., Job], title: str, expected: RoleCategory
) -> None:
    assert classify_role(job_factory(title)) is expected


def test_non_technical_rule_takes_precedence(job_factory: Callable[..., Job]) -> None:
    assert classify_role(job_factory("Data Sales Intern")) is RoleCategory.NON_TECHNICAL
