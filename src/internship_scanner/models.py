"""Provider-neutral domain models."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any


class RoleCategory(StrEnum):
    """Technical role classification retained from the prototype."""

    AI_ML = "AI/ML"
    RESEARCH_ENGINEERING = "Research engineering"
    SOFTWARE_ENGINEERING = "Software engineering"
    INFRASTRUCTURE_SECURITY = "Infrastructure and security"
    DATA = "Data"
    NON_TECHNICAL = "Non-technical"
    OTHER = "Other"


class RemoteStatus(StrEnum):
    """Normalized workplace arrangement."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    OFFICE = "office"
    UNKNOWN = "unknown"


class InternshipType(StrEnum):
    """Normalized internship program type."""

    INTERNSHIP = "internship"
    PLACEMENT = "placement"
    CO_OP = "co-op"
    WORKING_STUDENT = "working_student"
    UNKNOWN = "unknown"


TECHNICAL_CATEGORIES = frozenset(
    {
        RoleCategory.AI_ML,
        RoleCategory.RESEARCH_ENGINEERING,
        RoleCategory.SOFTWARE_ENGINEERING,
        RoleCategory.INFRASTRUCTURE_SECURITY,
        RoleCategory.DATA,
    }
)


@dataclass(frozen=True, slots=True)
class Company:
    """A company job board discovered by one provider."""

    name: str
    identifier: str
    provider: str

    @property
    def key(self) -> str:
        """Return the provider-scoped stable company key."""

        return f"{self.provider}:{self.identifier}".lower()

    def to_dict(self) -> dict[str, str]:
        """Serialize the company for persistent caching."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Company":
        """Restore a company from cached data."""

        return cls(
            name=str(value["name"]),
            identifier=str(value["identifier"]),
            provider=str(value["provider"]),
        )


@dataclass(frozen=True, slots=True)
class Job:
    """Universal job representation emitted by every ATS provider."""

    id: str
    title: str
    company: str
    description: str
    location: str
    country: str | None
    employment_type: str | None
    internship_type: InternshipType
    remote_status: RemoteStatus
    salary: str | None
    url: str
    provider: str
    company_identifier: str
    posted_date: datetime | None
    deadline: datetime | None
    tags: tuple[str, ...] = ()
    category: RoleCategory = RoleCategory.OTHER

    # Phase 1 compatibility properties.
    @property
    def source(self) -> str:
        """Return the provider name (Phase 1 compatibility)."""

        return self.provider

    @property
    def source_job_id(self) -> str:
        """Return the provider job identifier (Phase 1 compatibility)."""

        return self.id

    @property
    def description_html(self) -> str:
        """Return the normalized description (Phase 1 compatibility)."""

        return self.description

    @property
    def updated_at(self) -> str | None:
        """Return an ISO date when known (Phase 1 compatibility)."""

        return self.posted_date.isoformat() if self.posted_date else None

    @property
    def application_url(self) -> str:
        """Return the application URL (Phase 1 compatibility)."""

        return self.url

    def with_category(self, category: RoleCategory) -> "Job":
        """Return a copy assigned to ``category``."""

        return replace(self, category=category)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the job for persistent caching."""

        value = asdict(self)
        value["internship_type"] = self.internship_type.value
        value["remote_status"] = self.remote_status.value
        value["category"] = self.category.value
        value["posted_date"] = (
            self.posted_date.isoformat() if self.posted_date else None
        )
        value["deadline"] = self.deadline.isoformat() if self.deadline else None
        value["tags"] = list(self.tags)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Job":
        """Restore a job from cached data."""

        return cls(
            id=str(value["id"]),
            title=str(value["title"]),
            company=str(value["company"]),
            description=str(value["description"]),
            location=str(value["location"]),
            country=str(value["country"]) if value.get("country") else None,
            employment_type=(
                str(value["employment_type"]) if value.get("employment_type") else None
            ),
            internship_type=InternshipType(value["internship_type"]),
            remote_status=RemoteStatus(value["remote_status"]),
            salary=str(value["salary"]) if value.get("salary") else None,
            url=str(value["url"]),
            provider=str(value["provider"]),
            company_identifier=str(value["company_identifier"]),
            posted_date=_parse_cached_datetime(value.get("posted_date")),
            deadline=_parse_cached_datetime(value.get("deadline")),
            tags=tuple(str(tag) for tag in value.get("tags", [])),
            category=RoleCategory(value.get("category", RoleCategory.OTHER)),
        )


def _parse_cached_datetime(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value else None
