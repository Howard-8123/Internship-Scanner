"""Cross-provider job deduplication."""

import re
import unicodedata

from internship_scanner.models import Job


def deduplicate_jobs(jobs: list[Job]) -> list[Job]:
    """Collapse likely identical listings, retaining the most complete record."""

    unique: dict[tuple[str, ...], Job] = {}
    for job in jobs:
        key = _job_key(job)
        existing = unique.get(key)
        if existing is None or _completeness(job) > _completeness(existing):
            unique[key] = job
    return list(unique.values())


def _job_key(job: Job) -> tuple[str, ...]:
    return (
        _canonical_company(job.company),
        _canonical(job.title),
        _canonical(job.country or ""),
        _canonical_location(job.location),
    )


def _canonical(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", normalized.casefold()).strip()


def _canonical_company(value: str) -> str:
    words = _canonical(value).split()
    legal_suffixes = {
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "limited",
        "llc",
        "ltd",
        "plc",
    }
    return " ".join(word for word in words if word not in legal_suffixes)


def _canonical_location(value: str) -> str:
    words = _canonical(value).split()
    noise = {
        "gb",
        "hk",
        "hkg",
        "hybrid",
        "office",
        "remote",
        "uk",
        "united",
        "kingdom",
    }
    return " ".join(word for word in words if word not in noise)


def _completeness(job: Job) -> tuple[int, float]:
    populated = sum(
        bool(value)
        for value in (
            job.description,
            job.country,
            job.employment_type,
            job.salary,
            job.url,
            job.deadline,
            job.tags,
        )
    )
    timestamp = job.posted_date.timestamp() if job.posted_date else float("-inf")
    return populated, timestamp
