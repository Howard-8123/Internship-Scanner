"""Pure functions for identifying and classifying internships."""

import re

from internship_scanner.models import Job, RoleCategory

INTERNSHIP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bintern(ship)?s?\b",
        r"\bplacement\b",
        r"\bco-?op\b",
        r"\bworking student\b",
    )
)

NON_TECHNICAL_KEYWORDS = (
    "sales",
    "marketing",
    "social media",
    "customer advocacy",
    "customer services",
    "legal",
    "finance",
    "recruiting",
    "human resources",
    "business development",
    "communications",
)

CATEGORY_KEYWORDS: tuple[tuple[RoleCategory, tuple[str, ...]], ...] = (
    (
        RoleCategory.AI_ML,
        (
            "machine learning",
            "artificial intelligence",
            "generative ai",
            "data science",
        ),
    ),
    (RoleCategory.RESEARCH_ENGINEERING, ("research engineer",)),
    (
        RoleCategory.SOFTWARE_ENGINEERING,
        ("software", "developer", "backend", "frontend", "full stack", "full-stack"),
    ),
    (
        RoleCategory.INFRASTRUCTURE_SECURITY,
        (
            "network",
            "security",
            "cyber",
            "cloud",
            "platform",
            "infrastructure",
            "systems",
        ),
    ),
    (RoleCategory.DATA, ("data", "analytics")),
)


def is_internship(job: Job) -> bool:
    """Return whether the title contains an internship marker."""

    return any(pattern.search(job.title) for pattern in INTERNSHIP_PATTERNS)


def classify_role(job: Job) -> RoleCategory:
    """Classify a role using the prototype's ordered title rules."""

    title = job.title.lower()
    if any(keyword in title for keyword in NON_TECHNICAL_KEYWORDS):
        return RoleCategory.NON_TECHNICAL
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in title for keyword in keywords):
            return category
    return RoleCategory.OTHER
