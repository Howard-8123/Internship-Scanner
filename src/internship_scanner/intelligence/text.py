"""Deterministic conversion of universal jobs into inference documents."""

import hashlib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from internship_scanner.models import Job

WHITESPACE = re.compile(r"\s+")
ROLE_SECTION_MARKERS = (
    "about the role",
    "about the department",
    "role overview",
    "job description",
    "what you'll do",
    "what you will do",
    "responsibilities",
)
TRAILING_BOILERPLATE_MARKERS = (
    "what makes cloudflare special",
    "equal opportunity employer",
    "equal employment opportunity",
    "diversity and inclusion",
)


@dataclass(frozen=True, slots=True)
class JobDocument:
    """Stable normalized text and digest used throughout the AI pipeline."""

    text: str
    digest: str

    @classmethod
    def from_text(cls, text: str) -> "JobDocument":
        """Create a document and its stable digest from normalized text."""

        return cls(text=text, digest=hashlib.sha256(text.encode()).hexdigest())


def build_job_document(job: Job) -> JobDocument:
    """Create a provider-neutral document containing relevant job facts."""

    description = extract_role_text(job.description)
    tags = ", ".join(job.tags)
    parts = (
        f"Title: {normalize_text(job.title)}",
        f"Role title: {normalize_text(job.title)}",
        f"Employment type: {normalize_text(job.employment_type or '')}",
        f"Tags: {normalize_text(tags)}",
        f"Role-specific description: {description}",
    )
    text = "\n".join(part for part in parts if not part.endswith(": "))
    return JobDocument.from_text(text)


def normalize_text(value: str) -> str:
    """Unescape nested entities, remove markup, and collapse whitespace."""

    unescaped = html.unescape(html.unescape(value))
    parser = _TextExtractor()
    parser.feed(unescaped)
    parser.close()
    normalized = html.unescape(parser.text).replace(
        "\N{RIGHT SINGLE QUOTATION MARK}", "'"
    )
    return WHITESPACE.sub(" ", normalized).strip()


def extract_role_text(value: str) -> str:
    """Remove common employer boilerplate around the role-specific description."""

    text = normalize_text(value)
    folded = text.casefold()
    starts = [folded.find(marker) for marker in ROLE_SECTION_MARKERS]
    valid_starts = [position for position in starts if position >= 0]
    if valid_starts:
        start = min(valid_starts)
        text = text[start:]
        folded = text.casefold()

    ends = [folded.find(marker) for marker in TRAILING_BOILERPLATE_MARKERS]
    valid_ends = [position for position in ends if position > 0]
    if valid_ends:
        text = text[: min(valid_ends)]
    return text.strip()


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML without retaining element names."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        """Return all collected text fragments."""

        return " ".join(self._parts)

    def handle_data(self, data: str) -> None:
        """Retain human-visible node content."""

        self._parts.append(data)
