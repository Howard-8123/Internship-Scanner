"""Tests for reproducible documentation visualizations."""

from __future__ import annotations

import shutil
from pathlib import Path

from internship_scanner.visualization import generate_readme_visuals

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = (
    "recommendation-quality-before-after.png",
    "semantic-margin-separation.png",
    "llm-evaluation-corpus.png",
    "scan-verification-funnel.png",
    "cli-diagnostics.png",
)


def test_readme_visuals_are_generated_from_versioned_project_data(
    tmp_path: Path,
) -> None:
    """The complete documented figure set must be reproducible as valid PNGs."""

    sources = (
        Path("docs/data/recommendation_quality_audit.json"),
        Path("docs/data/scan_verification.json"),
        Path("docs/data/cli-diagnostics.txt"),
        Path("src/internship_scanner/data/llm_quality.json"),
    )
    for relative_path in sources:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative_path, destination)

    outputs = generate_readme_visuals(tmp_path)

    assert tuple(path.name for path in outputs) == EXPECTED_IMAGES
    for output in outputs:
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert output.stat().st_size > 10_000


def test_readme_embeds_every_generated_quantitative_figure() -> None:
    """README claims and generated evidence must remain visibly connected."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for image_name in EXPECTED_IMAGES:
        assert f"docs/images/{image_name}" in readme
        assert (PROJECT_ROOT / "docs" / "images" / image_name).is_file()
