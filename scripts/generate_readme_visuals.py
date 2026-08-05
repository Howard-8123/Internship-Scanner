"""Generate README figures from version-controlled project measurements."""

from pathlib import Path

from internship_scanner.visualization import generate_readme_visuals


def main() -> int:
    """Generate figures and print their repository-relative paths."""

    project_root = Path(__file__).resolve().parents[1]
    for output in generate_readme_visuals(project_root):
        print(output.relative_to(project_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
