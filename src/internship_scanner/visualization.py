"""Reproducible Matplotlib figures for measured README results."""

import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

BLUE = "#2563EB"
ORANGE = "#D97706"
GREEN = "#059669"
RED = "#DC2626"
GRAY = "#64748B"
LIGHT_GRAY = "#CBD5E1"
DARK = "#0F172A"


def generate_readme_visuals(project_root: Path) -> tuple[Path, ...]:
    """Generate every quantitative README figure from version-controlled data."""

    data_dir = project_root / "docs" / "data"
    image_dir = project_root / "docs" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    audit = _load_json(data_dir / "recommendation_quality_audit.json")
    scan = _load_json(data_dir / "scan_verification.json")
    corpus = _load_json(
        project_root / "src" / "internship_scanner" / "data" / "llm_quality.json"
    )

    outputs = (
        image_dir / "recommendation-quality-before-after.png",
        image_dir / "semantic-margin-separation.png",
        image_dir / "llm-evaluation-corpus.png",
        image_dir / "scan-verification-funnel.png",
    )
    plot_recommendation_comparison(audit, outputs[0])
    plot_semantic_margins(audit, outputs[1])
    plot_evaluation_corpus(corpus, outputs[2])
    plot_scan_funnel(scan, outputs[3])

    terminal_path = data_dir / "cli-diagnostics.txt"
    if terminal_path.exists():
        terminal_output = image_dir / "cli-diagnostics.png"
        render_cli_screenshot(
            terminal_path.read_text(encoding="utf-8"), terminal_output
        )
        return (*outputs, terminal_output)
    return outputs


def plot_recommendation_comparison(data: dict[str, Any], output: Path) -> None:
    """Compare actual pre/post-remediation scores and emitted label counts."""

    all_roles = data["roles"]
    roles = sorted(
        all_roles, key=lambda role: float(role["before_score"]), reverse=True
    )[:6]
    names = [_short_title(str(role["title"])) for role in roles]
    positions = np.arange(len(roles), dtype=float)
    before_scores = np.array([role["before_score"] for role in roles], dtype=float)
    after_scores = np.array([role["after_score"] for role in roles], dtype=float)
    before_labels = np.array(
        [role["before_label_count"] for role in roles], dtype=float
    )
    after_labels = np.array([role["after_label_count"] for role in roles], dtype=float)
    accepted = np.array([role["after_accepted"] for role in roles], dtype=bool)
    old_label_mean = float(
        np.mean([role["before_label_count"] for role in all_roles], dtype=float)
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        before_scores,
        width,
        color=ORANGE,
        label="Before remediation",
    )
    after_colors = np.where(accepted, GREEN, GRAY)
    axes[0].bar(
        positions + width / 2,
        after_scores,
        width,
        color=after_colors,
        label="After remediation",
    )
    rejected_positions = positions[~accepted] + width / 2
    axes[0].scatter(
        rejected_positions,
        after_scores[~accepted] + 0.006,
        marker="x",
        color=RED,
        s=80,
        linewidths=2,
        label="Rejected by relative gates",
        zorder=3,
    )
    axes[0].axhline(
        float(data["recommendation_threshold"]),
        color=DARK,
        linestyle="--",
        linewidth=1.2,
        label="Score threshold",
    )
    axes[0].set_title(
        "Six highest pre-remediation scores from the 11-role quality audit"
    )
    axes[0].set_ylabel("Final ranking score")
    axes[0].set_xticks(positions, names, rotation=18, ha="right")
    axes[0].set_ylim(0.34, 0.5)
    axes[0].legend(ncols=2, frameon=False)
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)

    axes[1].bar(
        positions - width / 2,
        before_labels,
        width,
        color=ORANGE,
        label="Before remediation",
    )
    axes[1].bar(
        positions + width / 2,
        after_labels,
        width,
        color=after_colors,
        label="After remediation",
    )
    axes[1].axhline(3, color=DARK, linestyle="--", linewidth=1.2, label="Label cap")
    axes[1].axhline(
        old_label_mean,
        color=ORANGE,
        linestyle=":",
        linewidth=1.6,
        label=f"Old 11-role mean ({old_label_mean:.2f})",
    )
    axes[1].set_title("Emitted technical labels are now bounded")
    axes[1].set_ylabel("Number of emitted labels")
    axes[1].set_xlabel("Role")
    axes[1].set_xticks(positions, names, rotation=18, ha="right")
    axes[1].set_ylim(0, max(before_labels) + 2)
    axes[1].legend(ncols=2, frameon=False)
    axes[1].grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    _save(figure, output)


def plot_semantic_margins(data: dict[str, Any], output: Path) -> None:
    """Plot actual internship/permanent and technical/nontechnical separation."""

    roles = data["roles"]
    internship = np.array([role["internship_margin"] for role in roles], dtype=float)
    technical = np.array([role["technical_margin"] for role in roles], dtype=float)
    accepted = np.array([role["after_accepted"] for role in roles], dtype=bool)

    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.scatter(
        internship[~accepted],
        technical[~accepted],
        color=GRAY,
        marker="x",
        s=70,
        linewidths=2,
        label="Rejected",
    )
    axis.scatter(
        internship[accepted],
        technical[accepted],
        color=GREEN,
        marker="o",
        edgecolor=DARK,
        s=95,
        label="Recommended",
        zorder=3,
    )
    label_offsets = {
        "Litigation Counsel": (6, -18),
        "Technical Account Manager": (6, 8),
        "Presales Customer Engineer, Enterprise": (6, 8),
        "Presales Customer Engineer": (6, -20),
    }
    for role, x_value, y_value in zip(roles, internship, technical, strict=True):
        title = str(role["title"])
        axis.annotate(
            _short_title(title),
            (x_value, y_value),
            xytext=label_offsets.get(title, (6, 7)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(
        float(data["internship_margin_threshold"]),
        color=BLUE,
        linestyle="--",
        label="Internship margin threshold",
    )
    axis.axhline(
        float(data["technical_margin_threshold"]),
        color=ORANGE,
        linestyle="--",
        label="Technical margin threshold",
    )
    axis.set_title("Relative semantic evidence separates the accepted internships")
    axis.set_xlabel("Internship similarity minus permanent-role similarity")
    axis.set_ylabel("Technical similarity minus nontechnical similarity")
    axis.legend(frameon=False)
    axis.grid(color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    _save(figure, output)


def plot_evaluation_corpus(corpus: list[dict[str, Any]], output: Path) -> None:
    """Visualize the real version-controlled LLM benchmark composition."""

    related = np.array(
        [
            bool(case["expected_is_internship"] and case["expected_is_technical"])
            for case in corpus
        ],
        dtype=bool,
    )
    class_counts = np.array([related.sum(), (~related).sum()], dtype=int)
    labels = Counter(
        label
        for case, is_related in zip(corpus, related, strict=True)
        if is_related
        for label in case["expected_labels"]
    )
    label_names = list(labels)
    label_counts = np.array([labels[label] for label in label_names], dtype=int)

    figure, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    axes[0].bar(
        ["Technical internships", "Negative controls"],
        class_counts,
        color=[GREEN, GRAY],
    )
    axes[0].set_title("Balanced live-LLM evaluation classes")
    axes[0].set_ylabel("Number of labeled cases")
    axes[0].set_ylim(0, max(class_counts) + 3)
    for index, count in enumerate(class_counts):
        axes[0].text(index, count + 0.4, str(count), ha="center", fontweight="bold")
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)

    order = np.argsort(label_counts)
    axes[1].barh(
        np.array(label_names)[order],
        label_counts[order],
        color=BLUE,
    )
    axes[1].set_title("Expected labels in positive cases")
    axes[1].set_xlabel("Number of labeled occurrences")
    axes[1].set_ylabel("Technical label")
    axes[1].set_xlim(0, max(label_counts) + 1)
    axes[1].grid(axis="x", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    _save(figure, output)


def plot_scan_funnel(data: dict[str, Any], output: Path) -> None:
    """Plot actual job counts across the verified semantic scan stages."""

    stages = ["Downloaded", "UK/HK evaluated", "Recommended"]
    counts = np.array(
        [
            data["downloaded_jobs"],
            data["target_country_jobs_evaluated"],
            data["retained_recommendations"],
        ],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = axis.barh(stages[::-1], counts[::-1], color=[GREEN, BLUE, GRAY])
    axis.set_xscale("log")
    axis.set_title(f"Verified Cloudflare scan funnel — {data['captured_at']}")
    axis.set_xlabel("Number of jobs (log scale)")
    axis.set_ylabel("Pipeline stage")
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    for bar, count in zip(bars, counts[::-1], strict=True):
        axis.text(
            bar.get_width() * 1.08,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontweight="bold",
        )
    _save(figure, output)


def render_cli_screenshot(text: str, output: Path) -> None:
    """Render captured real CLI output as a readable documentation screenshot."""

    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    wrapped_lines = [
        wrapped
        for line in lines
        for wrapped in (
            textwrap.wrap(line, width=112, subsequent_indent="    ") or [""]
        )
    ]
    display = "\n".join(wrapped_lines[:42])
    height = max(7.0, min(11.0, len(display.splitlines()) * 0.27 + 1.5))
    figure = plt.figure(figsize=(14, height), facecolor=DARK)
    figure.text(
        0.025,
        0.965,
        "Internship Scanner — real diagnostic output",
        color="#F8FAFC",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.025,
        0.91,
        display,
        color="#E2E8F0",
        family="monospace",
        fontsize=10,
        va="top",
        linespacing=1.35,
    )
    _save(figure, output, facecolor=DARK)


def _short_title(title: str) -> str:
    return "\n".join(textwrap.wrap(title, width=22))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(figure: Figure, output: Path, *, facecolor: str = "white") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
        facecolor=facecolor,
        metadata={"Software": "Internship Scanner reproducible visualization"},
    )
    plt.close(figure)
