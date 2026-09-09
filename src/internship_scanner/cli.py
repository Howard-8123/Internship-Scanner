"""Command-line presentation for the aggregation engine."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from internship_scanner import __version__
from internship_scanner.aggregation import AggregationResult
from internship_scanner.bootstrap import build_engine, build_quality_evaluator
from internship_scanner.config import Settings
from internship_scanner.exceptions import InternshipScannerError
from internship_scanner.intelligence.evaluation import load_evaluation_cases

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure concise terminal logging."""

    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def log_result(
    result: AggregationResult,
    *,
    diagnostics: bool = False,
    top_k: int | None = None,
) -> None:
    """Log aggregation counts and every retained internship."""

    LOGGER.info("Discovered %d companies", result.discovered_company_count)
    LOGGER.info("Downloaded %d jobs", result.downloaded_job_count)
    LOGGER.info("Retained %d UK/Hong Kong internships", len(result.internships))
    recommendations = {item.job.url: item for item in result.recommendations}
    jobs = result.internships[:top_k] if top_k is not None else result.internships
    for job in jobs:
        recommendation = recommendations.get(job.url)
        intelligence = (
            f"score={recommendation.score:.3f}; "
            f"labels={', '.join(recommendation.labels) or 'unlabelled'}"
            if recommendation
            else "score unavailable"
        )
        LOGGER.info(
            "%s — %s [%s]\n%s\n%s\n%s (%s)\n%s\n",
            job.company,
            job.title,
            job.provider,
            job.category,
            intelligence,
            job.location,
            job.remote_status,
            job.url,
        )
        if diagnostics and recommendation is not None:
            semantic = recommendation.semantic
            analysis = recommendation.analysis
            LOGGER.info(
                "signals: semantic=%.4f internship=%.4f permanent=%.4f "
                "internship_margin=%.4f technical=%.4f nontechnical=%.4f "
                "technical_margin=%.4f ai=%.4f llm_confidence=%s "
                "llm_internship=%s",
                semantic.semantic_similarity,
                semantic.internship_similarity,
                semantic.permanent_similarity,
                semantic.internship_margin,
                semantic.technical_similarity,
                semantic.nontechnical_similarity,
                semantic.technical_margin,
                semantic.ai_similarity,
                f"{analysis.confidence:.4f}" if analysis else "disabled",
                f"{analysis.internship_relevance:.4f}" if analysis else "disabled",
            )
    if result.failures:
        LOGGER.warning("%d company boards failed", len(result.failures))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI before any settings, network, or model initialization."""

    parser = argparse.ArgumentParser(
        prog="internship-scanner",
        description="Find and rank technical internships from configured ATS boards.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="show component similarity and LLM scores for recommendations",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        help="print only the highest-ranked K recommendations",
    )
    subparsers = parser.add_subparsers(dest="command")
    evaluate = subparsers.add_parser(
        "evaluate",
        help="measure a real configured LLM against the labeled quality corpus",
    )
    evaluate.add_argument(
        "--dataset",
        type=Path,
        help="optional JSON corpus; defaults to the packaged 32-case benchmark",
    )
    evaluate.add_argument(
        "--minimum-accuracy",
        type=_unit_float,
        default=0.92,
        help="required binary technical-internship accuracy (default: 0.92)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run aggregation and return a process exit code."""

    arguments = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        if arguments.command == "evaluate":
            return _run_evaluation(
                settings, arguments.dataset, arguments.minimum_accuracy
            )
        mode = (
            "keyword fallback"
            if not settings.ai_enabled
            else (
                f"hybrid semantic + {settings.llm_provider} LLM"
                if settings.llm_provider != "none"
                else "semantic-only (LLM disabled)"
            )
        )
        LOGGER.info("Intelligence mode: %s", mode)
        result = build_engine(settings).aggregate()
        log_result(
            result,
            diagnostics=arguments.diagnostics,
            top_k=arguments.top_k,
        )
        return 1 if result.failures and result.downloaded_job_count == 0 else 0
    except (InternshipScannerError, OSError, ValueError) as error:
        LOGGER.error("Command failed: %s", error)
        return 1


def _run_evaluation(
    settings: Settings, dataset: Path | None, minimum_accuracy: float
) -> int:
    cases = load_evaluation_cases(dataset)
    LOGGER.info(
        "Running live %s evaluation on %d labeled cases",
        settings.llm_provider,
        len(cases),
    )
    report = build_quality_evaluator(settings).evaluate(cases)
    LOGGER.info(
        "LLM technical-internship accuracy: %.2f%% (%d/%d)",
        report.accuracy * 100,
        report.correct,
        report.total,
    )
    LOGGER.info(
        "Related-role label precision=%.2f%% recall=%.2f%% F1=%.2f%%",
        report.label_precision * 100,
        report.label_recall * 100,
        report.label_f1 * 100,
    )
    for failure in report.failures:
        LOGGER.warning(
            "%s expected_related=%s predicted_related=%s: %s",
            failure.case_id,
            failure.expected_related,
            failure.predicted_related,
            failure.reasoning,
        )
    if report.accuracy < minimum_accuracy:
        LOGGER.error(
            "Accuracy %.2f%% is below required %.2f%%",
            report.accuracy * 100,
            minimum_accuracy * 100,
        )
        return 1
    LOGGER.info("Accuracy gate passed (required %.2f%%)", minimum_accuracy * 100)
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed
