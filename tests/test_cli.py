"""Tests for command-line composition and logging."""

import logging
from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest

from internship_scanner.aggregation import AggregationResult, ProviderFailure
from internship_scanner.cli import build_parser, configure_logging, log_result, main
from internship_scanner.exceptions import ConfigurationError
from internship_scanner.intelligence.evaluation import EvaluationReport
from internship_scanner.models import Company, Job


def test_configure_logging() -> None:
    with patch("internship_scanner.cli.logging.basicConfig") as basic_config:
        configure_logging("DEBUG")
    basic_config.assert_called_once_with(
        level="DEBUG", format="%(levelname)s %(message)s"
    )


def test_log_result_includes_counts_jobs_and_failures(
    caplog: pytest.LogCaptureFixture, job_factory: Callable[..., Job]
) -> None:
    result = AggregationResult(
        2,
        10,
        (job_factory(),),
        (ProviderFailure(Company("Broken", "broken", "lever"), "failed"),),
    )
    with caplog.at_level(logging.INFO):
        log_result(result)
    assert "Discovered 2 companies" in caplog.text
    assert "Software Engineer Intern" in caplog.text
    assert "1 company boards failed" in caplog.text


@patch("internship_scanner.cli.log_result")
@patch("internship_scanner.cli.build_engine")
@patch("internship_scanner.cli.Settings.from_env")
def test_main_returns_success(settings_from_env: Mock, build: Mock, log: Mock) -> None:
    result = AggregationResult(1, 1, (), ())
    build.return_value.aggregate.return_value = result
    settings_from_env.return_value = Mock(
        log_level="INFO", ai_enabled=True, llm_provider="none"
    )
    assert main([]) == 0
    log.assert_called_once_with(result, diagnostics=False, top_k=None)


@patch("internship_scanner.cli.build_engine")
@patch("internship_scanner.cli.Settings.from_env")
def test_main_returns_failure_when_every_board_failed(
    settings_from_env: Mock, build: Mock
) -> None:
    settings_from_env.return_value = Mock(
        log_level="INFO", ai_enabled=True, llm_provider="none"
    )
    failure = ProviderFailure(Company("Broken", "broken", "lever"), "failed")
    build.return_value.aggregate.return_value = AggregationResult(1, 0, (), (failure,))
    assert main([]) == 1


@patch("internship_scanner.cli.Settings.from_env")
def test_main_handles_expected_error(
    settings_from_env: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    settings_from_env.side_effect = ConfigurationError("invalid")
    with caplog.at_level(logging.ERROR):
        assert main([]) == 1
    assert "Command failed: invalid" in caplog.text


def test_help_exits_before_initializing_the_application(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("internship_scanner.cli.Settings.from_env") as settings,
        pytest.raises(SystemExit) as exit_info,
    ):
        main(["--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out.lower()
    settings.assert_not_called()


def test_version_exits_before_initializing_the_application(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("internship_scanner.cli.Settings.from_env") as settings,
        pytest.raises(SystemExit) as exit_info,
    ):
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "internship-scanner 0.3.0\n"
    settings.assert_not_called()


def test_parser_rejects_invalid_top_k() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--top-k", "0"])


@patch("internship_scanner.cli.load_evaluation_cases")
@patch("internship_scanner.cli.build_quality_evaluator")
@patch("internship_scanner.cli.Settings.from_env")
def test_evaluate_command_enforces_accuracy_gate(
    settings_from_env: Mock,
    build_evaluator: Mock,
    load_cases: Mock,
) -> None:
    settings_from_env.return_value = Mock(log_level="INFO", llm_provider="openai")
    load_cases.return_value = tuple(range(32))
    build_evaluator.return_value.evaluate.return_value = EvaluationReport(
        total=32,
        correct=29,
        accuracy=29 / 32,
        label_precision=1.0,
        label_recall=1.0,
        label_f1=1.0,
        failures=(),
    )
    assert main(["evaluate", "--minimum-accuracy", "0.92"]) == 1

    build_evaluator.return_value.evaluate.return_value = EvaluationReport(
        total=32,
        correct=30,
        accuracy=30 / 32,
        label_precision=1.0,
        label_recall=1.0,
        label_f1=1.0,
        failures=(),
    )
    assert main(["evaluate", "--minimum-accuracy", "0.92"]) == 0
