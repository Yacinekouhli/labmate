from datetime import UTC, datetime

import pytest

from labmate.tools.benchmarks import LocalBenchmarkBackend, lookup_benchmarks


def fixed_now() -> datetime:
    return datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


def test_local_benchmark_lookup_returns_ranked_tabular_context() -> None:
    backend = LocalBenchmarkBackend(now=fixed_now)

    result = lookup_benchmarks("tabular classification auc credit", backend=backend)

    assert result.backend == "local"
    assert result.retrieved_at == "2026-05-25T12:00:00Z"
    assert result.benchmarks[0].name == "Home Credit Default Risk"
    assert result.benchmarks[0].metric == "roc_auc"
    assert result.benchmarks[0].target == "TARGET"
    assert result.benchmarks[0].provenance_url == result.benchmarks[0].url
    assert "relational" in result.benchmarks[0].tags
    assert any("matched" in signal for signal in result.benchmarks[0].matched_signals)
    assert "curated and incomplete" in result.warnings[0]


def test_local_benchmark_lookup_returns_forecasting_metric_context() -> None:
    backend = LocalBenchmarkBackend(now=fixed_now)

    result = lookup_benchmarks("hierarchical retail forecasting", backend=backend)

    assert result.benchmarks[0].name == "M5 Forecasting - Accuracy"
    assert result.benchmarks[0].task_type == "hierarchical_time_series_forecasting"
    assert result.benchmarks[0].metric == "weighted_root_mean_squared_scaled_error"
    assert any(
        "rolling-origin" in suggestion for suggestion in result.benchmarks[0].baseline_suggestions
    )


def test_local_benchmark_lookup_reports_no_matches() -> None:
    backend = LocalBenchmarkBackend(now=fixed_now)

    result = lookup_benchmarks("quantum protein docking", backend=backend)

    assert result.benchmarks == ()
    assert result.warnings[0] == "No local benchmark entries matched the query."


def test_local_benchmark_lookup_rejects_invalid_max_results() -> None:
    backend = LocalBenchmarkBackend(now=fixed_now)

    with pytest.raises(ValueError, match="max_results"):
        lookup_benchmarks("tabular", backend=backend, max_results=0)
