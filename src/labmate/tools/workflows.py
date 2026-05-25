"""Read-only workflow helpers that compose Labmate research tools."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from labmate.tools.benchmarks import BenchmarkLookupResult, LocalBenchmarkBackend, lookup_benchmarks
from labmate.tools.datasets import inspect_local_dataset

METRIC_PATTERNS = (
    ("roc_auc", re.compile(r"\b(?:roc[-_ ]?auc|auc)\b", re.IGNORECASE)),
    ("log_loss", re.compile(r"\b(?:log[-_ ]?loss|logloss)\b", re.IGNORECASE)),
    ("rmse", re.compile(r"\b(?:rmse|root mean squared error)\b", re.IGNORECASE)),
    ("mae", re.compile(r"\b(?:mae|mean absolute error)\b", re.IGNORECASE)),
    ("accuracy", re.compile(r"\baccuracy\b", re.IGNORECASE)),
    ("f1", re.compile(r"\bf1(?: score)?\b", re.IGNORECASE)),
    ("map_at_k", re.compile(r"\b(?:map@k|mean average precision)\b", re.IGNORECASE)),
    ("ndcg", re.compile(r"\bndcg\b", re.IGNORECASE)),
    ("mape", re.compile(r"\bmape\b", re.IGNORECASE)),
    ("smape", re.compile(r"\bsmape\b", re.IGNORECASE)),
)


def build_research_brief(
    path: str | Path,
    *,
    task_hint: str | None = None,
    benchmark_query: str | None = None,
    sample_size: int = 3,
    max_profile_rows: int = 250_000,
    max_benchmarks: int = 3,
) -> dict[str, Any]:
    """Build a concise first-pass brief for an unknown local ML dataset."""

    if max_benchmarks < 1:
        raise ValueError("max_benchmarks must be positive")

    dataset = inspect_local_dataset(
        path,
        sample_size=sample_size,
        max_profile_rows=max_profile_rows,
    )
    dataset_summary = _dataset_summary(dataset)
    inferred_task = _infer_task(dataset_summary, task_hint=task_hint)
    query = benchmark_query or _benchmark_query(inferred_task)
    benchmarks = lookup_benchmarks(
        query,
        backend=LocalBenchmarkBackend(),
        max_results=max_benchmarks,
    )
    research_plan = _research_plan(
        path=path,
        benchmark_query=query,
        inferred_task=inferred_task,
        max_benchmarks=max_benchmarks,
    )

    return {
        "kind": "ml_research_brief",
        "dataset_path": str(path),
        "task_hint": task_hint,
        "benchmark_query": query,
        "inferred_task": inferred_task,
        "dataset_summary": dataset_summary,
        "benchmark_context": benchmarks.to_dict(),
        "evidence": _evidence(dataset_summary, benchmarks),
        "modeling_plan": _modeling_plan(dataset_summary, inferred_task, benchmarks),
        "research_plan": research_plan,
        "recommended_next_commands": _recommended_next_commands(research_plan),
        "implementation_checklist": _implementation_checklist(dataset_summary, benchmarks),
        "warnings": _brief_warnings(dataset_summary, benchmarks, inferred_task),
    }


def _dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    if dataset["kind"] in {"local_dataset_directory", "local_dataset_archive"}:
        files = [_file_summary(file_info) for file_info in dataset["files"]]
        relations = dataset.get("relations", {})
        files = _suppress_expected_file_warnings(files, relations)
        return {
            "kind": dataset["kind"],
            "path": dataset["path"],
            "files": files,
            "context_files": _context_files(dataset.get("context_files", [])),
            "relations": relations,
            "warnings": _collect_dataset_warnings(
                files,
                dataset.get("warnings", []),
                relations=relations,
            ),
        }

    file_summary = _file_summary(dataset)
    return {
        "kind": dataset["kind"],
        "path": dataset["path"],
        "files": [file_summary],
        "context_files": [],
        "relations": {},
        "warnings": _collect_dataset_warnings(
            [file_summary],
            dataset.get("warnings", []),
            relations={},
        ),
    }


def _file_summary(file_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": file_info["file_name"],
        "row_count": file_info["row_count"],
        "row_count_status": file_info["row_count_status"],
        "profiled_row_count": file_info["profiled_row_count"],
        "column_count": len(file_info["columns"]),
        "columns": [_column_summary(column) for column in file_info["columns"]],
        "target_column_hints": list(file_info.get("target_column_hints", [])),
        "leakage_risk_hints": list(file_info.get("leakage_risk_hints", [])),
        "warnings": list(file_info.get("warnings", [])),
    }


def _column_summary(column: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": column["name"],
        "inferred_type": column["inferred_type"],
        "missing_rate": column["missing_rate"],
        "unique_values_profiled": column["unique_values_profiled"],
        "unique_values_truncated": column["unique_values_truncated"],
        "top_values": list(column.get("top_values", [])),
        "role_hints": list(column["role_hints"]),
    }


def _context_files(context_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "file_name": context_file["file_name"],
            "kind": context_file["kind"],
            "size_bytes": context_file["size_bytes"],
            "snippet": context_file["snippet"],
            "json_keys": list(context_file.get("json_keys", [])),
        }
        for context_file in context_files
    ]


def _collect_dataset_warnings(
    files: list[dict[str, Any]],
    directory_warnings: list[str],
    *,
    relations: dict[str, Any],
) -> list[dict[str, str]]:
    expected_without_target = {
        file_name
        for file_name in (relations.get("test_file"), relations.get("sample_submission_file"))
        if isinstance(file_name, str)
    }
    warnings = [{"scope": "dataset", "message": warning} for warning in directory_warnings]
    for file_info in files:
        warnings.extend(
            {"scope": file_info["file_name"], "message": warning}
            for warning in file_info.get("warnings", [])
            if not _is_expected_missing_target_warning(
                file_name=file_info["file_name"],
                warning=warning,
                expected_without_target=expected_without_target,
            )
        )
    return warnings


def _suppress_expected_file_warnings(
    files: list[dict[str, Any]],
    relations: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_without_target = {
        file_name
        for file_name in (relations.get("test_file"), relations.get("sample_submission_file"))
        if isinstance(file_name, str)
    }
    normalized = []
    for file_info in files:
        file_info = dict(file_info)
        file_info["warnings"] = [
            warning
            for warning in file_info.get("warnings", [])
            if not _is_expected_missing_target_warning(
                file_name=file_info["file_name"],
                warning=warning,
                expected_without_target=expected_without_target,
            )
        ]
        normalized.append(file_info)
    return normalized


def _is_expected_missing_target_warning(
    *,
    file_name: str,
    warning: str,
    expected_without_target: set[str],
) -> bool:
    return (
        file_name in expected_without_target
        and warning == "No obvious target column detected from column names."
    )


def _infer_task(dataset_summary: dict[str, Any], *, task_hint: str | None) -> dict[str, Any]:
    target_columns = _target_columns(dataset_summary)
    if task_hint:
        return {
            "task_type": task_hint,
            "confidence": "user_supplied",
            "target_columns": target_columns,
            "reason": "task_hint was provided by the caller",
        }

    train_file = _training_file_summary(dataset_summary)
    target_profile = _target_profile(train_file, target_columns)
    if target_profile is None:
        return {
            "task_type": "tabular modeling",
            "confidence": "low",
            "target_columns": target_columns,
            "reason": "no clear target column was found",
        }

    inferred_type = target_profile["inferred_type"]
    unique_count = target_profile["unique_values_profiled"]
    if inferred_type in {"boolean", "string"} or unique_count <= 20:
        task_type = "tabular classification"
        reason = "target appears categorical or low-cardinality"
    else:
        task_type = "tabular regression"
        reason = "target appears numeric with many profiled values"

    confidence = "medium" if train_file.get("profiled_row_count", 0) >= 50 else "low"
    return {
        "task_type": task_type,
        "confidence": confidence,
        "target_columns": target_columns,
        "reason": reason,
    }


def _target_columns(dataset_summary: dict[str, Any]) -> list[str]:
    relations = dataset_summary.get("relations", {})
    likely_targets = relations.get("likely_target_columns")
    if isinstance(likely_targets, list) and likely_targets:
        return [str(column) for column in likely_targets]

    columns: list[str] = []
    for file_info in dataset_summary["files"]:
        for hint in file_info.get("target_column_hints", []):
            column = hint.get("column")
            if isinstance(column, str) and column not in columns:
                columns.append(column)
    return columns


def _training_file_summary(dataset_summary: dict[str, Any]) -> dict[str, Any]:
    train_file = dataset_summary.get("relations", {}).get("train_file")
    if isinstance(train_file, str):
        for file_info in dataset_summary["files"]:
            if file_info["file_name"] == train_file:
                return file_info
    return dataset_summary["files"][0]


def _target_profile(
    train_file: dict[str, Any],
    target_columns: list[str],
) -> dict[str, Any] | None:
    if not target_columns:
        return None
    target_column = target_columns[0]
    for column in train_file["columns"]:
        if column["name"] == target_column:
            return column
    return None


def _target_distribution(
    dataset_summary: dict[str, Any],
    target_columns: list[str],
) -> dict[str, Any] | None:
    train_file = _training_file_summary(dataset_summary)
    target_profile = _target_profile(train_file, target_columns)
    if target_profile is None:
        return None

    return {
        "column": target_profile["name"],
        "profiled_row_count": train_file.get("profiled_row_count", 0),
        "missing_rate": target_profile["missing_rate"],
        "unique_values_profiled": target_profile["unique_values_profiled"],
        "unique_values_truncated": target_profile["unique_values_truncated"],
        "top_values": list(target_profile.get("top_values", [])),
    }


def _benchmark_query(inferred_task: dict[str, Any]) -> str:
    task_type = str(inferred_task["task_type"]).casefold()
    if inferred_task["confidence"] == "user_supplied":
        return str(inferred_task["task_type"])
    if "regression" in task_type:
        return "tabular regression rmse kaggle"
    if "classification" in task_type:
        return "tabular classification auc kaggle"
    return "tabular machine learning kaggle"


def _evidence(
    dataset_summary: dict[str, Any],
    benchmarks: BenchmarkLookupResult,
) -> dict[str, Any]:
    target_columns = _target_columns(dataset_summary)
    return {
        "dataset_files": [
            {
                "file_name": file_info["file_name"],
                "row_count": file_info["row_count"],
                "column_count": file_info["column_count"],
            }
            for file_info in dataset_summary["files"]
        ],
        "target_columns": target_columns,
        "target_distribution": _target_distribution(dataset_summary, target_columns),
        "benchmark_urls": [
            benchmark.provenance_url or benchmark.url for benchmark in benchmarks.benchmarks
        ],
        "context_files": [
            {"file_name": context_file["file_name"], "kind": context_file["kind"]}
            for context_file in dataset_summary["context_files"]
        ],
        "metric_hints": _metric_hints(dataset_summary),
        "dataset_warning_count": len(dataset_summary["warnings"]),
    }


def _modeling_plan(
    dataset_summary: dict[str, Any],
    inferred_task: dict[str, Any],
    benchmarks: BenchmarkLookupResult,
) -> dict[str, Any]:
    target_columns = _target_columns(dataset_summary)
    feature_columns = _feature_columns(dataset_summary, target_columns)
    id_columns = _id_columns(dataset_summary)
    target_distribution = _target_distribution(dataset_summary, target_columns)
    metric_hints = _metric_hints(dataset_summary)
    benchmark_metric = benchmarks.benchmarks[0].metric if benchmarks.benchmarks else None
    metric = metric_hints[0]["metric"] if metric_hints else benchmark_metric
    readiness = _modeling_readiness(target_columns, feature_columns)

    return {
        "readiness": readiness,
        "target_columns": target_columns,
        "target_distribution": target_distribution,
        "id_columns": id_columns,
        "feature_columns": feature_columns,
        "suggested_metric": metric,
        "validation_strategy": _validation_strategy(
            inferred_task=inferred_task,
            metric=metric,
            ready=readiness == "ready_for_baseline",
        ),
        "baseline_experiments": _baseline_experiments(
            inferred_task=inferred_task,
            metric=metric,
            ready=readiness == "ready_for_baseline",
            benchmarks=benchmarks,
        ),
    }


def _feature_columns(
    dataset_summary: dict[str, Any],
    target_columns: list[str],
) -> list[str]:
    schema_alignment = dataset_summary.get("relations", {}).get("train_test_schema_alignment")
    if isinstance(schema_alignment, dict):
        common_features = schema_alignment.get("common_feature_columns")
        if isinstance(common_features, list) and common_features:
            return [str(column) for column in common_features]

    target_column_set = set(target_columns)
    id_column_set = set(_id_columns(dataset_summary))
    train_file = _training_file_summary(dataset_summary)
    return [
        column["name"]
        for column in train_file["columns"]
        if column["name"] not in target_column_set and column["name"] not in id_column_set
    ]


def _id_columns(dataset_summary: dict[str, Any]) -> list[str]:
    schema_alignment = dataset_summary.get("relations", {}).get("train_test_schema_alignment")
    if isinstance(schema_alignment, dict):
        id_columns = schema_alignment.get("id_columns")
        if isinstance(id_columns, list) and id_columns:
            return [str(column) for column in id_columns]

    train_file = _training_file_summary(dataset_summary)
    return [
        column["name"] for column in train_file["columns"] if "id" in column.get("role_hints", [])
    ]


def _modeling_readiness(target_columns: list[str], feature_columns: list[str]) -> str:
    if not target_columns:
        return "needs_target_confirmation"
    if not feature_columns:
        return "needs_feature_confirmation"
    return "ready_for_baseline"


def _validation_strategy(
    *,
    inferred_task: dict[str, Any],
    metric: str | None,
    ready: bool,
) -> dict[str, Any]:
    if not ready:
        return {
            "name": "not_ready",
            "reason": "confirm target and feature columns before creating a validation split",
        }

    task_type = str(inferred_task["task_type"]).casefold()
    if "classification" in task_type:
        return {
            "name": "stratified_k_fold",
            "reason": "classification targets need class-balance preservation across folds",
            "metric": metric or "task_metric_from_rules",
        }
    if "time" in task_type or "forecast" in task_type:
        return {
            "name": "rolling_origin",
            "reason": "forecasting tasks need chronological validation to avoid future leakage",
            "metric": metric or "task_metric_from_rules",
        }
    return {
        "name": "k_fold",
        "reason": "tabular regression/modeling baseline with no detected time split",
        "metric": metric or "task_metric_from_rules",
    }


def _baseline_experiments(
    *,
    inferred_task: dict[str, Any],
    metric: str | None,
    ready: bool,
    benchmarks: BenchmarkLookupResult,
) -> list[dict[str, Any]]:
    if not ready:
        return [
            {
                "name": "schema_confirmation",
                "model_family": "none",
                "purpose": "confirm target, ID, feature, metric, and submission columns",
                "expected_output": "documented schema decision before modeling",
            }
        ]

    task_type = str(inferred_task["task_type"]).casefold()
    experiments = [
        {
            "name": "dummy_baseline",
            "model_family": "dummy",
            "purpose": "establish a floor before feature engineering or heavy models",
            "metric": metric or "task_metric_from_rules",
        },
        {
            "name": "linear_baseline",
            "model_family": (
                "logistic_regression" if "classification" in task_type else "regularized_linear"
            ),
            "purpose": "validate preprocessing and feature handling with a simple model",
            "metric": metric or "task_metric_from_rules",
        },
        {
            "name": "tree_boosting_baseline",
            "model_family": "gradient_boosting",
            "purpose": "strong first tabular baseline after leakage checks pass",
            "metric": metric or "task_metric_from_rules",
        },
    ]

    if benchmarks.benchmarks:
        experiments[2]["benchmark_reference"] = benchmarks.benchmarks[0].name
    return experiments


def _metric_hints(dataset_summary: dict[str, Any]) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for context_file in dataset_summary["context_files"]:
        snippet = str(context_file.get("snippet", ""))
        for metric, pattern in METRIC_PATTERNS:
            match = pattern.search(snippet)
            if not match:
                continue
            key = (metric, context_file["file_name"])
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                {
                    "metric": metric,
                    "source_file": context_file["file_name"],
                    "matched_text": match.group(0),
                }
            )
    return hints


def _research_plan(
    *,
    path: str | Path,
    benchmark_query: str,
    inferred_task: dict[str, Any],
    max_benchmarks: int,
) -> list[dict[str, Any]]:
    dataset_path = str(path)
    task_type = str(inferred_task["task_type"])
    literature_query = f"{task_type} baseline gradient boosting"
    docs_query = "sklearn ColumnTransformer pipeline"
    github_query = f"{task_type} kaggle baseline"
    return [
        _research_action(
            priority=1,
            tool="dataset_inspect",
            command=_command("labmate", "dataset-inspect", dataset_path, "--sample-size", "5"),
            arguments={"path": dataset_path, "sample_size": 5},
            purpose="Verify schema, split alignment, target hints, and leakage warnings.",
            evidence_to_extract=(
                "target columns",
                "train/test feature alignment",
                "sample submission format",
                "dataset warnings",
            ),
        ),
        _research_action(
            priority=2,
            tool="benchmark_lookup",
            command=_command(
                "labmate", "benchmark-lookup", benchmark_query, "--max-results", str(max_benchmarks)
            ),
            arguments={"query": benchmark_query, "max_results": max_benchmarks},
            purpose="Confirm metric, task protocol, baseline families, and common pitfalls.",
            query=benchmark_query,
            evidence_to_extract=(
                "metric",
                "validation protocol",
                "baseline suggestions",
                "pitfalls",
            ),
        ),
        _research_action(
            priority=3,
            tool="literature_search",
            command=_command(
                "labmate",
                "literature-search",
                literature_query,
                "--max-results",
                "5",
            ),
            arguments={"query": literature_query, "max_results": 5},
            purpose="Find papers that justify baseline and modeling choices.",
            query=literature_query,
            evidence_to_extract=("paper URLs", "method names", "relevance signals"),
        ),
        _research_action(
            priority=4,
            tool="citation_graph",
            command=_command("labmate", "citation-graph", "arxiv:1603.02754", "--max-results", "3"),
            arguments={"paper_id": "arxiv:1603.02754", "max_results": 3},
            purpose="Inspect citation context for gradient-boosted tree baselines.",
            evidence_to_extract=("related papers", "citation edges"),
        ),
        _research_action(
            priority=5,
            tool="docs_fetch",
            command=_command("labmate", "docs-fetch", docs_query, "--max-results", "3"),
            arguments={"query": docs_query, "max_results": 3},
            purpose="Fetch framework documentation for a robust preprocessing pipeline.",
            query=docs_query,
            evidence_to_extract=("official docs URLs", "API names", "version caveats"),
        ),
        _research_action(
            priority=6,
            tool="github_find_examples",
            command=_command("labmate", "github-find-examples", github_query, "--max-results", "3"),
            arguments={"query": github_query, "max_results": 3},
            purpose="Find implementation examples before editing model code.",
            query=github_query,
            evidence_to_extract=("repository URLs", "file snippets", "matched signals"),
        ),
    ]


def _research_action(
    *,
    priority: int,
    tool: str,
    command: str,
    arguments: dict[str, Any],
    purpose: str,
    evidence_to_extract: tuple[str, ...],
    query: str | None = None,
) -> dict[str, Any]:
    action = {
        "priority": priority,
        "tool": tool,
        "command": command,
        "arguments": arguments,
        "purpose": purpose,
        "evidence_to_extract": list(evidence_to_extract),
    }
    if query is not None:
        action["query"] = query
    return action


def _recommended_next_commands(research_plan: list[dict[str, Any]]) -> list[str]:
    return [str(action["command"]) for action in research_plan]


def _command(*parts: str) -> str:
    return shlex.join(parts)


def _implementation_checklist(
    dataset_summary: dict[str, Any],
    benchmarks: BenchmarkLookupResult,
) -> list[str]:
    checklist = [
        "Confirm the competition metric and submission format from the source page.",
        "Create a validation split before tuning against any public leaderboard feedback.",
        "Start with a dummy or simple linear baseline before adding heavier models.",
        "Keep preprocessing identical for train and test feature columns.",
    ]
    if dataset_summary["context_files"]:
        checklist.insert(
            1,
            "Use local context files to verify metric, rules, and submission columns.",
        )
    if dataset_summary["warnings"]:
        checklist.append("Resolve or explicitly justify dataset warnings before modeling.")

    for benchmark in benchmarks.benchmarks[:1]:
        checklist.extend(benchmark.baseline_suggestions[:2])
        checklist.extend(f"Watch for: {pitfall}" for pitfall in benchmark.pitfalls[:2])

    return checklist


def _brief_warnings(
    dataset_summary: dict[str, Any],
    benchmarks: BenchmarkLookupResult,
    inferred_task: dict[str, Any],
) -> list[str]:
    warnings = [
        "Research brief is a planning aid; verify competition rules and metrics at source URLs."
    ]
    warnings.extend(warning["message"] for warning in dataset_summary["warnings"])
    warnings.extend(_target_distribution_warnings(dataset_summary, inferred_task))
    warnings.extend(benchmarks.warnings)
    return warnings


def _target_distribution_warnings(
    dataset_summary: dict[str, Any],
    inferred_task: dict[str, Any],
) -> list[str]:
    task_type = str(inferred_task["task_type"]).casefold()
    if "classification" not in task_type:
        return []

    distribution = _target_distribution(dataset_summary, _target_columns(dataset_summary))
    if distribution is None:
        return []

    warnings = []
    if distribution["missing_rate"] > 0:
        warnings.append(
            f"Target {distribution['column']} has missing values in profiled rows; "
            "confirm label handling before training."
        )

    top_values = distribution["top_values"]
    if not top_values:
        return warnings

    if len(top_values) == 1:
        warnings.append(
            f"Target {distribution['column']} has only one observed non-missing value in "
            "profiled rows; confirm this is the intended training file."
        )
        return warnings

    if distribution["unique_values_truncated"]:
        return warnings

    dominant = top_values[0]
    dominant_rate = float(dominant["rate"])
    if dominant_rate >= 0.9:
        warnings.append(
            f"Target {distribution['column']} is imbalanced in profiled rows; "
            f"dominant value {dominant['value']!r} has observed rate {dominant_rate:.3f}."
        )
    return warnings
