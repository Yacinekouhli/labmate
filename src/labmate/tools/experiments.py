"""Read-only experiment ledger inspection helpers."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT_FILE_KINDS = {
    "experiment-log.md": "experiment_log",
    "experiment_log.md": "experiment_log",
    "experiment_log.tsv": "experiment_ledger",
    "experiments.csv": "experiment_ledger",
    "experiments.tsv": "experiment_ledger",
    "results.csv": "experiment_ledger",
    "results.tsv": "experiment_ledger",
    "runs.csv": "experiment_ledger",
    "runs.tsv": "experiment_ledger",
}
DEFAULT_LEDGER_NAMES = (
    "results.tsv",
    "experiments.tsv",
    "runs.tsv",
    "experiment_log.tsv",
    "results.csv",
    "experiments.csv",
    "runs.csv",
)
RECOMMENDED_LEDGER_COLUMNS = (
    "timestamp_utc",
    "commit",
    "experiment",
    "model_family",
    "features",
    "validation_strategy",
    "metric",
    "score",
    "score_direction",
    "status",
    "artifacts",
    "notes",
)
MAX_LEDGER_ROWS_TO_COUNT = 1_000


class ExperimentSummaryError(ValueError):
    """Raised when an experiment ledger cannot be summarized."""


def experiment_file_kind(path: Path) -> str | None:
    """Return the known experiment file kind for a path, if any."""

    return EXPERIMENT_FILE_KINDS.get(path.name.lower())


def summarize_ledger_table(
    path: str | Path,
    *,
    max_rows: int = MAX_LEDGER_ROWS_TO_COUNT,
) -> dict[str, Any]:
    """Return a compact schema/count summary for a TSV or CSV ledger."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive")

    ledger_path = Path(path)
    delimiter = _delimiter_for_path(ledger_path)
    try:
        with ledger_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                columns = next(reader)
            except StopIteration:
                return {
                    "format": ledger_path.suffix.lower().removeprefix("."),
                    "columns": [],
                    "completed_run_count": 0,
                    "row_count_status": "exact",
                    "read_status": "empty",
                }

            completed_run_count = 0
            row_count_status = "exact"
            for _row in reader:
                if completed_run_count >= max_rows:
                    row_count_status = "bounded"
                    break
                completed_run_count += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return {
            "format": ledger_path.suffix.lower().removeprefix("."),
            "columns": [],
            "completed_run_count": None,
            "row_count_status": "unknown",
            "read_status": "unreadable",
            "error": str(exc),
        }

    return {
        "format": ledger_path.suffix.lower().removeprefix("."),
        "columns": columns,
        "completed_run_count": completed_run_count,
        "row_count_status": row_count_status,
        "read_status": "ok",
    }


def summarize_experiments(
    path: str | Path,
    *,
    max_rows: int = MAX_LEDGER_ROWS_TO_COUNT,
) -> dict[str, Any]:
    """Summarize an experiment ledger file or a directory containing one."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive")

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ExperimentSummaryError(f"Experiment path does not exist: {root}")

    ledger_path = _resolve_ledger_path(root)
    if ledger_path is None:
        return {
            "kind": "experiment_summary",
            "path": str(root),
            "status": "not_found",
            "ledger": None,
            "metric_summary": None,
            "best_run": None,
            "latest_run": None,
            "status_counts": {},
            "warnings": [
                "No supported experiment ledger was found. Expected results.tsv or a "
                "similar CSV/TSV ledger."
            ],
            "recommended_next_actions": [
                "Create results.tsv using the research-brief experiment_tracking_plan "
                "before running models."
            ],
        }

    if ledger_path.suffix.lower() not in {".csv", ".tsv"}:
        return _metadata_only_summary(root, ledger_path)

    rows, ledger, warnings = _read_ledger_rows(ledger_path, max_rows=max_rows)
    metric_summary = _metric_summary(rows)
    best_run = _best_run(rows, metric_summary)
    latest_run = _latest_run(rows)
    status_counts = dict(Counter(row.get("status", "") or "unknown" for row in rows))

    recommended_next_actions = [
        f"Continue logging runs in {ledger['path']}.",
        "Compare new runs against the best validation score before changing model complexity.",
    ]
    if best_run is None:
        recommended_next_actions.insert(1, "Log a scored dummy or simple baseline next.")

    return {
        "kind": "experiment_summary",
        "path": str(root),
        "status": "ok" if ledger["read_status"] == "ok" else ledger["read_status"],
        "ledger": ledger,
        "metric_summary": metric_summary,
        "best_run": best_run,
        "latest_run": latest_run,
        "status_counts": status_counts,
        "warnings": warnings,
        "recommended_next_actions": recommended_next_actions,
    }


def _metadata_only_summary(root: Path, ledger_path: Path) -> dict[str, Any]:
    relative_path = _relative_path(root if root.is_dir() else ledger_path.parent, ledger_path)
    return {
        "kind": "experiment_summary",
        "path": str(root),
        "status": "metadata_only",
        "ledger": {
            "path": relative_path,
            "kind": experiment_file_kind(ledger_path),
            "read_status": "metadata_only",
        },
        "metric_summary": None,
        "best_run": None,
        "latest_run": None,
        "status_counts": {},
        "warnings": ["Experiment log is not a CSV/TSV ledger; summarize it manually."],
        "recommended_next_actions": [
            f"Continue documenting runs in {relative_path}.",
            "Create results.tsv if structured score comparison is needed.",
        ],
    }


def _resolve_ledger_path(root: Path) -> Path | None:
    if root.is_file():
        if experiment_file_kind(root):
            return root
        raise ExperimentSummaryError(f"Unsupported experiment file: {root}")

    if not root.is_dir():
        raise ExperimentSummaryError(f"Experiment path is not a file or directory: {root}")

    for name in DEFAULT_LEDGER_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate

    try:
        candidates = [
            path for path in root.iterdir() if path.is_file() and experiment_file_kind(path)
        ]
    except OSError as exc:
        raise ExperimentSummaryError(str(exc)) from exc

    return sorted(candidates, key=lambda path: path.name.lower())[0] if candidates else None


def _read_ledger_rows(
    ledger_path: Path,
    *,
    max_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any], list[str]]:
    delimiter = _delimiter_for_path(ledger_path)
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with ledger_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = list(reader.fieldnames or [])
            bounded = False
            for row in reader:
                if len(rows) >= max_rows:
                    bounded = True
                    warnings.append(
                        f"Ledger row scan stopped at max_rows={max_rows}; summary is bounded."
                    )
                    break
                rows.append({str(key): str(value or "") for key, value in row.items() if key})
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        ledger = summarize_ledger_table(ledger_path, max_rows=max_rows)
        return [], _ledger_metadata(ledger_path, ledger), [str(exc)]

    ledger = {
        "format": ledger_path.suffix.lower().removeprefix("."),
        "columns": columns,
        "completed_run_count": len(rows),
        "row_count_status": "bounded" if bounded else "exact",
        "read_status": "ok",
    }
    missing_columns = [
        column for column in RECOMMENDED_LEDGER_COLUMNS if column not in set(columns)
    ]
    if missing_columns:
        warnings.append("Ledger is missing recommended columns: " + ", ".join(missing_columns))

    return rows, _ledger_metadata(ledger_path, ledger), warnings


def _ledger_metadata(ledger_path: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(ledger)
    metadata["path"] = ledger_path.name
    metadata["kind"] = experiment_file_kind(ledger_path)
    return metadata


def _metric_summary(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    scored_rows = [_scored_row(row) for row in rows]
    scored_rows = [row for row in scored_rows if row is not None]
    if not scored_rows:
        return None

    metrics = Counter(row["metric"] or "unknown" for row in scored_rows)
    directions = Counter(
        row["score_direction"] or _metric_direction(row["metric"]) for row in scored_rows
    )
    return {
        "primary_metric": metrics.most_common(1)[0][0],
        "score_direction": directions.most_common(1)[0][0],
        "scored_run_count": len(scored_rows),
        "metrics_seen": dict(metrics),
    }


def _best_run(
    rows: list[dict[str, str]],
    metric_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if metric_summary is None:
        return None

    direction = str(metric_summary["score_direction"])
    scored_rows = [_scored_row(row) for row in rows]
    scored_rows = [row for row in scored_rows if row is not None]
    if not scored_rows:
        return None

    reverse = direction != "minimize"
    best = sorted(scored_rows, key=lambda row: row["score"], reverse=reverse)[0]
    return _run_summary(best["row"], score=best["score"])


def _latest_run(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return _run_summary(rows[-1])


def _scored_row(row: dict[str, str]) -> dict[str, Any] | None:
    score = _parse_float(row.get("score", ""))
    if score is None:
        return None
    return {
        "row": row,
        "score": score,
        "metric": row.get("metric", ""),
        "score_direction": row.get("score_direction", ""),
    }


def _run_summary(row: dict[str, str], *, score: float | None = None) -> dict[str, Any]:
    score_value = score if score is not None else _parse_float(row.get("score", ""))
    return {
        "timestamp_utc": row.get("timestamp_utc", ""),
        "commit": row.get("commit", ""),
        "experiment": row.get("experiment", ""),
        "model_family": row.get("model_family", ""),
        "metric": row.get("metric", ""),
        "score": score_value,
        "score_direction": row.get("score_direction", ""),
        "status": row.get("status", ""),
        "notes": row.get("notes", ""),
    }


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _metric_direction(metric: str) -> str:
    normalized = metric.casefold()
    if any(signal in normalized for signal in ("rmse", "mae", "loss", "error", "mape")):
        return "minimize"
    if any(signal in normalized for signal in ("auc", "accuracy", "f1", "map", "ndcg")):
        return "maximize"
    return "unknown"


def _delimiter_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    raise ExperimentSummaryError(f"Unsupported experiment ledger format: {path}")


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
