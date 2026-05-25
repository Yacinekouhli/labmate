"""Kaggle competition workspace workflow helpers."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from labmate.contracts import ExitCode
from labmate.tools.datasets import DatasetInspectionError, inspect_local_dataset
from labmate.tools.experiments import RECOMMENDED_LEDGER_COLUMNS
from labmate.tools.workflows import build_research_brief

KAGGLE_HOSTS = {"kaggle.com", "www.kaggle.com"}
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
GENERATED_MARKER = "<!-- labmate:generated -->"
DEFAULT_WORKSPACE_DIRS = ("data", "runs", "submissions", "reports", ".labmate")
MAX_FILE_LIST = 200
MAX_OUTPUT_CHARS = 2_000
BASELINE_STRATEGIES = {"auto", "target_mode", "target_mean", "sample_default", "zero"}


class KaggleWorkflowError(ValueError):
    """Raised when a Kaggle workflow request is invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "kaggle_workflow_error",
        exit_code: ExitCode = ExitCode.TOOL_ERROR,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.retryable = retryable
        self.details = details or {}


def start_kaggle_competition(
    competition: str,
    *,
    workspace: str | Path | None = None,
    data_dir: str | Path = "data",
    download: bool = True,
    force_download: bool = False,
    sample_size: int = 3,
    max_profile_rows: int = 250_000,
) -> dict[str, Any]:
    """Create or update a local Kaggle workspace and build a first-pass plan."""

    slug = normalize_competition_slug(competition)
    workspace_path = Path(workspace or slug).expanduser().resolve()
    if workspace_path.exists() and not workspace_path.is_dir():
        raise KaggleWorkflowError(
            "workspace must be a directory path",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
            details={"workspace": str(workspace_path)},
        )
    if sample_size < 0:
        raise KaggleWorkflowError(
            "sample_size must be non-negative",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
        )
    if max_profile_rows <= 0:
        raise KaggleWorkflowError(
            "max_profile_rows must be positive",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
        )

    workspace_created = not workspace_path.exists()
    workspace_path.mkdir(parents=True, exist_ok=True)
    data_path = _resolve_child_path(workspace_path, data_dir)
    directory_actions = _ensure_directories(workspace_path, data_path)

    file_actions: list[dict[str, Any]] = []
    file_actions.append(
        _ensure_text_file(
            workspace_path / ".gitignore",
            _gitignore_text(),
            root=workspace_path,
            update_generated=False,
            always_update=False,
        )
    )
    file_actions.append(
        _ensure_text_file(
            workspace_path / "results.tsv",
            "\t".join(RECOMMENDED_LEDGER_COLUMNS) + "\n",
            root=workspace_path,
            update_generated=False,
            always_update=False,
        )
    )
    file_actions.append(
        _ensure_text_file(
            workspace_path / "program.md",
            _program_text(slug),
            root=workspace_path,
            update_generated=False,
            always_update=False,
        )
    )

    download_result = _download_competition(
        slug,
        data_path=data_path,
        download=download,
        force_download=force_download,
    )
    data_result = _inspect_available_data(
        workspace_path=workspace_path,
        data_path=data_path,
        sample_size=sample_size,
        max_profile_rows=max_profile_rows,
    )

    metadata = _metadata(slug, competition, workspace_path, data_path, data_result)
    file_actions.append(
        _ensure_text_file(
            workspace_path / ".labmate" / "competition.json",
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            root=workspace_path,
            update_generated=False,
            always_update=True,
        )
    )
    file_actions.append(
        _ensure_text_file(
            workspace_path / "reports" / "competition_brief.md",
            _brief_text(slug, data_result),
            root=workspace_path,
            update_generated=True,
            always_update=False,
        )
    )

    warnings = _workflow_warnings(download_result, data_result)
    return {
        "kind": "kaggle_competition_workspace",
        "competition": {
            "input": competition,
            "slug": slug,
            "url": f"https://www.kaggle.com/competitions/{slug}",
        },
        "workspace": {
            "path": str(workspace_path),
            "created": workspace_created,
            "data_dir": _relative_to(workspace_path, data_path),
            "directories": directory_actions,
            "files": file_actions,
        },
        "kaggle_access": _kaggle_access_summary(),
        "download": download_result,
        "data": data_result,
        "experiment_tracking": {
            "ledger_path": "results.tsv",
            "columns": list(RECOMMENDED_LEDGER_COLUMNS),
            "created": any(
                action["path"] == "results.tsv" and action["action"] == "written"
                for action in file_actions
            ),
        },
        "agent_handoff": _agent_handoff(slug, workspace_path, data_result),
        "next_actions": _next_actions(slug, workspace_path, data_path, data_result),
        "submission_policy": {
            "status": "manual_approval_required",
            "message": (
                "Labmate does not auto-submit. A host agent must ask for explicit user "
                "approval before running Kaggle submission commands or MCP submission tools."
            ),
            "approved_surfaces": [
                "kaggle competitions submit -c <slug> -f <file> -m <message>",
                "Kaggle MCP submit_to_competition with explicit user approval",
            ],
        },
        "warnings": warnings,
    }


def create_kaggle_baseline(
    workspace: str | Path,
    *,
    competition: str | None = None,
    data_dir: str | Path = "data",
    strategy: str = "auto",
    run_name: str | None = None,
    overwrite: bool = False,
    sample_size: int = 3,
    max_profile_rows: int = 250_000,
) -> dict[str, Any]:
    """Create a sample-submission-compatible constant baseline and log it."""

    if strategy not in BASELINE_STRATEGIES:
        raise KaggleWorkflowError(
            f"Unsupported baseline strategy: {strategy}",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
            details={"strategy": strategy, "supported": sorted(BASELINE_STRATEGIES)},
        )

    workspace_path = _resolve_workspace(workspace)
    data_path = _resolve_child_path(workspace_path, data_dir)
    if not data_path.is_dir():
        raise KaggleWorkflowError(
            "baseline generation requires an extracted local data directory",
            code="data_not_ready",
            exit_code=ExitCode.TOOL_ERROR,
            details={"data_dir": _relative_to(workspace_path, data_path)},
        )

    workspace_slug = (
        normalize_competition_slug(competition)
        if competition
        else _competition_slug_from_workspace(workspace_path)
    )
    run_id = _run_id(run_name)
    submission_path = workspace_path / "submissions" / f"{run_id}.csv"
    manifest_path = workspace_path / "runs" / run_id / "manifest.json"
    if submission_path.exists() and not overwrite:
        raise KaggleWorkflowError(
            "submission file already exists; pass overwrite=true to replace it",
            code="artifact_exists",
            exit_code=ExitCode.USAGE_ERROR,
            details={"submission_path": _relative_to(workspace_path, submission_path)},
        )

    data_result = _inspect_available_data(
        workspace_path=workspace_path,
        data_path=data_path,
        sample_size=sample_size,
        max_profile_rows=max_profile_rows,
    )
    if data_result["status"] != "inspected":
        raise KaggleWorkflowError(
            "baseline generation requires inspectable train/test/sample_submission files",
            code="data_not_ready",
            exit_code=ExitCode.TOOL_ERROR,
            details={"data": data_result},
        )

    inspection = data_result["inspection"]
    if not isinstance(inspection, dict) or inspection.get("kind") != "local_dataset_directory":
        raise KaggleWorkflowError(
            "baseline generation currently requires extracted top-level CSV/TSV files",
            code="unsupported_dataset_layout",
            exit_code=ExitCode.TOOL_ERROR,
            details={"inspected_path": data_result.get("inspected_path")},
        )

    baseline = _baseline_predictions(
        workspace_path=workspace_path,
        data_path=data_path,
        inspection=inspection,
        strategy=strategy,
    )
    _write_submission(
        submission_path,
        fieldnames=baseline["fieldnames"],
        rows=baseline["rows"],
    )
    validation = validate_kaggle_submission(
        submission_path,
        workspace=workspace_path,
        data_dir=data_dir,
    )
    research_brief = data_result["research_brief"]
    manifest = _baseline_manifest(
        workspace_path=workspace_path,
        competition=workspace_slug,
        run_id=run_id,
        strategy=strategy,
        baseline=baseline,
        submission_path=submission_path,
        validation=validation,
        research_brief=research_brief,
    )
    _write_json(manifest_path, manifest)
    ledger_row = _baseline_ledger_row(
        workspace_path=workspace_path,
        run_id=run_id,
        baseline=baseline,
        submission_path=submission_path,
        manifest_path=manifest_path,
        research_brief=research_brief,
    )
    ledger_result = _append_ledger_row(workspace_path / "results.tsv", ledger_row)

    return {
        "kind": "kaggle_baseline_run",
        "competition": {
            "slug": workspace_slug,
            "url": f"https://www.kaggle.com/competitions/{workspace_slug}"
            if workspace_slug
            else None,
        },
        "workspace": {
            "path": str(workspace_path),
            "data_dir": _relative_to(workspace_path, data_path),
        },
        "run": {
            "name": run_id,
            "strategy": strategy,
            "model_family": "constant_baseline",
            "created_at_utc": manifest["created_at_utc"],
        },
        "prediction": {
            "output_columns": baseline["output_columns"],
            "fill_values": baseline["fill_values"],
            "source": baseline["source"],
            "row_count": len(baseline["rows"]),
        },
        "artifacts": {
            "submission_path": _relative_to(workspace_path, submission_path),
            "manifest_path": _relative_to(workspace_path, manifest_path),
            "ledger_path": "results.tsv",
        },
        "validation": validation,
        "ledger": ledger_result,
        "submission_policy": _submission_policy(),
        "next_actions": [
            {
                "priority": 1,
                "action": "inspect_baseline_submission",
                "command": (
                    "labmate kaggle validate-submission "
                    f"{_relative_to(workspace_path, submission_path)} --workspace {workspace_path}"
                ),
                "purpose": "Confirm row count, columns, and IDs before any submit attempt.",
            },
            {
                "priority": 2,
                "action": "implement_model_baseline",
                "purpose": (
                    "Replace the constant baseline with a simple validated model and compare "
                    "against this logged floor."
                ),
            },
            {
                "priority": 99,
                "action": "submit",
                "approval": "required",
                "purpose": "Submit only after explicit user approval of file and message.",
            },
        ],
        "warnings": list(validation.get("warnings", [])),
    }


def validate_kaggle_submission(
    submission: str | Path,
    *,
    workspace: str | Path,
    data_dir: str | Path = "data",
) -> dict[str, Any]:
    """Validate a candidate submission against the local sample submission."""

    workspace_path = _resolve_workspace(workspace)
    data_path = _resolve_child_path(workspace_path, data_dir)
    submission_path = _resolve_submission_path(workspace_path, submission)
    if not submission_path.is_file():
        raise KaggleWorkflowError(
            "submission file does not exist",
            code="submission_not_found",
            exit_code=ExitCode.TOOL_ERROR,
            details={"submission": str(submission)},
        )

    inspection = inspect_local_dataset(data_path, sample_size=1, max_profile_rows=1_000)
    if inspection.get("kind") != "local_dataset_directory":
        raise KaggleWorkflowError(
            "submission validation requires extracted top-level data files",
            code="unsupported_dataset_layout",
            exit_code=ExitCode.TOOL_ERROR,
            details={"data_dir": _relative_to(workspace_path, data_path)},
        )

    relations = inspection.get("relations", {})
    sample_name = relations.get("sample_submission_file")
    if not isinstance(sample_name, str):
        raise KaggleWorkflowError(
            "sample submission file was not detected",
            code="sample_submission_not_found",
            exit_code=ExitCode.TOOL_ERROR,
            details={"data_dir": _relative_to(workspace_path, data_path)},
        )

    sample_path = data_path / sample_name
    sample_rows, sample_columns = _read_tabular_rows(sample_path)
    candidate_rows, candidate_columns = _read_tabular_rows(submission_path)
    common_id_columns = relations.get("sample_submission_alignment", {}).get(
        "common_id_columns",
        [],
    )
    id_columns = [str(column) for column in common_id_columns] if common_id_columns else []
    errors: list[str] = []
    warnings: list[str] = []
    columns_match = sample_columns == candidate_columns
    row_counts_match = len(sample_rows) == len(candidate_rows)
    ids_match = _ids_match(sample_rows, candidate_rows, id_columns)

    if not columns_match:
        errors.append("Submission columns do not exactly match sample submission columns.")
    if not row_counts_match:
        errors.append("Submission row count does not match sample submission row count.")
    if id_columns and not ids_match:
        errors.append("Submission ID columns do not match sample submission order.")
    if not id_columns:
        warnings.append(
            "No shared ID columns were detected; validation used columns and rows only."
        )

    return {
        "kind": "kaggle_submission_validation",
        "status": "ok" if not errors else "failed",
        "workspace": str(workspace_path),
        "submission_path": _relative_to(workspace_path, submission_path),
        "sample_submission_path": _relative_to(workspace_path, sample_path),
        "schema": {
            "expected_columns": sample_columns,
            "actual_columns": candidate_columns,
            "columns_match": columns_match,
        },
        "rows": {
            "expected_count": len(sample_rows),
            "actual_count": len(candidate_rows),
            "row_counts_match": row_counts_match,
        },
        "id_alignment": {
            "id_columns": id_columns,
            "matches_sample": ids_match,
        },
        "errors": errors,
        "warnings": warnings,
        "submission_policy": _submission_policy(),
    }


def normalize_competition_slug(value: str) -> str:
    """Normalize a Kaggle competition URL or slug into a competition slug."""

    raw = value.strip()
    if not raw:
        raise KaggleWorkflowError(
            "competition must be a Kaggle competition URL or slug",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
        )

    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc:
        host = parsed.netloc.lower()
        if host not in KAGGLE_HOSTS:
            raise KaggleWorkflowError(
                f"Unsupported Kaggle host: {parsed.netloc}",
                code="invalid_competition",
                exit_code=ExitCode.USAGE_ERROR,
                details={"competition": value},
            )
        parts = [part for part in parsed.path.split("/") if part]
        slug = _slug_from_url_parts(parts)
    else:
        slug = raw.removesuffix("/").split("/")[-1]

    slug = slug.strip()
    if not SLUG_PATTERN.match(slug):
        raise KaggleWorkflowError(
            f"Invalid Kaggle competition slug: {slug!r}",
            code="invalid_competition",
            exit_code=ExitCode.USAGE_ERROR,
            details={"competition": value, "slug": slug},
        )
    return slug


def _resolve_workspace(workspace: str | Path) -> Path:
    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        raise KaggleWorkflowError(
            "workspace does not exist or is not a directory",
            code="workspace_not_found",
            exit_code=ExitCode.TOOL_ERROR,
            details={"workspace": str(workspace_path)},
        )
    return workspace_path


def _competition_slug_from_workspace(workspace_path: Path) -> str | None:
    metadata_path = workspace_path / ".labmate" / "competition.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    competition = metadata.get("competition")
    if not isinstance(competition, dict):
        return None
    slug = competition.get("slug")
    return str(slug) if slug else None


def _run_id(run_name: str | None) -> str:
    raw_name = run_name or f"constant-baseline-{_timestamp_for_path()}"
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_name.strip()).strip("-._")
    if not normalized:
        raise KaggleWorkflowError(
            "run_name must contain at least one path-safe character",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
            details={"run_name": run_name or ""},
        )
    return normalized


def _timestamp_for_path() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _slug_from_url_parts(parts: list[str]) -> str:
    if not parts:
        raise KaggleWorkflowError(
            "Kaggle URL does not contain a competition slug",
            code="invalid_competition",
            exit_code=ExitCode.USAGE_ERROR,
        )
    for marker in ("competitions", "c"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return parts[-1]


def _baseline_predictions(
    *,
    workspace_path: Path,
    data_path: Path,
    inspection: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    relations = inspection.get("relations", {})
    sample_name = relations.get("sample_submission_file")
    train_name = relations.get("train_file")
    alignment = relations.get("sample_submission_alignment")
    if not isinstance(sample_name, str):
        raise KaggleWorkflowError(
            "sample submission file was not detected",
            code="sample_submission_not_found",
            exit_code=ExitCode.TOOL_ERROR,
            details={"data_dir": _relative_to(workspace_path, data_path)},
        )
    if not isinstance(alignment, dict):
        raise KaggleWorkflowError(
            "sample submission alignment could not be inferred",
            code="sample_submission_not_found",
            exit_code=ExitCode.TOOL_ERROR,
            details={"sample_submission_file": sample_name},
        )

    sample_path = data_path / sample_name
    sample_rows, fieldnames = _read_tabular_rows(sample_path)
    output_columns = [
        str(column) for column in alignment.get("submission_output_columns", []) if column
    ]
    if not output_columns:
        raise KaggleWorkflowError(
            "sample submission has no prediction output columns",
            code="invalid_submission_format",
            exit_code=ExitCode.TOOL_ERROR,
            details={"sample_submission_file": sample_name},
        )

    train_rows: list[dict[str, str]] = []
    if isinstance(train_name, str):
        train_rows, _train_columns = _read_tabular_rows(data_path / train_name)

    fill_values: dict[str, str] = {}
    fill_sources: dict[str, str] = {}
    effective_strategies: dict[str, str] = {}
    for column in output_columns:
        fill_value, source, effective_strategy = _prediction_fill_value(
            column,
            strategy=strategy,
            train_rows=train_rows,
            sample_rows=sample_rows,
        )
        fill_values[column] = fill_value
        fill_sources[column] = source
        effective_strategies[column] = effective_strategy

    candidate_rows = []
    for sample_row in sample_rows:
        row = dict(sample_row)
        for column in output_columns:
            row[column] = fill_values[column]
        candidate_rows.append(row)

    return {
        "rows": candidate_rows,
        "fieldnames": fieldnames,
        "output_columns": output_columns,
        "fill_values": fill_values,
        "source": fill_sources,
        "effective_strategies": effective_strategies,
        "sample_submission_file": sample_name,
        "train_file": train_name,
    }


def _prediction_fill_value(
    column: str,
    *,
    strategy: str,
    train_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> tuple[str, str, str]:
    train_values = [_clean_value(row.get(column, "")) for row in train_rows]
    train_values = [value for value in train_values if value != ""]
    sample_values = [_clean_value(row.get(column, "")) for row in sample_rows]
    sample_values = [value for value in sample_values if value != ""]

    if strategy == "zero":
        return "0", "constant_zero", "zero"
    if strategy == "sample_default" or not train_values:
        return _mode_or_default(sample_values, default="0"), "sample_submission", "sample_default"
    if strategy == "target_mode":
        return _mode_or_default(train_values, default="0"), "train_target_mode", "target_mode"
    if strategy == "target_mean":
        return _mean_or_mode(train_values), "train_target_mean", "target_mean"

    if _looks_regression_like(train_values):
        return _mean_or_mode(train_values), "train_target_mean", "target_mean"
    return _mode_or_default(train_values, default="0"), "train_target_mode", "target_mode"


def _clean_value(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _mode_or_default(values: list[str], *, default: str) -> str:
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _mean_or_mode(values: list[str]) -> str:
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except ValueError:
            return _mode_or_default(values, default="0")
    if not numeric_values:
        return "0"
    mean = sum(numeric_values) / len(numeric_values)
    if mean.is_integer():
        return str(int(mean))
    return f"{mean:.12g}"


def _looks_regression_like(values: list[str]) -> bool:
    if len(set(values)) <= 20:
        return False
    try:
        for value in values:
            float(value)
    except ValueError:
        return False
    return True


def _read_tabular_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise KaggleWorkflowError(
            "expected tabular file does not exist",
            code="dataset_file_not_found",
            exit_code=ExitCode.TOOL_ERROR,
            details={"path": str(path)},
        )
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise KaggleWorkflowError(
                "tabular file has no header row",
                code="invalid_dataset_file",
                exit_code=ExitCode.TOOL_ERROR,
                details={"path": str(path)},
            )
        rows = [
            {name: "" if row.get(name) is None else str(row.get(name)) for name in fieldnames}
            for row in reader
        ]
    return rows, fieldnames


def _write_submission(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _baseline_manifest(
    *,
    workspace_path: Path,
    competition: str | None,
    run_id: str,
    strategy: str,
    baseline: dict[str, Any],
    submission_path: Path,
    validation: dict[str, Any],
    research_brief: Any,
) -> dict[str, Any]:
    modeling_plan = (
        research_brief.get("modeling_plan", {}) if isinstance(research_brief, dict) else {}
    )
    return {
        "schema_version": "labmate.kaggle.baseline.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "competition": competition,
        "run": {
            "name": run_id,
            "strategy": strategy,
            "model_family": "constant_baseline",
        },
        "inputs": {
            "sample_submission_file": baseline["sample_submission_file"],
            "train_file": baseline["train_file"],
            "output_columns": baseline["output_columns"],
        },
        "prediction": {
            "fill_values": baseline["fill_values"],
            "source": baseline["source"],
            "effective_strategies": baseline["effective_strategies"],
        },
        "artifacts": {
            "submission_path": _relative_to(workspace_path, submission_path),
        },
        "validation": validation,
        "modeling_context": {
            "suggested_metric": modeling_plan.get("suggested_metric"),
            "validation_strategy": modeling_plan.get("validation_strategy"),
        },
    }


def _baseline_ledger_row(
    *,
    workspace_path: Path,
    run_id: str,
    baseline: dict[str, Any],
    submission_path: Path,
    manifest_path: Path,
    research_brief: Any,
) -> dict[str, str]:
    tracking_plan = (
        research_brief.get("experiment_tracking_plan", {})
        if isinstance(research_brief, dict)
        else {}
    )
    metric = str(tracking_plan.get("primary_metric") or "task_metric_from_rules")
    score_direction = str(tracking_plan.get("score_direction") or "unknown")
    validation_strategy = str(tracking_plan.get("validation_strategy") or "not_validated")
    artifacts = (
        f"{_relative_to(workspace_path, submission_path)}; "
        f"{_relative_to(workspace_path, manifest_path)}"
    )
    fill_summary = ", ".join(
        f"{column}={value}" for column, value in baseline["fill_values"].items()
    )
    return {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "commit": _git_commit(workspace_path),
        "experiment": run_id,
        "model_family": "constant_baseline",
        "features": "none",
        "validation_strategy": validation_strategy,
        "metric": metric,
        "score": "",
        "score_direction": score_direction,
        "status": "submission_ready",
        "artifacts": artifacts,
        "notes": f"Constant baseline; {fill_summary}; not submitted.",
    }


def _append_ledger_row(ledger_path: Path, row: dict[str, str]) -> dict[str, Any]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if not ledger_path.exists():
        ledger_path.write_text("\t".join(RECOMMENDED_LEDGER_COLUMNS) + "\n", encoding="utf-8")

    with ledger_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            columns = next(reader)
        except StopIteration:
            columns = list(RECOMMENDED_LEDGER_COLUMNS)

    missing_columns = [column for column in RECOMMENDED_LEDGER_COLUMNS if column not in columns]
    if missing_columns:
        raise KaggleWorkflowError(
            "results.tsv is missing recommended Labmate columns",
            code="invalid_experiment_ledger",
            exit_code=ExitCode.TOOL_ERROR,
            details={"missing_columns": missing_columns},
        )

    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writerow(row)

    return {
        "appended": True,
        "ledger_path": ledger_path.name,
        "row": {column: row.get(column, "") for column in columns},
    }


def _git_commit(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"


def _resolve_submission_path(workspace_path: Path, submission: str | Path) -> Path:
    submission_path = Path(submission)
    resolved = (
        submission_path if submission_path.is_absolute() else workspace_path / submission_path
    )
    resolved = resolved.expanduser().resolve()
    try:
        resolved.relative_to(workspace_path)
    except ValueError as exc:
        raise KaggleWorkflowError(
            "submission path must stay inside the Kaggle workspace",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
            details={"workspace": str(workspace_path), "submission": str(submission)},
        ) from exc
    return resolved


def _ids_match(
    sample_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    id_columns: list[str],
) -> bool | None:
    if not id_columns:
        return None
    if len(sample_rows) != len(candidate_rows):
        return False
    for sample_row, candidate_row in zip(sample_rows, candidate_rows, strict=True):
        for column in id_columns:
            if sample_row.get(column, "") != candidate_row.get(column, ""):
                return False
    return True


def _submission_policy() -> dict[str, Any]:
    return {
        "status": "manual_approval_required",
        "message": (
            "This artifact is not submitted. Ask for explicit user approval before "
            "running Kaggle submission commands or MCP submission tools."
        ),
    }


def _resolve_child_path(workspace_path: Path, child: str | Path) -> Path:
    child_path = Path(child)
    resolved = child_path if child_path.is_absolute() else workspace_path / child_path
    resolved = resolved.expanduser().resolve()
    try:
        resolved.relative_to(workspace_path)
    except ValueError as exc:
        raise KaggleWorkflowError(
            "data_dir must stay inside the Kaggle workspace",
            code="invalid_arguments",
            exit_code=ExitCode.USAGE_ERROR,
            details={"workspace": str(workspace_path), "data_dir": str(child)},
        ) from exc
    return resolved


def _ensure_directories(workspace_path: Path, data_path: Path) -> list[dict[str, Any]]:
    actions = []
    for relative in DEFAULT_WORKSPACE_DIRS:
        directory = data_path if relative == "data" else workspace_path / relative
        if directory.exists() and not directory.is_dir():
            raise KaggleWorkflowError(
                "workspace directory path collides with a file",
                code="invalid_arguments",
                exit_code=ExitCode.USAGE_ERROR,
                details={"path": str(directory)},
            )
        existed = directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        actions.append(
            {
                "path": _relative_to(workspace_path, directory),
                "action": "exists" if existed else "created",
            }
        )
    return actions


def _ensure_text_file(
    path: Path,
    content: str,
    *,
    root: Path,
    update_generated: bool,
    always_update: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = _relative_to(root, path)
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if always_update or (update_generated and existing.startswith(GENERATED_MARKER)):
            path.write_text(content, encoding="utf-8")
            return {
                "path": relative,
                "action": "updated",
                "reason": "generated file refreshed",
            }
        return {"path": relative, "action": "skipped", "reason": "exists; kept local file"}

    path.write_text(content, encoding="utf-8")
    return {"path": relative, "action": "written", "reason": "missing"}


def _download_competition(
    slug: str,
    *,
    data_path: Path,
    download: bool,
    force_download: bool,
) -> dict[str, Any]:
    command = ["kaggle", "competitions", "download", "-c", slug, "-p", str(data_path)]
    if force_download:
        command.append("--force")

    if not download:
        return {
            "requested": False,
            "status": "skipped",
            "command": command,
            "reason": "download disabled by caller",
        }

    kaggle_executable = shutil.which("kaggle")
    if kaggle_executable is None:
        return {
            "requested": True,
            "status": "unavailable",
            "command": command,
            "reason": "kaggle CLI was not found on PATH",
            "recommended_fix": (
                "Install/configure the Kaggle CLI or let a host agent use an available "
                "Kaggle MCP download_competition tool."
            ),
        }

    before = _relative_file_list(data_path)
    try:
        completed = subprocess.run(
            [kaggle_executable, "competitions", "download", "-c", slug, "-p", str(data_path)]
            + (["--force"] if force_download else []),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "requested": True,
            "status": "failed",
            "command": command,
            "reason": "kaggle CLI download timed out",
            "stdout": _excerpt(exc.stdout or ""),
            "stderr": _excerpt(exc.stderr or ""),
            "files_before": before,
            "files_after": _relative_file_list(data_path),
        }
    except OSError as exc:
        return {
            "requested": True,
            "status": "failed",
            "command": command,
            "reason": str(exc),
            "files_before": before,
            "files_after": _relative_file_list(data_path),
        }
    after = _relative_file_list(data_path)
    return {
        "requested": True,
        "status": "ok" if completed.returncode == 0 else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "reason": None if completed.returncode == 0 else "kaggle CLI exited with a non-zero status",
        "stdout": _excerpt(completed.stdout),
        "stderr": _excerpt(completed.stderr),
        "files_before": before,
        "files_after": after,
    }


def _inspect_available_data(
    *,
    workspace_path: Path,
    data_path: Path,
    sample_size: int,
    max_profile_rows: int,
) -> dict[str, Any]:
    inspected_path = _dataset_candidate(data_path)
    if inspected_path is None:
        return {
            "status": "missing",
            "path": _relative_to(workspace_path, data_path),
            "inspected_path": None,
            "inspection": None,
            "research_brief": None,
            "warnings": [
                "No local CSV/TSV files or zip archives were found yet. Download the "
                "competition data before modeling."
            ],
        }

    try:
        inspection = inspect_local_dataset(
            inspected_path,
            sample_size=sample_size,
            max_profile_rows=max_profile_rows,
        )
        research_brief = build_research_brief(
            inspected_path,
            sample_size=sample_size,
            max_profile_rows=max_profile_rows,
            max_benchmarks=3,
        )
    except DatasetInspectionError as exc:
        return {
            "status": "inspection_failed",
            "path": _relative_to(workspace_path, data_path),
            "inspected_path": _relative_to(workspace_path, inspected_path),
            "inspection": None,
            "research_brief": None,
            "warnings": [str(exc)],
        }

    return {
        "status": "inspected",
        "path": _relative_to(workspace_path, data_path),
        "inspected_path": _relative_to(workspace_path, inspected_path),
        "inspection": inspection,
        "research_brief": research_brief,
        "warnings": list(research_brief.get("warnings", [])),
    }


def _dataset_candidate(data_path: Path) -> Path | None:
    if not data_path.exists():
        return None
    if any(path.is_file() and _is_tabular_name(path.name) for path in data_path.iterdir()):
        return data_path
    archives = sorted(
        path for path in data_path.iterdir() if path.is_file() and path.suffix.lower() == ".zip"
    )
    return archives[0] if archives else None


def _is_tabular_name(name: str) -> bool:
    suffixes = [suffix.lower() for suffix in Path(name).suffixes]
    if suffixes[-2:] in ([".csv", ".gz"], [".tsv", ".gz"]):
        return True
    return Path(name).suffix.lower() in {".csv", ".tsv"}


def _metadata(
    slug: str,
    competition_input: str,
    workspace_path: Path,
    data_path: Path,
    data_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "labmate.kaggle.v1",
        "updated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "competition": {
            "input": competition_input,
            "slug": slug,
            "url": f"https://www.kaggle.com/competitions/{slug}",
        },
        "workspace": str(workspace_path),
        "data_dir": _relative_to(workspace_path, data_path),
        "data_status": data_result["status"],
    }


def _brief_text(slug: str, data_result: dict[str, Any]) -> str:
    lines = [
        GENERATED_MARKER,
        f"# {slug} Competition Brief",
        "",
        f"- Competition: https://www.kaggle.com/competitions/{slug}",
        f"- Data status: {data_result['status']}",
        "- Results ledger: `results.tsv`",
        "- Submission policy: ask for explicit approval before submitting.",
        "",
    ]
    research_brief = data_result.get("research_brief")
    if isinstance(research_brief, dict):
        modeling_plan = research_brief.get("modeling_plan", {})
        inferred_task = research_brief.get("inferred_task", {})
        target_columns = ", ".join(inferred_task.get("target_columns", [])) or "unknown"
        validation_strategy = modeling_plan.get("validation_strategy", {}).get(
            "name",
            "unknown",
        )
        suggested_metric = modeling_plan.get("suggested_metric") or "verify from competition rules"
        lines.extend(
            [
                "## Inferred Task",
                "",
                f"- Type: {inferred_task.get('task_type', 'unknown')}",
                f"- Confidence: {inferred_task.get('confidence', 'unknown')}",
                f"- Target columns: {target_columns}",
                "",
                "## Baseline Plan",
                "",
            ]
        )
        for experiment in modeling_plan.get("baseline_experiments", []):
            lines.append(
                f"- `{experiment.get('name')}`: {experiment.get('purpose', 'baseline check')}"
            )
        lines.extend(
            [
                "",
                "## Validation",
                "",
                f"- Strategy: {validation_strategy}",
                f"- Metric: {suggested_metric}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Next Step",
                "",
                "Download or place Kaggle data under `data/`, then rerun:",
                "",
                f"```bash\nlabmate kaggle start {slug} --workspace . --no-download\n```",
                "",
            ]
        )

    warnings = data_result.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:10])
        lines.append("")
    return "\n".join(lines)


def _program_text(slug: str) -> str:
    ledger_header = "\t".join(RECOMMENDED_LEDGER_COLUMNS)
    return f"""# Program

## Goal

Make measurable progress on the Kaggle competition `{slug}`.

Competition URL: https://www.kaggle.com/competitions/{slug}

## Scope

Allowed changes:

- inspect local data and competition metadata
- create reproducible baselines and validation scripts
- write submissions under `submissions/`
- log every experiment in `results.tsv`

Forbidden changes:

- do not submit to Kaggle without explicit user approval
- do not tune only against public leaderboard feedback
- do not commit raw competition data, run artifacts, or secrets

## Evidence Required

Run these before editing model code:

- `labmate kaggle start {slug} --workspace . --no-download`
- `labmate dataset-inspect data/`
- `labmate experiment-summary results.tsv`
- `labmate research-brief data/`
- `labmate benchmark-lookup "<task metric kaggle>"`

If Kaggle MCP tools are available, use them for competition details, downloads,
submissions, and submission polling. Keep submit actions separate from research
and ask the user before each submission.

## Metric

Primary metric: verify from Kaggle evaluation page.

Validation command: define before model tuning.

## Output

Write results to `results.tsv` with columns:

```text
{ledger_header}
```

## Keep Criteria

Keep an experiment only if it records the command/config, commit, validation
score, artifact path, and interpretation.

## Stop Criteria

Stop when Kaggle credentials, rules acceptance, data access, compute, or
submission quota blocks progress, or when the user asks to pause.
"""


def _gitignore_text() -> str:
    return """# Labmate Kaggle workspace
data/
runs/
submissions/
*.zip
__pycache__/
.DS_Store
"""


def _kaggle_access_summary() -> dict[str, Any]:
    return {
        "kaggle_cli_available": shutil.which("kaggle") is not None,
        "secret_policy": "Labmate never reads model-provider auth files or stores Kaggle secrets.",
        "mcp_handoff": [
            {
                "tool": "kaggle.get_competition_details",
                "purpose": "Fetch current overview, metric, rules, and deadlines when available.",
            },
            {
                "tool": "kaggle.download_competition",
                "purpose": "Download competition data through the host MCP client when configured.",
            },
            {
                "tool": "kaggle.list_submissions",
                "purpose": "Read prior submissions and public scores.",
            },
            {
                "tool": "kaggle.submit_to_competition",
                "purpose": "Submit only after explicit user approval.",
                "approval": "required",
            },
        ],
    }


def _agent_handoff(slug: str, workspace_path: Path, data_result: dict[str, Any]) -> dict[str, Any]:
    workspace = str(workspace_path)
    return {
        "claude_project_command": f"/kagglethis {slug}",
        "claude_mcp_prompt": f"/mcp__labmate__kagglethis {slug}",
        "codex_goal": (
            f"/goal Work on Kaggle competition {slug} in {workspace}. Start with "
            "`labmate kaggle start` and do not submit without explicit approval."
        ),
        "subagent": "kaggle-researcher",
        "data_ready": data_result["status"] == "inspected",
    }


def _next_actions(
    slug: str,
    workspace_path: Path,
    data_path: Path,
    data_result: dict[str, Any],
) -> list[dict[str, Any]]:
    actions = [
        {
            "priority": 1,
            "surface": "kaggle",
            "action": "verify_competition_rules",
            "purpose": "Confirm metric, submission format, rules, and deadline at the source.",
            "url": f"https://www.kaggle.com/competitions/{slug}",
        }
    ]
    if data_result["status"] != "inspected":
        actions.append(
            {
                "priority": 2,
                "surface": "kaggle",
                "action": "download_data",
                "command": f"kaggle competitions download -c {slug} -p {data_path}",
                "mcp_tool": "kaggle.download_competition",
                "purpose": "Place train/test/sample_submission files under data/.",
            }
        )
    else:
        actions.extend(
            [
                {
                    "priority": 2,
                    "surface": "cli",
                    "action": "inspect_data",
                    "command": "labmate dataset-inspect data/",
                    "purpose": "Confirm schema, leakage warnings, and sample submission alignment.",
                },
                {
                    "priority": 3,
                    "surface": "cli",
                    "action": "summarize_experiments",
                    "command": "labmate experiment-summary results.tsv",
                    "purpose": "Avoid repeating failed runs and keep best run visible.",
                },
                {
                    "priority": 4,
                    "surface": "cli",
                    "action": "create_constant_baseline",
                    "command": f"labmate kaggle baseline {workspace_path}",
                    "purpose": (
                        "Create a validated sample-submission-compatible floor before "
                        "modeling or leaderboard submissions."
                    ),
                },
                {
                    "priority": 5,
                    "surface": "agent",
                    "action": "implement_model_baseline",
                    "purpose": (
                        "Replace the constant baseline with a simple model after the "
                        "submission path and ledger are trusted."
                    ),
                },
            ]
        )
    actions.append(
        {
            "priority": 99,
            "surface": "kaggle",
            "action": "submit",
            "purpose": "Submit a generated file only after explicit user approval.",
            "approval": "required",
        }
    )
    return actions


def _workflow_warnings(
    download_result: dict[str, Any],
    data_result: dict[str, Any],
) -> list[str]:
    warnings = []
    if download_result["status"] in {"unavailable", "failed"}:
        warnings.append(str(download_result.get("reason") or "Kaggle download did not complete."))
    warnings.extend(str(warning) for warning in data_result.get("warnings", []))
    return warnings


def _relative_file_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    files = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        files.append(_relative_to(path, candidate))
        if len(files) >= MAX_FILE_LIST:
            files.append("<truncated>")
            break
    return files


def _relative_to(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _excerpt(text: object) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    compact = str(text).strip()
    if len(compact) > MAX_OUTPUT_CHARS:
        return compact[:MAX_OUTPUT_CHARS].rstrip() + "..."
    return compact
