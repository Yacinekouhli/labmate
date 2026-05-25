"""Read-only local dataset inspection tools."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".csv", ".tsv"}
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_MAX_PROFILE_ROWS = 250_000
MAX_TRACKED_UNIQUE_VALUES = 1_000
MAX_TOP_VALUES = 10
MAX_CONTEXT_FILES = 10
MAX_CONTEXT_SNIPPET_CHARS = 1_500

MISSING_MARKERS = {"", "na", "n/a", "nan", "none", "null", "nil", "missing", "?"}
ID_COLUMN_NAMES = {"id", "idx", "index", "rowid", "row_id", "sample_id"}
TARGET_COLUMN_NAMES = {
    "target",
    "label",
    "labels",
    "class",
    "y",
    "outcome",
    "response",
    "survived",
    "saleprice",
    "prediction",
}
SPLIT_HINT_NAMES = {"fold", "split", "train", "test", "valid", "validation", "is_train"}
FUTURE_HINT_NAMES = {"future", "next", "post", "after"}
CONTEXT_FILE_STEMS = {
    "readme",
    "data_description",
    "description",
    "overview",
    "evaluation",
    "rules",
    "competition",
    "dataset_metadata",
    "kaggle",
}
CONTEXT_FILE_SUFFIXES = {".md", ".txt", ".json"}


class DatasetInspectionError(ValueError):
    """Raised when a local dataset cannot be inspected."""


def inspect_local_dataset(
    path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_profile_rows: int = DEFAULT_MAX_PROFILE_ROWS,
) -> dict[str, Any]:
    """Inspect a local CSV/TSV file or a directory containing CSV/TSV files."""

    dataset_path = Path(path)
    if dataset_path.is_dir():
        return inspect_local_directory(
            dataset_path,
            sample_size=sample_size,
            max_profile_rows=max_profile_rows,
        )
    return inspect_tabular_file(
        dataset_path,
        sample_size=sample_size,
        max_profile_rows=max_profile_rows,
    )


def inspect_local_directory(
    path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_profile_rows: int = DEFAULT_MAX_PROFILE_ROWS,
) -> dict[str, Any]:
    """Inspect top-level CSV/TSV files in a local dataset directory."""

    directory = Path(path)
    if not directory.is_dir():
        raise DatasetInspectionError(f"Not a directory: {directory}")

    files = [
        inspect_tabular_file(
            file_path,
            sample_size=sample_size,
            max_profile_rows=max_profile_rows,
        )
        for file_path in sorted(directory.iterdir())
        if file_path.is_file() and _is_supported_tabular_path(file_path)
    ]
    if not files:
        raise DatasetInspectionError(f"No supported CSV/TSV files found in: {directory}")

    relations = _directory_relations(files)

    return {
        "kind": "local_dataset_directory",
        "path": str(directory),
        "files": files,
        "context_files": _directory_context_files(directory),
        "relations": relations,
        "warnings": _directory_warnings(files, relations),
    }


def inspect_tabular_file(
    path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_profile_rows: int = DEFAULT_MAX_PROFILE_ROWS,
) -> dict[str, Any]:
    """Inspect a local CSV or TSV file with deterministic stdlib parsing."""

    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetInspectionError(f"Not a file: {file_path}")
    if not _is_supported_tabular_path(file_path):
        raise DatasetInspectionError(f"Unsupported dataset file type: {file_path.name}")
    if sample_size < 0:
        raise DatasetInspectionError("sample_size must be non-negative")
    if max_profile_rows <= 0:
        raise DatasetInspectionError("max_profile_rows must be positive")

    delimiter = _delimiter_for_path(file_path)
    with _open_tabular_file(file_path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise DatasetInspectionError(f"No header row found in: {file_path}")

        columns = [_new_column_profile(name, position) for position, name in enumerate(fieldnames)]
        sample_rows: list[dict[str, str]] = []
        row_count = 0
        truncated = False

        for row in reader:
            if row_count >= max_profile_rows:
                truncated = True
                break

            normalized_row = {name: row.get(name, "") for name in fieldnames}
            if len(sample_rows) < sample_size:
                sample_rows.append(normalized_row)

            for column in columns:
                _update_column_profile(column, normalized_row.get(column["name"], ""))
            row_count += 1

    finalized_columns = [_finalize_column_profile(column, row_count) for column in columns]
    target_hints = _target_column_hints(finalized_columns, file_path.name)
    leakage_hints = _leakage_risk_hints(finalized_columns)

    return {
        "kind": "tabular_file",
        "path": str(file_path),
        "file_name": file_path.name,
        "format": _tabular_suffix(file_path).removeprefix("."),
        "compression": _compression(file_path),
        "delimiter": delimiter,
        "row_count": None if truncated else row_count,
        "row_count_status": "bounded" if truncated else "exact",
        "profiled_row_count": row_count,
        "max_profile_rows": max_profile_rows,
        "columns": finalized_columns,
        "sample_rows": sample_rows,
        "target_column_hints": target_hints,
        "leakage_risk_hints": leakage_hints,
        "warnings": _file_warnings(finalized_columns, target_hints, truncated),
    }


def _delimiter_for_path(path: Path) -> str:
    if _tabular_suffix(path) == ".tsv":
        return "\t"
    return ","


def _open_tabular_file(path: Path):
    if _compression(path) == "gzip":
        return gzip.open(path, mode="rt", newline="", encoding="utf-8-sig")
    return path.open(newline="", encoding="utf-8-sig")


def _is_supported_tabular_path(path: Path) -> bool:
    return _tabular_suffix(path) in SUPPORTED_SUFFIXES


def _tabular_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] in ([".csv", ".gz"], [".tsv", ".gz"]):
        return suffixes[-2]
    return path.suffix.lower()


def _compression(path: Path) -> str | None:
    return "gzip" if path.suffix.lower() == ".gz" else None


def _new_column_profile(name: str, position: int) -> dict[str, Any]:
    return {
        "name": name,
        "position": position,
        "missing_count": 0,
        "non_missing_count": 0,
        "type_counts": Counter(),
        "unique_values": set(),
        "value_counts": Counter(),
        "unique_values_truncated": False,
    }


def _update_column_profile(column: dict[str, Any], raw_value: str | None) -> None:
    value = "" if raw_value is None else raw_value.strip()
    if _is_missing(value):
        column["missing_count"] += 1
        return

    column["non_missing_count"] += 1
    column["type_counts"][_infer_scalar_type(value)] += 1
    unique_values = column["unique_values"]
    if value in unique_values or len(unique_values) < MAX_TRACKED_UNIQUE_VALUES:
        unique_values.add(value)
        column["value_counts"][value] += 1
    else:
        column["unique_values_truncated"] = True


def _finalize_column_profile(column: dict[str, Any], row_count: int) -> dict[str, Any]:
    missing_count = column["missing_count"]
    non_missing_count = column["non_missing_count"]
    type_counts = column["type_counts"]
    unique_values = column["unique_values"]
    missing_rate = missing_count / row_count if row_count else 0.0

    return {
        "name": column["name"],
        "position": column["position"],
        "inferred_type": _final_type(type_counts),
        "missing_count": missing_count,
        "missing_rate": round(missing_rate, 6),
        "non_missing_count": non_missing_count,
        "unique_values_profiled": len(unique_values),
        "unique_values_truncated": column["unique_values_truncated"],
        "top_values": _top_values(column["value_counts"], row_count),
        "role_hints": _role_hints(column["name"], row_count, len(unique_values)),
    }


def _top_values(value_counts: Counter[str], row_count: int) -> list[dict[str, Any]]:
    if not value_counts:
        return []
    return [
        {
            "value": value,
            "count": count,
            "rate": round(count / row_count, 6) if row_count else 0.0,
        }
        for value, count in value_counts.most_common(MAX_TOP_VALUES)
    ]


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_MARKERS


def _infer_scalar_type(value: str) -> str:
    lower_value = value.lower()
    if lower_value in {"true", "false", "yes", "no"}:
        return "boolean"
    try:
        int(value)
    except ValueError:
        pass
    else:
        return "integer"
    try:
        float(value)
    except ValueError:
        pass
    else:
        return "number"
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "string"
    return "datetime"


def _final_type(type_counts: Counter[str]) -> str:
    if not type_counts:
        return "unknown"
    observed = set(type_counts)
    if observed == {"integer"}:
        return "integer"
    if observed <= {"integer", "number"}:
        return "number"
    if observed == {"boolean"}:
        return "boolean"
    if observed == {"datetime"}:
        return "datetime"
    return "string"


def _role_hints(name: str, row_count: int, unique_count: int) -> list[str]:
    normalized = _normalize_name(name)
    hints: list[str] = []
    if normalized in ID_COLUMN_NAMES or normalized.endswith("_id"):
        hints.append("id")
    if row_count > 0 and unique_count == row_count:
        hints.append("unique_per_row")
    if normalized in TARGET_COLUMN_NAMES:
        hints.append("target_name")
    if normalized in SPLIT_HINT_NAMES:
        hints.append("split_indicator")
    if any(token in normalized for token in FUTURE_HINT_NAMES):
        hints.append("future_information")
    return hints


def _target_column_hints(
    columns: Iterable[dict[str, Any]],
    file_name: str,
) -> list[dict[str, Any]]:
    file_role = _file_role(file_name)
    hints = []

    for column in columns:
        name = column["name"]
        normalized = _normalize_name(name)
        role_hints = set(column["role_hints"])
        reasons: list[str] = []
        score = 0.0

        if "target_name" in role_hints:
            score += 0.8
            reasons.append("column name looks like a supervised-learning target")
        if file_role == "sample_submission" and "id" not in role_hints:
            score += 0.95
            reasons.append("non-id column in a sample submission file")
        if normalized.startswith("is_") or normalized.startswith("has_"):
            score += 0.35
            reasons.append("binary label-style column name")

        if score:
            hints.append(
                {
                    "column": name,
                    "score": round(min(score, 1.0), 3),
                    "reasons": reasons,
                }
            )

    return sorted(hints, key=lambda hint: (-hint["score"], hint["column"]))


def _leakage_risk_hints(columns: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    hints = []
    for column in columns:
        name = column["name"]
        normalized = _normalize_name(name)
        role_hints = set(column["role_hints"])

        if "split_indicator" in role_hints:
            hints.append(
                {
                    "column": name,
                    "risk": "split_indicator",
                    "reason": (
                        "split/fold columns are useful for validation but can leak evaluation setup"
                    ),
                }
            )
        if "future_information" in role_hints:
            hints.append(
                {
                    "column": name,
                    "risk": "future_information",
                    "reason": (
                        "column name suggests information that may not be available "
                        "at prediction time"
                    ),
                }
            )
        if "target" in normalized and normalized not in TARGET_COLUMN_NAMES:
            hints.append(
                {
                    "column": name,
                    "risk": "target_encoded_feature",
                    "reason": "column name suggests target encoding or target-derived information",
                }
            )
        if "unique_per_row" in role_hints and "id" not in role_hints:
            hints.append(
                {
                    "column": name,
                    "risk": "high_cardinality_identifier",
                    "reason": (
                        "column is unique in the profiled rows and may behave like an identifier"
                    ),
                }
            )

    return hints


def _file_warnings(
    columns: list[dict[str, Any]],
    target_hints: list[dict[str, Any]],
    truncated: bool,
) -> list[str]:
    warnings = []
    if not target_hints:
        warnings.append("No obvious target column detected from column names.")
    if any(column["inferred_type"] == "unknown" for column in columns):
        warnings.append("At least one column has no non-missing values in the profiled rows.")
    if truncated:
        warnings.append("Profile row limit reached; row count and profile statistics are bounded.")
    return warnings


def _directory_relations(files: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for file_info in files:
        role = _file_role(file_info["file_name"])
        if role and role not in by_role:
            by_role[role] = file_info

    sample_submission = by_role.get("sample_submission")
    test = by_role.get("test")
    train = by_role.get("train")
    likely_target_columns = _likely_directory_targets(train, sample_submission)

    return {
        "train_file": train["file_name"] if train else None,
        "test_file": test["file_name"] if test else None,
        "sample_submission_file": sample_submission["file_name"] if sample_submission else None,
        "sample_submission_alignment": _sample_submission_alignment(sample_submission, test),
        "train_test_schema_alignment": _train_test_schema_alignment(
            train,
            test,
            likely_target_columns,
        ),
        "likely_target_columns": likely_target_columns,
    }


def _sample_submission_alignment(
    sample_submission: dict[str, Any] | None,
    test: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not sample_submission or not test:
        return None

    submission_id_columns = _id_column_names(sample_submission)
    test_id_columns = _id_column_names(test)
    common_id_columns = sorted(set(submission_id_columns) & set(test_id_columns))

    row_counts_match = None
    if sample_submission["row_count_status"] == "exact" and test["row_count_status"] == "exact":
        row_counts_match = sample_submission["row_count"] == test["row_count"]

    return {
        "common_id_columns": common_id_columns,
        "row_counts_match": row_counts_match,
        "submission_output_columns": [
            column["name"]
            for column in sample_submission["columns"]
            if column["name"] not in submission_id_columns
        ],
    }


def _likely_directory_targets(
    train: dict[str, Any] | None,
    sample_submission: dict[str, Any] | None,
) -> list[str]:
    candidates: list[str] = []
    if sample_submission:
        for hint in sample_submission["target_column_hints"]:
            candidates.append(hint["column"])
    if train:
        for hint in train["target_column_hints"]:
            candidates.append(hint["column"])
    return list(dict.fromkeys(candidates))


def _train_test_schema_alignment(
    train: dict[str, Any] | None,
    test: dict[str, Any] | None,
    likely_target_columns: list[str],
) -> dict[str, Any] | None:
    if not train or not test:
        return None

    train_columns = _column_names(train)
    test_columns = _column_names(test)
    test_column_set = set(test_columns)
    train_id_columns = _id_column_names(train)
    id_column_set = set(train_id_columns) | set(_id_column_names(test))
    target_column_set = set(likely_target_columns)

    common_columns = [column for column in train_columns if column in test_column_set]
    train_only_columns = [column for column in train_columns if column not in test_column_set]
    test_only_columns = [column for column in test_columns if column not in set(train_columns)]
    common_feature_columns = [
        column
        for column in common_columns
        if column not in id_column_set and column not in target_column_set
    ]

    return {
        "common_columns": common_columns,
        "common_feature_columns": common_feature_columns,
        "id_columns": [column for column in train_id_columns if column in test_column_set],
        "train_only_columns": train_only_columns,
        "test_only_columns": test_only_columns,
        "target_columns_absent_from_test": [
            column
            for column in likely_target_columns
            if column in train_columns and column not in test_column_set
        ],
        "target_columns_present_in_test": [
            column
            for column in likely_target_columns
            if column in train_columns and column in test_column_set
        ],
    }


def _directory_warnings(files: list[dict[str, Any]], relations: dict[str, Any]) -> list[str]:
    warnings = []
    roles = {_file_role(file_info["file_name"]) for file_info in files}
    if "train" not in roles:
        warnings.append("No train-like CSV/TSV file detected from file names.")
    if "test" not in roles:
        warnings.append("No test-like CSV/TSV file detected from file names.")
    if "sample_submission" not in roles:
        warnings.append("No sample-submission-like CSV/TSV file detected from file names.")

    for file_info in files:
        role = _file_role(file_info["file_name"])
        if role == "test" and file_info["target_column_hints"]:
            warnings.append(
                f"{file_info['file_name']} has target-like columns; "
                "verify no labels leaked into test."
            )

    schema_alignment = relations.get("train_test_schema_alignment")
    if schema_alignment:
        if not schema_alignment["common_feature_columns"]:
            warnings.append("No non-id feature columns are shared between train and test files.")
        if schema_alignment["target_columns_present_in_test"]:
            warnings.append(
                "Likely target columns are present in both train and test files; "
                "verify this is not label leakage."
            )
    return warnings


def _directory_context_files(directory: Path) -> list[dict[str, Any]]:
    context_files = []
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file() or not _is_context_file(file_path):
            continue
        context_files.append(_context_file_summary(file_path))
        if len(context_files) >= MAX_CONTEXT_FILES:
            break
    return context_files


def _is_context_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in CONTEXT_FILE_SUFFIXES:
        return False
    stem = _normalize_name(path.stem)
    return stem in CONTEXT_FILE_STEMS or any(token in stem for token in CONTEXT_FILE_STEMS)


def _context_file_summary(path: Path) -> dict[str, Any]:
    snippet = _read_context_snippet(path)
    summary: dict[str, Any] = {
        "file_name": path.name,
        "path": str(path),
        "kind": _context_file_kind(path),
        "size_bytes": path.stat().st_size,
        "snippet": snippet,
    }
    if path.suffix.lower() == ".json":
        summary["json_keys"] = _json_keys(snippet)
    return summary


def _read_context_snippet(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as handle:
        text = handle.read(MAX_CONTEXT_SNIPPET_CHARS + 1)
    text = " ".join(text.split())
    if len(text) > MAX_CONTEXT_SNIPPET_CHARS:
        return text[:MAX_CONTEXT_SNIPPET_CHARS].rstrip() + "..."
    return text


def _context_file_kind(path: Path) -> str:
    stem = _normalize_name(path.stem)
    if stem in {"data_description", "description"}:
        return "data_description"
    if stem in {"evaluation", "rules"}:
        return "competition_rules"
    if stem in {"kaggle", "competition", "dataset_metadata"} or path.suffix.lower() == ".json":
        return "metadata"
    return "documentation"


def _json_keys(snippet: str) -> list[str]:
    try:
        value = json.loads(snippet.removesuffix("..."))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)[:25]


def _column_names(file_info: dict[str, Any]) -> list[str]:
    return [column["name"] for column in file_info["columns"]]


def _id_column_names(file_info: dict[str, Any]) -> list[str]:
    return [
        column["name"]
        for column in file_info["columns"]
        if "id" in column["role_hints"] or column["position"] == 0
    ]


def _file_role(file_name: str) -> str | None:
    normalized = _normalize_name(_tabular_stem(file_name))
    if "sample_submission" in normalized or normalized in {"submission", "sample"}:
        return "sample_submission"
    if normalized in {"train", "training"} or normalized.startswith("train_"):
        return "train"
    if normalized in {"test", "testing"} or normalized.startswith("test_"):
        return "test"
    return None


def _tabular_stem(file_name: str) -> str:
    path = Path(file_name)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] in ([".csv", ".gz"], [".tsv", ".gz"]):
        return path.name[: -len("".join(path.suffixes[-2:]))]
    return path.stem


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized
