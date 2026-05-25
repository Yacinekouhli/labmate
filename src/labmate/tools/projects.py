"""Read-only local project discovery for unknown ML repositories."""

from __future__ import annotations

import csv
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_SUFFIXES = {".csv", ".tsv"}
ARCHIVE_SUFFIXES = {".zip"}
NOTEBOOK_SUFFIX = ".ipynb"
SCRIPT_SUFFIXES = {".py", ".R", ".r"}
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
DEPENDENCY_FILES = {
    "environment.yml": "conda_environment",
    "environment.yaml": "conda_environment",
    "pyproject.toml": "python_project",
    "requirements.txt": "python_requirements",
    "setup.py": "python_package",
    "uv.lock": "uv_lock",
}
AGENT_FILES = {
    "AGENTS.md": "agent_instructions",
    "program.md": "labmate_program",
    ".mcp.json": "mcp_config",
}
EXPERIMENT_FILES = {
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
MAX_LEDGER_ROWS_TO_COUNT = 1_000
CONTEXT_STEMS = {
    "data_description",
    "description",
    "evaluation",
    "kaggle",
    "overview",
    "readme",
    "rules",
}


class ProjectScanError(ValueError):
    """Raised when a local project cannot be scanned."""


def scan_local_project(
    path: str | Path,
    *,
    max_depth: int = 4,
    max_entries: int = 500,
) -> dict[str, Any]:
    """Scan a local ML repository for likely Labmate entrypoints."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_entries < 1:
        raise ValueError("max_entries must be positive")

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ProjectScanError(f"Project path does not exist: {root}")
    if not root.is_dir():
        raise ProjectScanError(f"Project path is not a directory: {root}")

    scan_state = _scan_tree(root, max_depth=max_depth, max_entries=max_entries)
    dataset_candidates = _dataset_candidates(root, scan_state["dataset_files"])
    code_entrypoints = _code_entrypoints(root, scan_state["code_files"])
    experiment_files = _experiment_file_summaries(root, scan_state["experiment_files"])

    return {
        "kind": "local_project_scan",
        "path": str(root),
        "scan_limits": {
            "max_depth": max_depth,
            "max_entries": max_entries,
            "visited_entries": scan_state["visited_entries"],
            "truncated": scan_state["truncated"],
        },
        "dataset_candidates": dataset_candidates,
        "code_entrypoints": code_entrypoints,
        "dependency_files": _relative_file_summaries(root, scan_state["dependency_files"]),
        "agent_files": _relative_file_summaries(root, scan_state["agent_files"]),
        "experiment_files": experiment_files,
        "experiment_tracking": _experiment_tracking_summary(experiment_files),
        "recommended_next_commands": _recommended_next_commands(dataset_candidates),
        "warnings": _scan_warnings(dataset_candidates, scan_state["truncated"]),
    }


def _scan_tree(root: Path, *, max_depth: int, max_entries: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "agent_files": [],
        "code_files": [],
        "dataset_files": [],
        "dependency_files": [],
        "experiment_files": [],
        "truncated": False,
        "visited_entries": 0,
    }
    _visit_directory(root, root, depth=0, max_depth=max_depth, max_entries=max_entries, state=state)
    return state


def _visit_directory(
    root: Path,
    directory: Path,
    *,
    depth: int,
    max_depth: int,
    max_entries: int,
    state: dict[str, Any],
) -> None:
    if state["visited_entries"] >= max_entries:
        state["truncated"] = True
        return

    try:
        entries = sorted(
            directory.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())
        )
    except OSError:
        return

    for entry in entries:
        if state["visited_entries"] >= max_entries:
            state["truncated"] = True
            return

        if entry.is_dir():
            if _ignore_dir(entry):
                continue
            state["visited_entries"] += 1
            if depth < max_depth:
                _visit_directory(
                    root,
                    entry,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_entries=max_entries,
                    state=state,
                )
            continue

        if not entry.is_file():
            continue

        state["visited_entries"] += 1
        _classify_file(root, entry, state)


def _ignore_dir(path: Path) -> bool:
    return path.name in IGNORE_DIRS or path.name.startswith(".ipynb_checkpoints")


def _classify_file(root: Path, file_path: Path, state: dict[str, Any]) -> None:
    name = file_path.name

    if _is_supported_dataset_file(file_path):
        state["dataset_files"].append(file_path)
    suffix = file_path.suffix.lower()
    if suffix == NOTEBOOK_SUFFIX or suffix in SCRIPT_SUFFIXES:
        state["code_files"].append(file_path)
    if name in DEPENDENCY_FILES:
        state["dependency_files"].append(file_path)
    if name in AGENT_FILES or _is_agent_config(file_path, root):
        state["agent_files"].append(file_path)
    if _experiment_file_kind(file_path):
        state["experiment_files"].append(file_path)


def _is_agent_config(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return relative.startswith(".codex/") or relative.startswith(".claude/")


def _dataset_candidates(root: Path, dataset_files: list[Path]) -> list[dict[str, Any]]:
    files_by_directory: dict[Path, list[Path]] = defaultdict(list)
    for file_path in dataset_files:
        files_by_directory[file_path.parent].append(file_path)

    candidates: list[dict[str, Any]] = []
    for directory, files in files_by_directory.items():
        roles = _dataset_roles(files)
        if len(files) > 1 or roles:
            candidates.append(_directory_dataset_candidate(root, directory, files, roles))
        else:
            candidates.append(_file_dataset_candidate(root, files[0]))

    return sorted(candidates, key=lambda candidate: (-candidate["score"], candidate["path"]))


def _directory_dataset_candidate(
    root: Path,
    directory: Path,
    files: list[Path],
    roles: dict[str, str],
) -> dict[str, Any]:
    role_score = (
        4 * int("train" in roles) + 3 * int("test" in roles) + 3 * int("sample_submission" in roles)
    )
    file_score = min(len(files), 5)
    context_score = _context_score(directory)
    score = role_score + file_score + context_score
    kind = "kaggle_dataset_directory" if role_score else "tabular_dataset_directory"

    return {
        "path": _relative_path(root, directory),
        "absolute_path": str(directory),
        "kind": kind,
        "score": score,
        "files": [_relative_path(root, file_path) for file_path in sorted(files)],
        "roles": roles,
        "reasons": _candidate_reasons(kind, roles, len(files), context_score),
        "recommended_command": _command("labmate", "research-brief", str(directory)),
    }


def _file_dataset_candidate(root: Path, file_path: Path) -> dict[str, Any]:
    is_archive = _is_supported_dataset_archive(file_path)
    return {
        "path": _relative_path(root, file_path),
        "absolute_path": str(file_path),
        "kind": "dataset_archive" if is_archive else "tabular_file",
        "score": 3 if is_archive else 2,
        "files": [_relative_path(root, file_path)],
        "roles": {},
        "reasons": [
            "zip archive; run dataset-inspect to inspect members"
            if is_archive
            else "single CSV/TSV file"
        ],
        "recommended_command": _command("labmate", "research-brief", str(file_path)),
    }


def _dataset_roles(files: list[Path]) -> dict[str, str]:
    roles = {}
    for file_path in files:
        normalized = _normalize_name(_tabular_stem(file_path))
        role = _file_role(normalized)
        if role and role not in roles:
            roles[role] = file_path.name
    return roles


def _is_supported_dataset_file(path: Path) -> bool:
    return _tabular_suffix(path) in DATASET_SUFFIXES or _is_supported_dataset_archive(path)


def _is_supported_dataset_archive(path: Path) -> bool:
    return path.suffix.lower() in ARCHIVE_SUFFIXES


def _tabular_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] in ([".csv", ".gz"], [".tsv", ".gz"]):
        return suffixes[-2]
    return path.suffix.lower()


def _tabular_stem(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] in ([".csv", ".gz"], [".tsv", ".gz"]):
        return path.name[: -len("".join(path.suffixes[-2:]))]
    return path.stem


def _file_role(normalized_stem: str) -> str | None:
    if "sample_submission" in normalized_stem or normalized_stem in {"sample", "submission"}:
        return "sample_submission"
    if normalized_stem in {"train", "training"} or normalized_stem.startswith("train_"):
        return "train"
    if normalized_stem in {"test", "testing"} or normalized_stem.startswith("test_"):
        return "test"
    return None


def _context_score(directory: Path) -> int:
    try:
        names = [path.stem for path in directory.iterdir() if path.is_file()]
    except OSError:
        return 0
    return min(
        sum(1 for name in names if any(token in _normalize_name(name) for token in CONTEXT_STEMS)),
        3,
    )


def _candidate_reasons(
    kind: str,
    roles: dict[str, str],
    file_count: int,
    context_score: int,
) -> list[str]:
    reasons = [f"{file_count} CSV/TSV files"]
    if kind == "kaggle_dataset_directory":
        reasons.append("Kaggle-style split file names detected")
    for role, file_name in sorted(roles.items()):
        reasons.append(f"{role}: {file_name}")
    if context_score:
        reasons.append("local competition context file detected")
    return reasons


def _code_entrypoints(root: Path, code_files: list[Path]) -> list[dict[str, Any]]:
    scored = []
    for file_path in code_files:
        score, reasons = _code_score(file_path)
        if score <= 0:
            continue
        scored.append(
            {
                "path": _relative_path(root, file_path),
                "kind": "notebook" if file_path.suffix.lower() == NOTEBOOK_SUFFIX else "script",
                "score": score,
                "reasons": reasons,
            }
        )

    return sorted(scored, key=lambda item: (-item["score"], item["path"]))[:20]


def _code_score(path: Path) -> tuple[int, list[str]]:
    normalized = _normalize_name(path.stem)
    score = 0
    reasons: list[str] = []
    for token, weight in {
        "baseline": 4,
        "train": 4,
        "model": 3,
        "notebook": 2,
        "submit": 3,
        "main": 2,
        "eda": 2,
        "experiment": 2,
    }.items():
        if token in normalized:
            score += weight
            reasons.append(f"name contains {token}")
    if path.suffix.lower() == NOTEBOOK_SUFFIX:
        score += 1
        reasons.append("notebook")
    return score, reasons


def _relative_file_summaries(root: Path, files: list[Path]) -> list[dict[str, str]]:
    return [
        {
            "path": _relative_path(root, file_path),
            "kind": _known_file_kind(file_path, root),
        }
        for file_path in sorted(files)
    ]


def _known_file_kind(path: Path, root: Path) -> str:
    if path.name in DEPENDENCY_FILES:
        return DEPENDENCY_FILES[path.name]
    if path.name in AGENT_FILES:
        return AGENT_FILES[path.name]
    if _is_agent_config(path, root):
        return "agent_config"
    return "file"


def _experiment_file_summaries(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [
        _experiment_file_summary(root, file_path)
        for file_path in sorted(files, key=lambda path: _relative_path(root, path))
    ]


def _experiment_file_summary(root: Path, file_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": _relative_path(root, file_path),
        "kind": _experiment_file_kind(file_path),
    }
    if file_path.suffix.lower() in {".csv", ".tsv"}:
        summary.update(_ledger_table_summary(file_path))
    else:
        summary["read_status"] = "metadata_only"
    return summary


def _experiment_file_kind(path: Path) -> str | None:
    return EXPERIMENT_FILES.get(path.name.lower())


def _ledger_table_summary(path: Path) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                columns = next(reader)
            except StopIteration:
                return {
                    "format": path.suffix.lower().removeprefix("."),
                    "columns": [],
                    "completed_run_count": 0,
                    "row_count_status": "exact",
                    "read_status": "empty",
                }

            completed_run_count = 0
            row_count_status = "exact"
            for _row in reader:
                if completed_run_count >= MAX_LEDGER_ROWS_TO_COUNT:
                    row_count_status = "bounded"
                    break
                completed_run_count += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return {
            "format": path.suffix.lower().removeprefix("."),
            "columns": [],
            "completed_run_count": None,
            "row_count_status": "unknown",
            "read_status": "unreadable",
            "error": str(exc),
        }

    return {
        "format": path.suffix.lower().removeprefix("."),
        "columns": columns,
        "completed_run_count": completed_run_count,
        "row_count_status": row_count_status,
        "read_status": "ok",
    }


def _experiment_tracking_summary(experiment_files: list[dict[str, Any]]) -> dict[str, Any]:
    if not experiment_files:
        return {
            "status": "not_found",
            "recommended_ledger_path": "results.tsv",
            "notes": [
                "Create results.tsv using the research-brief experiment_tracking_plan "
                "before the first run."
            ],
        }

    recommended_ledger_path = next(
        (
            file_info["path"]
            for file_info in experiment_files
            if file_info["kind"] == "experiment_ledger"
        ),
        experiment_files[0]["path"],
    )
    return {
        "status": "existing_tracking_found",
        "recommended_ledger_path": recommended_ledger_path,
        "notes": [
            f"Continue logging runs in {recommended_ledger_path}; do not start a new ledger."
        ],
    }


def _recommended_next_commands(candidates: list[dict[str, Any]]) -> list[str]:
    commands = ["labmate tools"]
    if candidates:
        commands.append(str(candidates[0]["recommended_command"]))
        commands.append(_command("labmate", "dataset-inspect", str(candidates[0]["absolute_path"])))
    return commands


def _scan_warnings(candidates: list[dict[str, Any]], truncated: bool) -> list[str]:
    warnings = []
    if not candidates:
        warnings.append("No local CSV/TSV or zip dataset candidates were found within scan limits.")
    if truncated:
        warnings.append(
            "Project scan hit max_entries; rerun with a higher limit for full coverage."
        )
    return warnings


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return "." if relative.as_posix() == "." else relative.as_posix()


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _command(*parts: str) -> str:
    return shlex.join(parts)
