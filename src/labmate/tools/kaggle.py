"""Kaggle competition workspace workflow helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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
                    "surface": "agent",
                    "action": "implement_baseline",
                    "purpose": (
                        "Create a dummy/simple baseline with trusted validation before "
                        "leaderboard submissions."
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
