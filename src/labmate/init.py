"""Non-destructive project scaffolding for Labmate harness integrations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal

Harness = Literal["codex", "claude-code", "generic"]
PlanAction = Literal["write", "skip", "overwrite"]
TemplateSource = Path | Traversable


@dataclass(frozen=True)
class TemplateFile:
    """A source template and its destination inside a target project."""

    source: TemplateSource
    destination: Path


@dataclass(frozen=True)
class PlannedFile:
    """A file operation proposed by an init plan."""

    source: TemplateSource
    destination: Path
    relative_destination: str
    action: PlanAction
    reason: str


@dataclass(frozen=True)
class InitPlan:
    """Dry-run result for initializing one agent harness."""

    harness: Harness
    project_root: Path
    files: tuple[PlannedFile, ...]
    goal_prompt: str
    follow_up_commands: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def writable_files(self) -> tuple[PlannedFile, ...]:
        """Files that would be written by applying this plan."""

        return tuple(file for file in self.files if file.action in {"write", "overwrite"})


@dataclass(frozen=True)
class AppliedInit:
    """Result of applying an init plan."""

    plan: InitPlan
    written: tuple[Path, ...]
    skipped: tuple[Path, ...]


class UnknownHarnessError(ValueError):
    """Raised when the requested harness does not have a Labmate integration."""


class MissingTemplateError(FileNotFoundError):
    """Raised when a source integration template is missing from the checkout."""


def plan_init(
    harness: str,
    project_root: str | Path,
    *,
    overwrite: bool = False,
    template_root: str | Path | None = None,
) -> InitPlan:
    """Return the files and prompts needed to initialize Labmate in a project.

    The function performs no writes. Existing files are marked as ``skip`` unless
    ``overwrite`` is true, which lets agents show a safe dry run before mutating
    a user's Kaggle or research repository.
    """

    normalized_harness = _normalize_harness(harness)
    resolved_template_root = _resolve_template_root(template_root)
    resolved_project_root = Path(project_root).expanduser().resolve()

    files = tuple(
        _planned_file(template, resolved_project_root, overwrite=overwrite)
        for template in _templates_for_harness(normalized_harness, resolved_template_root)
    )

    return InitPlan(
        harness=normalized_harness,
        project_root=resolved_project_root,
        files=files,
        goal_prompt=_goal_prompt(normalized_harness),
        follow_up_commands=_follow_up_commands(normalized_harness),
        notes=_notes_for_harness(normalized_harness),
    )


def apply_init(plan: InitPlan) -> AppliedInit:
    """Copy the files selected by an init plan.

    Applying the default plan is non-destructive because pre-existing files are
    planned as ``skip``. To replace existing files, build the plan with
    ``overwrite=True`` so the destructive choice is visible in the dry-run output.
    """

    written: list[Path] = []
    skipped: list[Path] = []

    for planned_file in plan.files:
        if planned_file.action == "skip":
            skipped.append(planned_file.destination)
            continue

        planned_file.destination.parent.mkdir(parents=True, exist_ok=True)
        planned_file.destination.write_bytes(planned_file.source.read_bytes())
        written.append(planned_file.destination)

    return AppliedInit(plan=plan, written=tuple(written), skipped=tuple(skipped))


def _normalize_harness(harness: str) -> Harness:
    normalized = harness.strip().lower().replace("_", "-")
    aliases = {
        "agent": "generic",
        "agents": "generic",
        "claude": "claude-code",
        "claude-code": "claude-code",
        "codex": "codex",
        "cursor": "generic",
        "generic": "generic",
        "mcp": "generic",
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as exc:
        supported = ", ".join(sorted(set(aliases.values())))
        message = f"Unsupported harness {harness!r}. Supported: {supported}."
        raise UnknownHarnessError(message) from exc


def _resolve_template_root(template_root: str | Path | None) -> TemplateSource:
    if template_root is not None:
        root = Path(template_root).expanduser().resolve()
        if _template_root_has_files(root):
            return root
        raise MissingTemplateError(f"Could not locate Labmate templates under {root}")

    resource_root = files("labmate").joinpath("resources")
    if _template_root_has_files(resource_root):
        return resource_root

    source_root = Path(__file__).resolve().parents[2]
    if _template_root_has_files(source_root):
        return source_root

    raise MissingTemplateError(
        "Could not locate Labmate templates. Pass template_root when running outside "
        "the source checkout."
    )


def _template_root_has_files(root: TemplateSource) -> bool:
    return (
        _join(root, "AGENTS.md").is_file()
        and _join(root, "templates").is_dir()
        and _join(root, "integrations").is_dir()
    )


def _join(root: TemplateSource, *parts: str) -> TemplateSource:
    if isinstance(root, Path):
        return root.joinpath(*parts)
    return root.joinpath(*parts)


def _templates_for_harness(
    harness: Harness, template_root: TemplateSource
) -> tuple[TemplateFile, ...]:
    common = (
        TemplateFile(_join(template_root, "AGENTS.md"), Path("AGENTS.md")),
        TemplateFile(_join(template_root, "templates", "program.md"), Path("program.md")),
    )

    if harness == "codex":
        harness_templates = (
            TemplateFile(
                _join(template_root, "integrations", "codex", "agents", "ml-researcher.toml"),
                Path(".codex") / "agents" / "ml-researcher.toml",
            ),
            TemplateFile(
                _join(
                    template_root,
                    "integrations",
                    "codex",
                    "agents",
                    "kaggle-researcher.toml",
                ),
                Path(".codex") / "agents" / "kaggle-researcher.toml",
            ),
            TemplateFile(
                _join(
                    template_root,
                    "integrations",
                    "codex",
                    "plugin",
                    "skills",
                    "ml-research",
                    "SKILL.md",
                ),
                Path(".codex") / "skills" / "ml-research" / "SKILL.md",
            ),
            TemplateFile(
                _join(template_root, "integrations", "codex", "plugin", ".mcp.json"),
                Path(".codex") / "labmate.mcp.json",
            ),
        )
    elif harness == "claude-code":
        harness_templates = (
            TemplateFile(
                _join(
                    template_root,
                    "integrations",
                    "claude-code",
                    "agents",
                    "ml-researcher.md",
                ),
                Path(".claude") / "agents" / "ml-researcher.md",
            ),
            TemplateFile(
                _join(
                    template_root,
                    "integrations",
                    "claude-code",
                    "agents",
                    "kaggle-researcher.md",
                ),
                Path(".claude") / "agents" / "kaggle-researcher.md",
            ),
            TemplateFile(
                _join(
                    template_root,
                    "integrations",
                    "claude-code",
                    "commands",
                    "kagglethis.md",
                ),
                Path(".claude") / "commands" / "kagglethis.md",
            ),
            TemplateFile(
                _join(
                    template_root,
                    "integrations",
                    "claude-code",
                    "skills",
                    "ml-research",
                    "SKILL.md",
                ),
                Path(".claude") / "skills" / "ml-research" / "SKILL.md",
            ),
            TemplateFile(
                _join(template_root, "integrations", "claude-code", "plugin", ".mcp.json"),
                Path(".mcp.json"),
            ),
        )
    else:
        harness_templates = (
            TemplateFile(
                _join(template_root, "integrations", "generic", ".mcp.json"),
                Path(".mcp.json"),
            ),
        )

    return common + harness_templates


def _planned_file(template: TemplateFile, project_root: Path, *, overwrite: bool) -> PlannedFile:
    if not template.source.is_file():
        raise MissingTemplateError(f"Missing Labmate template: {template.source}")

    destination = project_root / template.destination
    if destination.exists() and not destination.is_file():
        action = "skip"
        reason = "exists and is not a file"
    elif destination.exists():
        action: PlanAction = "overwrite" if overwrite else "skip"
        reason = "exists and overwrite is enabled" if overwrite else "exists; keep local file"
    else:
        action = "write"
        reason = "missing"

    return PlannedFile(
        source=template.source,
        destination=destination,
        relative_destination=template.destination.as_posix(),
        action=action,
        reason=reason,
    )


def _goal_prompt(harness: Harness) -> str:
    if harness == "codex":
        return (
            "/goal Follow program.md. For Kaggle work, start with `labmate kaggle start "
            "<competition>`, spawn kaggle_researcher for the research pass, verify dataset "
            "schema/metric/validation, and do not submit without explicit approval."
        )

    if harness == "generic":
        return (
            "Follow AGENTS.md and program.md. Start with `labmate project-scan <project-root>`, "
            "then run the recommended `labmate research-brief ...` command before editing ML code."
        )

    return (
        "/goal program.md acceptance criteria are satisfied, the research summary cites sources, "
        "dataset schema is verified, and tests pass\n"
        "/kagglethis <competition-url-or-slug>"
    )


def _follow_up_commands(harness: Harness) -> tuple[str, ...]:
    if harness == "codex":
        return ("codex mcp add labmate -- uv run labmate-mcp",)

    if harness == "generic":
        return (
            "uv run labmate tools",
            "uv run labmate project-scan <project-root>",
        )

    return ("claude mcp add --transport stdio labmate -- uv run labmate-mcp",)


def _notes_for_harness(harness: Harness) -> tuple[str, ...]:
    common = (
        "Review AGENTS.md and program.md before editing an existing repository.",
        "Existing files are skipped by default so local agent configuration is preserved.",
    )

    if harness == "codex":
        return common + (
            ".codex/labmate.mcp.json is a mergeable MCP snippet; add it through Codex MCP config.",
            "Use kaggle_researcher for competition research before editing model code.",
        )

    if harness == "generic":
        return common + (
            ".mcp.json is a portable MCP snippet for hosts that read mcpServers-style config.",
            "Cursor currently uses the generic setup; no Cursor-specific files are generated.",
        )

    return common + (
        ".mcp.json is written only when absent; merge manually if the project "
        "already has MCP config.",
        "Use /kagglethis <competition-url-or-slug> for Kaggle competition bootstrapping.",
    )
