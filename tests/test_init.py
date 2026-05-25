from pathlib import Path

import pytest

from labmate.init import UnknownHarnessError, apply_init, plan_init


def _actions_by_destination(plan):
    return {file.relative_destination: file.action for file in plan.files}


def test_codex_init_plan_lists_project_files(tmp_path: Path) -> None:
    plan = plan_init("codex", tmp_path)

    assert plan.harness == "codex"
    assert _actions_by_destination(plan) == {
        "AGENTS.md": "write",
        "program.md": "write",
        ".codex/agents/ml-researcher.toml": "write",
        ".codex/skills/ml-research/SKILL.md": "write",
        ".codex/labmate.mcp.json": "write",
    }
    assert "/goal Follow program.md" in plan.goal_prompt
    assert plan.follow_up_commands == ("codex mcp add labmate -- uv run labmate-mcp",)


def test_claude_init_plan_lists_project_files(tmp_path: Path) -> None:
    plan = plan_init("claude", tmp_path)

    assert plan.harness == "claude-code"
    assert _actions_by_destination(plan) == {
        "AGENTS.md": "write",
        "program.md": "write",
        ".claude/agents/ml-researcher.md": "write",
        ".claude/skills/ml-research/SKILL.md": "write",
        ".mcp.json": "write",
    }
    assert "/ml-research program.md" in plan.goal_prompt
    assert plan.follow_up_commands == (
        "claude mcp add --transport stdio labmate -- uv run labmate-mcp",
    )


def test_init_plan_skips_existing_files_by_default(tmp_path: Path) -> None:
    existing_program = tmp_path / "program.md"
    existing_program.write_text("custom program\n")

    plan = plan_init("codex", tmp_path)

    assert _actions_by_destination(plan)["program.md"] == "skip"
    assert "program.md" not in {file.relative_destination for file in plan.writable_files}

    result = apply_init(plan)

    assert existing_program.read_text() == "custom program\n"
    assert existing_program in result.skipped


def test_init_plan_can_request_overwrites(tmp_path: Path) -> None:
    existing_program = tmp_path / "program.md"
    existing_program.write_text("custom program\n")

    plan = plan_init("codex", tmp_path, overwrite=True)

    assert _actions_by_destination(plan)["program.md"] == "overwrite"


def test_init_plan_does_not_overwrite_directory_collisions(tmp_path: Path) -> None:
    (tmp_path / "program.md").mkdir()

    plan = plan_init("codex", tmp_path, overwrite=True)

    assert _actions_by_destination(plan)["program.md"] == "skip"


def test_apply_init_copies_only_planned_writes(tmp_path: Path) -> None:
    existing_agents = tmp_path / "AGENTS.md"
    existing_agents.write_text("keep local instructions\n")

    plan = plan_init("claude-code", tmp_path)
    result = apply_init(plan)

    assert existing_agents.read_text() == "keep local instructions\n"
    assert (tmp_path / "program.md").exists()
    assert (tmp_path / ".claude" / "agents" / "ml-researcher.md").exists()
    assert (tmp_path / ".claude" / "skills" / "ml-research" / "SKILL.md").exists()
    assert (tmp_path / ".mcp.json").exists()
    assert existing_agents in result.skipped
    assert {path.relative_to(tmp_path).as_posix() for path in result.written} == {
        "program.md",
        ".claude/agents/ml-researcher.md",
        ".claude/skills/ml-research/SKILL.md",
        ".mcp.json",
    }


def test_unknown_harness_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnknownHarnessError):
        plan_init("cursor", tmp_path)
