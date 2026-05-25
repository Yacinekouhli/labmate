# Agent Notes

## Project Intent

Labmate is an agent-agnostic ML research toolkit for coding agents.

The core project must stay independent from any one harness. Codex, Claude Code,
Cursor, and other agents should all use the same underlying CLI tools, MCP tool
schemas, and workflow documents.

## Design Rules

- Keep the core runtime-neutral.
- Prefer read-only research and inspection tools before mutating tools.
- Do not reuse model-provider auth files or private backend endpoints.
- Keep CLI output reproducible and machine-readable with `--json`.
- Treat MCP as an integration layer over the same tool registry used by CLI scripts.
- Keep Codex and Claude Code integrations as thin wrappers in `integrations/`.

## Initial Development Checks

- Run `uv run ruff check .` once Python code exists.
- Run `uv run ruff format --check .` once Python code exists.
- Run `git diff --check` before commits.
