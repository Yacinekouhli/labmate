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
- Keep CLI output reproducible and machine-readable.
- Treat MCP as an integration layer over the same tool registry used by CLI scripts.
- Keep Codex and Claude Code integrations as thin wrappers in `integrations/`.

## ML Research Workflow

When `program.md` asks for evidence, gather it before editing code:

```bash
labmate kaggle start <competition-url-or-slug> --workspace <workspace>
labmate tools
labmate project-scan <project-root>
labmate experiment-summary <project-root-or-results.tsv>
labmate research-brief <dataset-path>
labmate dataset-inspect <dataset-path>
labmate benchmark-lookup "<task or dataset>"
labmate literature-search "<paper or method query>" --max-results 5
labmate docs-fetch "<framework API or concept>" --max-results 3
labmate github-find-examples "<implementation pattern>" --max-results 3
```

Summarize the command outputs, cited URLs, dataset risks, metric/protocol
assumptions, and implementation recommendations before changing model code.

For Kaggle competitions, use `labmate kaggle start` to create the workspace,
ledger, and agent handoff first. Submissions are never part of the research
step; ask for explicit user approval for the exact file and message before any
Kaggle submit command or MCP submission tool.

If `project-scan` reports existing experiment files, continue that ledger rather
than creating a parallel run tracker.

## Initial Development Checks

- Run `uv run ruff check .` once Python code exists.
- Run `uv run ruff format --check .` once Python code exists.
- Run `git diff --check` before commits.
