# Architecture

Labmate has one portable core and thin harness-specific wrappers.

## Core

The core owns:

- tool definitions
- input and output schemas
- backend implementations
- CLI entrypoints
- MCP server generation
- tests for contracts

The core must not depend on Codex, Claude Code, Cursor, or another harness.

## Tool Registry

Each capability is represented by a shared tool definition:

```python
ToolDefinition(
    name="dataset_inspect",
    description="Inspect dataset schema, splits, sample rows, and task fit.",
    read_only=True,
    backends=("huggingface", "kaggle", "openml", "uci", "local"),
)
```

This registry should power:

- CLI command discovery
- MCP tool schemas
- skill documentation examples
- tests
- allowlisting

## Harness Wrappers

Harness wrappers translate native UX into the same core contracts.

Codex wrapper:

- `.codex/agents/ml-researcher.toml`
- Codex plugin manifest
- Codex skill
- plugin-local `.mcp.json`

Claude Code wrapper:

- `.claude/skills/ml-research/SKILL.md`
- `.claude/agents/ml-researcher.md`
- Claude plugin manifest
- plugin-local `.mcp.json`

## Safety

The initial release is read-only. Mutating operations need a separate design
with dry-run behavior, approval requirements, and audit summaries.

