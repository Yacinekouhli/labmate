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

## MCP Surface

The MCP server is an adapter boundary, not a second source of truth. The
`labmate.mcp_server` module derives MCP-compatible tool metadata directly from
the shared registry and exposes it through `list_mcp_tools()`,
`labmate-mcp --list-tools`, and a stdio MCP transport.

The MCP adapter exposes the read-only registry tools. Tool calls use the same
`labmate.tool.v1` response contract as CLI commands, including structured
failure payloads for invalid arguments and unavailable backends. This keeps
Codex, Claude Code, and other MCP clients aligned with the same registry used by
CLI commands and tests.

## CLI Contracts

CLI tool commands emit one JSON response shape for both successful and failed
runs. The contract is intentionally small so a coding agent can compare tool
outputs across runs during a Kaggle-style investigation.

Every response includes:

- `schema_version`
- `ok`
- `tool`
- `exit_code`
- either `result` for success or `error` for failure
- `metadata`

Failures use a structured error object with `code`, `message`, `retryable`, and
`details`. Backend-specific fields belong in `details` so callers can distinguish
a missing Kaggle token from a temporary literature API rate limit without parsing
free-form text.

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
