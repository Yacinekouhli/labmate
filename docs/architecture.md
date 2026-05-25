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
- concrete CLI/MCP usage examples
- tests
- allowlisting

Workflow tools, such as `research_brief`, may compose lower-level read-only
tools, but they still enter through the registry and return the same
`labmate.tool.v1` contract. When a workflow recommends more tool calls, it
should expose those as structured actions with tool names, arguments, purpose,
and evidence expectations, not only as shell strings.

Mutating setup tools, such as `kaggle_start`, also enter through the registry.
They must be explicit about risk, write only local workspace artifacts, return
the exact files touched, and keep irreversible or remote actions out of band.

## MCP Surface

The MCP server is an adapter boundary, not a second source of truth. The
`labmate.mcp_server` module derives MCP-compatible tool metadata directly from
the shared registry and exposes it through `list_mcp_tools()`,
`labmate-mcp --list-tools`, and a stdio MCP transport.

The MCP adapter exposes the shared registry tools with read-only/mutating risk
annotations. Tool calls use the same `labmate.tool.v1` response contract as CLI
commands, including structured failure payloads for invalid arguments and
unavailable backends. This keeps Codex, Claude Code, and other MCP clients
aligned with the same registry used by CLI commands and tests.

The MCP adapter also exposes prompts for host-native slash-command UX. The
first prompt is `kagglethis`, which tells Claude Code or another MCP host how to
call `kaggle_start`, when to use Kaggle MCP tools, and where submission approval
is required.

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
- `.codex/agents/kaggle-researcher.toml`
- Codex plugin manifest
- Codex skill
- plugin-local `.mcp.json`

Claude Code wrapper:

- `.claude/skills/ml-research/SKILL.md`
- `.claude/agents/ml-researcher.md`
- `.claude/agents/kaggle-researcher.md`
- `.claude/commands/kagglethis.md`
- Claude plugin manifest
- plugin-local `.mcp.json`

Generic wrapper:

- `AGENTS.md`
- `program.md`
- `.mcp.json` with a `labmate` stdio server entry

## Safety

Most research tools are read-only. Local workspace bootstrap is allowed when it
is explicit and auditable. Remote or irreversible operations, especially Kaggle
submissions, need separate approval requirements and audit summaries before they
run.
