# Labmate

Agent-agnostic ML research tooling for coding agents.

Labmate gives Codex, Claude Code, Cursor, and other coding agents a shared ML
research layer: papers, benchmarks, datasets, docs, code examples, and
experiment planning without depending on one agent runtime.

The host agent owns the model, approvals, shell access, file edits, and context
management. Labmate provides the portable layer:

- reusable `AGENTS.md` and `program.md` workflow conventions
- read-only CLI tools with stable JSON contracts
- an MCP server generated from the same tool registry
- skill documents for recurring ML research workflows
- native wrappers for Codex and Claude Code

## Status

Early project scaffold. Labmate currently has a shared tool registry, stable
CLI JSON contracts, a real stdio MCP server, and setup generators for Codex and
Claude Code. The first implementation goal is a read-only research pack:

- literature search
- citation graph
- dataset inspection
- benchmark lookup
- framework docs fetch
- GitHub example discovery

Mutating tools such as job submission, Kaggle submissions, repository writes,
and experiment-tracker writes are intentionally out of scope for the first
release.

Working today:

- `labmate tools` lists the shared registry.
- `labmate dataset-inspect <path>` inspects local CSV/TSV files and Kaggle-style
  folders containing `train.csv`, `test.csv`, and `sample_submission.csv`.
- `labmate literature-search --query ...` uses the arXiv backend.
- `labmate-mcp` starts a stdio MCP server exposing the read-only registry tools.
- `labmate init codex` and `labmate init claude-code` write non-destructive
  setup artifacts for existing ML repositories.

## Repository Layout

```text
docs/                  Product and architecture notes
templates/             Reproducible ML task templates
src/labmate/           Python package
integrations/codex/    Codex custom agent and plugin wrappers
integrations/claude-code/
                       Claude Code skills, subagents, and plugin wrappers
tests/                 Contract tests
```

## Design Principles

1. Runtime-neutral: any agent can drive Labmate.
2. Scripts first: every capability should work from the CLI.
3. MCP second: typed tools wrap the same registry as the CLI.
4. Read-only first: research and inspection before mutation.
5. Evidence-backed: papers, datasets, docs, and code examples should be cited.
6. Provider-safe: no scraping Codex, Claude, ChatGPT, or other model-provider auth files.

## Codex Sketch

```bash
codex mcp add labmate -- uv run labmate-mcp
```

Then in Codex:

```text
/goal Follow program.md. Research papers, verify dataset schema and current docs, then implement only evidence-backed ML changes.
Spawn ml_researcher for the research pass and wait for its summary before editing.
```

## Claude Code Sketch

```bash
claude mcp add --transport stdio labmate -- uv run labmate-mcp
```

Then in Claude Code:

```text
/goal program.md acceptance criteria are satisfied, the research summary cites sources, dataset schema is verified, and tests pass
/ml-research program.md
Use the ml-researcher subagent for literature, benchmark, dataset, docs, and GitHub evidence.
```
