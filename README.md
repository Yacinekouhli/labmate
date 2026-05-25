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
CLI JSON contracts, a real stdio MCP server, and setup generators for Codex,
Claude Code, and generic MCP/CLI hosts. The first implementation goal is a
read-only research pack:

- literature search
- citation graph
- dataset inspection
- benchmark lookup
- framework docs fetch
- GitHub example discovery

Mutating tools such as job submission, Kaggle submissions, repository writes,
and experiment-tracker writes are intentionally out of scope for the first
release.

## Quickstart

From a checkout:

```bash
uv sync
uv run labmate tools
uv run labmate project-scan /path/to/ml-repo
uv run labmate research-brief /path/to/kaggle/data --max-benchmarks 3
uv run labmate dataset-inspect /path/to/kaggle/data
uv run labmate benchmark-lookup "tabular classification"
uv run labmate docs-fetch "xgboost gpu parameters" --max-results 3
uv run labmate github-find-examples "xgboost tabular kaggle" --max-results 3
```

Add Labmate to a target ML repository without overwriting existing files:

```bash
uv run labmate init codex /path/to/ml-repo
uv run labmate init claude-code /path/to/ml-repo
uv run labmate init generic /path/to/ml-repo
```

Apply the generated files when the dry run looks right:

```bash
uv run labmate init codex /path/to/ml-repo --apply
```

Working today:

- `labmate tools` lists the shared registry with schemas, backends, and CLI/MCP
  usage examples.
- `labmate project-scan <path>` scans an unknown ML repo for likely datasets,
  code entrypoints, dependency files, agent setup, and next Labmate commands.
- `labmate research-brief <path>` creates a concise first-pass ML brief by
  combining local dataset inspection, rough task inference, local benchmark
  context, local metric hints from documentation files, target distribution,
  validation/split columns, a baseline modeling plan, a structured follow-up
  research plan, recommended commands, and an implementation checklist.
- `labmate dataset-inspect <path>` inspects local CSV/TSV files, gzipped
  CSV/TSV files, zip archives, and Kaggle-style folders containing
  train/test/submission files.
- `labmate literature-search <query>` uses the arXiv backend.
- `labmate citation-graph <paper-id>` uses a local ML paper corpus for citation
  context around common tabular references such as `arxiv:1603.02754`.
- `labmate benchmark-lookup <query>` searches a curated local benchmark catalog
  for task, metric, protocol, pitfalls, and baseline hints.
- `labmate docs-fetch <query>` searches an official-docs catalog, and
  `--url <docs-url>` fetches exact framework documentation pages.
- `labmate github-find-examples <query>` finds candidate public GitHub
  repositories for implementation evidence; with `--repository owner/repo`, it
  also returns small public file snippets from matched paths.
- `labmate-mcp` starts a stdio MCP server exposing the read-only registry tools.
- `labmate init codex`, `labmate init claude-code`, and `labmate init generic`
  write non-destructive setup artifacts for existing ML repositories.

Still stubbed or limited:

- `citation-graph --backend semantic_scholar` and `--backend openalex` return
  structured backend-unavailable failures; the default local corpus works.
- `benchmark-lookup --backend papers_with_code` and `--backend openml` return
  structured backend-unavailable failures; the default local catalog works.
- `github-find-examples` uses unauthenticated GitHub APIs. Cross-repository
  code search still needs a future authenticated backend.

## Agent Workflow

A coding agent should use Labmate before editing model code:

```bash
uv run labmate project-scan .
uv run labmate research-brief data/ --max-benchmarks 3
uv run labmate dataset-inspect data/
uv run labmate benchmark-lookup "tabular classification auc"
uv run labmate literature-search "tabular classification baseline" --max-results 5
uv run labmate citation-graph arxiv:1603.02754 --max-results 3
uv run labmate docs-fetch "sklearn ColumnTransformer pipeline" --max-results 3
uv run labmate github-find-examples "sklearn pipeline tabular classification" --max-results 3
```

The `research_plan` field contains ordered tool calls with arguments, purpose,
and evidence to extract. The implementation plan should cite the returned URLs,
dataset warnings, benchmark metric/protocol assumptions, modeling-plan choices,
and relevant examples before proposing changes.

## Repository Layout

```text
docs/                  Product and architecture notes
templates/             Reproducible ML task templates
src/labmate/           Python package
integrations/codex/    Codex custom agent and plugin wrappers
integrations/claude-code/
                       Claude Code skills, subagents, and plugin wrappers
integrations/generic/  Portable MCP snippet for other agent hosts
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
