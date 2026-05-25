# Labmate

Agent-agnostic ML research and Kaggle competition tooling for coding agents.

Labmate gives Codex, Claude Code, Cursor, and other coding agents a shared ML
research layer: Kaggle competition setup, papers, benchmarks, datasets, docs,
code examples, and experiment planning without depending on one agent runtime.

The host agent owns the model, approvals, shell access, file edits, and context
management. Labmate provides the portable layer:

- reusable `AGENTS.md` and `program.md` workflow conventions
- CLI tools with stable JSON contracts
- an MCP server generated from the same tool registry
- an MCP prompt that Claude Code can surface as `/mcp__labmate__kagglethis`
- skill documents for recurring ML research workflows
- native wrappers for Codex and Claude Code

## Status

Early project scaffold. Labmate currently has a shared tool registry, stable
CLI JSON contracts, a real stdio MCP server, and setup generators for Codex,
Claude Code, and generic MCP/CLI hosts. The first product slice is a
Kaggle-ready research pack:

- Kaggle workspace bootstrap
- literature search
- citation graph
- dataset inspection
- experiment summary
- benchmark lookup
- framework docs fetch
- GitHub example discovery
- Claude `/kagglethis` command and `kaggle-researcher` subagent templates

The Kaggle workspace bootstrap writes only local project scaffolding: directories,
`program.md`, `results.tsv`, metadata, and a generated brief. Kaggle submissions
remain approval-gated and are intentionally separate from research/setup.

## Quickstart

From a checkout:

```bash
uv sync
uv run labmate tools
uv run labmate kaggle start titanic --workspace ./titanic
uv run labmate project-scan /path/to/ml-repo
uv run labmate experiment-summary /path/to/ml-repo
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
- `labmate kaggle start <competition-url-or-slug>` creates or updates a local
  Kaggle workspace, normalizes competition URLs, creates `data/`, `runs/`,
  `submissions/`, `reports/`, `program.md`, `results.tsv`, and a generated
  competition brief, tries Kaggle CLI download when enabled, inspects local data
  when available, and returns a Claude/Codex/MCP handoff.
- `labmate project-scan <path>` scans an unknown ML repo for likely datasets,
  code entrypoints, dependency files, existing experiment ledgers, agent setup,
  and next Labmate commands.
- `labmate experiment-summary <path>` summarizes existing `results.tsv`-style
  experiment ledgers, including best run, latest run, metric direction, and
  status counts.
- `labmate research-brief <path>` creates a concise first-pass ML brief by
  combining local dataset inspection, rough task inference, local benchmark
  context, local metric hints from documentation files, target distribution,
  validation/split columns, sample-submission format, a baseline modeling plan,
  a structured experiment-tracking plan, prior experiment context when a nearby
  ledger exists, a follow-up research plan, recommended commands, and an
  implementation checklist.
- `labmate dataset-inspect <path>` inspects local CSV/TSV files, gzipped
  CSV/TSV files, zip archives, and Kaggle-style folders containing
  train/test/submission files, including sample-submission row-count alignment.
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
- `labmate-mcp` starts a stdio MCP server exposing the shared registry tools
  with read-only/mutating risk annotations.
- `labmate-mcp --list-prompts` shows the `kagglethis` MCP prompt for Claude Code
  slash-command discovery.
- `labmate init codex`, `labmate init claude-code`, and `labmate init generic`
  write non-destructive setup artifacts for existing ML repositories.

Still stubbed or limited:

- `citation-graph --backend semantic_scholar` and `--backend openalex` return
  structured backend-unavailable failures; the default local corpus works.
- `benchmark-lookup --backend papers_with_code` and `--backend openml` return
  structured backend-unavailable failures; the default local catalog works.
- `github-find-examples` uses unauthenticated GitHub APIs. Cross-repository
  code search still needs a future authenticated backend.
- `kaggle_start` can use a local Kaggle CLI if configured. If the CLI, auth, or
  rules acceptance is missing, it keeps the workspace and returns structured
  next actions for Kaggle MCP or manual setup.

## Kaggle UX

Claude Code project setup gives the cleanest flow:

```bash
uv run labmate init claude-code /path/to/workspace --apply
claude mcp add --transport stdio labmate -- uv run labmate-mcp
```

Then in Claude Code:

```text
/kagglethis https://www.kaggle.com/competitions/titanic ./titanic
```

If the MCP server is connected, Claude can also discover the MCP prompt:

```text
/mcp__labmate__kagglethis titanic
```

The command/subagent flow is:

1. create/update the local workspace with `kaggle_start`
2. use Kaggle CLI or Kaggle MCP for current competition details and data download
3. inspect `train`, `test`, and `sample_submission`
4. infer target, task, metric hints, leakage risks, validation strategy, and baseline plan
5. log every run in `results.tsv`
6. ask for explicit approval before any Kaggle submission

## Agent Workflow

A coding agent should use Labmate before editing model code:

```bash
uv run labmate kaggle start titanic --workspace ./titanic --no-download
uv run labmate project-scan .
uv run labmate experiment-summary .
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
experiment-tracking plan, prior experiments, and relevant examples before
proposing changes.

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
/goal Work on Kaggle competition titanic. Start with labmate kaggle start, spawn kaggle_researcher for the research pass, verify metric/schema/validation, and do not submit without explicit approval.
```

## Claude Code Sketch

```bash
claude mcp add --transport stdio labmate -- uv run labmate-mcp
```

Then in Claude Code:

```text
/kagglethis titanic
Use the kaggle-researcher subagent for rules, metric, dataset, validation, prior runs, and first baseline planning.
```
