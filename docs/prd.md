# Labmate PRD

## Summary

Labmate is a portable ML research and engineering toolkit for coding agents.
It extracts the useful part of an ML research assistant into reusable tools,
skills, MCP schemas, and task protocols.

## Problem

ML work inside coding agents needs live research context:

- recent papers and citations
- benchmark/task definitions
- dataset schema and license checks
- current framework docs
- working implementation examples
- reproducible experiment criteria

Standalone ML agents can own this end to end, but users increasingly work in
existing coding agents. Labmate should meet those agents where they are.

## Goals

- Provide ML research tools that any agent can drive, with explicit risk labels
  for read-only and mutating setup actions.
- Support direct CLI use and typed MCP use from one shared registry.
- Provide native-feeling Codex and Claude Code setup plus generic MCP setup for
  other hosts.
- Keep the core independent from a specific model provider or agent runtime.

## Non-Goals

- Build a chat UI.
- Implement a custom agent loop.
- Reuse Codex, ChatGPT, Claude, or other provider auth files.
- Ship auto-submission or opaque mutating ML actions in the first release.
- Depend on Hugging Face as the only backend.

## MVP

The first release should include:

- `program.md` task protocol
- shared tool registry with read-only/mutating risk metadata
- Kaggle workspace bootstrap from a competition URL or slug
- CLI JSON contracts
- local project scan for unknown ML repositories, including existing experiment ledgers
- experiment summary over existing local run ledgers
- safe local creation of `program.md`, `results.tsv`, workspace directories, and
  generated competition brief for Kaggle work
- literature search with arXiv-backed results
- first-pass research brief over local datasets, benchmark context, and ordered
  follow-up research actions, including provided validation/split-column
  guidance, sample-submission format, prior experiment context, and an
  experiment-tracking plan
- dataset inspection with plain/gzipped CSV and TSV support, zip archive
  inspection, Kaggle-style split hints, sample-submission row-count alignment,
  and target-distribution hints
- benchmark lookup with a curated local catalog
- docs fetch with local catalog and direct official-doc URL fetch
- GitHub example discovery with unauthenticated repository search and known-repo snippets
- Codex custom agent/plugin examples
- Codex `kaggle_researcher` custom agent example
- Claude Code `/kagglethis` command, skill, `kaggle-researcher` subagent, and
  plugin examples
- MCP `kagglethis` prompt that maps host slash-command UX to the shared
  `kaggle_start` tool
- non-destructive Codex, Claude Code, and generic MCP init plans for existing ML repositories

## Success Criteria

- A coding agent can read `AGENTS.md` and `program.md`, run Labmate tools, and produce an evidence-backed ML implementation and experiment-tracking plan.
- CLI and MCP surfaces expose the same tool definitions.
- Codex can use `/goal` and an `ml_researcher` custom agent.
- Codex can use `kaggle_researcher` for competition research before edits.
- Claude Code can use `/goal`, `/ml-research`, `/kagglethis`, `ml-researcher`,
  and `kaggle-researcher`.
- Kaggle submission remains separated from setup/research and requires explicit
  user approval for the exact file and message.
