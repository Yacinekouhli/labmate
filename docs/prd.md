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

- Provide read-only ML research tools that any agent can drive.
- Support direct CLI use and typed MCP use from one shared registry.
- Provide native-feeling Codex and Claude Code setup plus generic MCP setup for
  other hosts.
- Keep the core independent from a specific model provider or agent runtime.

## Non-Goals

- Build a chat UI.
- Implement a custom agent loop.
- Reuse Codex, ChatGPT, Claude, or other provider auth files.
- Ship mutating tools in the first release.
- Depend on Hugging Face as the only backend.

## MVP

The first release should include:

- `program.md` task protocol
- read-only tool registry
- CLI JSON contracts
- local project scan for unknown ML repositories
- literature search skeleton
- first-pass research brief over local datasets and benchmark context
- dataset inspection skeleton
- benchmark lookup skeleton
- docs fetch skeleton
- GitHub example discovery skeleton
- Codex custom agent/plugin examples
- Claude Code skill/subagent/plugin examples
- non-destructive Codex, Claude Code, and generic MCP init plans for existing ML repositories

## Success Criteria

- A coding agent can read `AGENTS.md` and `program.md`, run Labmate tools, and produce an evidence-backed ML implementation plan.
- CLI and MCP surfaces expose the same tool definitions.
- Codex can use `/goal` and an `ml_researcher` custom agent.
- Claude Code can use `/goal`, `/ml-research`, and an `ml-researcher` subagent.
