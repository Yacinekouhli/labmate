---
description: Start a Labmate Kaggle competition workflow.
argument-hint: <competition-url-or-slug> [workspace]
allowed-tools: Bash(uv run labmate:*), Bash(labmate:*), mcp__labmate
---

Start a Kaggle competition workflow for: `$ARGUMENTS`.

Use Labmate as the control plane:

1. Parse the first argument as the Kaggle competition URL or slug. Treat the
   optional second argument as the workspace path.
2. Call `labmate kaggle start <competition> --workspace <workspace>` or the
   Labmate MCP `kaggle_start` tool.
3. If data is missing and Kaggle MCP tools are available, use them for
   competition details and data download. If Kaggle access is blocked, report
   the exact auth, rules, or tooling blocker.
4. Delegate the research pass to `kaggle-researcher` before editing model code.
5. Implement only after metric, target, validation split, leakage risks, and
   submission format are verified.
6. Log every run in `results.tsv`.
7. Do not submit to Kaggle unless the user explicitly approves the exact file
   and submission message.
