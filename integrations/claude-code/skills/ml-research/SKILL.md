---
description: Use Labmate to research ML papers, datasets, benchmarks, docs, and code examples before implementation.
---

# ML Research

Use this skill when the user asks for evidence-backed ML implementation work,
model training changes, dataset inspection, benchmark research, or paper-driven
experiments.

Workflow:

1. Read `program.md` if present.
2. Ask the `ml-researcher` subagent to gather evidence when the research pass
   would add noisy context to the main conversation.
3. Use Labmate CLI or MCP tools for papers, citations, datasets, benchmarks,
   docs, and code examples.
4. Summarize sources and risks before editing code.
5. Only recommend implementation steps that are grounded in retrieved evidence.

Suggested Claude Code prompt:

```text
/goal program.md acceptance criteria are satisfied, the research summary cites sources, dataset schema is verified, and tests pass
/ml-research program.md
```

