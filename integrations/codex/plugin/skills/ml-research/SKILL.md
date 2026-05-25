---
name: ml-research
description: Use Labmate to research ML papers, datasets, benchmarks, docs, and code examples before implementation.
---

# ML Research

Use this skill when the user asks for evidence-backed ML implementation work,
model training changes, dataset inspection, benchmark research, or paper-driven
experiments.

Workflow:

1. Read `program.md` if present.
2. Identify the research questions blocking implementation.
3. Use Labmate CLI or MCP tools for papers, citations, datasets, benchmarks,
   docs, and code examples.
4. Summarize sources and risks before editing code.
5. Only recommend implementation steps that are grounded in retrieved evidence.

Suggested Codex prompt:

```text
/goal Follow program.md. Research papers, verify dataset schema and current docs, then implement only evidence-backed ML changes.
Spawn ml_researcher for the research pass and wait for its summary before editing.
```

