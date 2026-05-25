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
3. Run focused Labmate tools:
   - `labmate project-scan <project-root>`
   - `labmate research-brief <dataset-path>`
   - `labmate dataset-inspect <dataset-path>`
   - `labmate benchmark-lookup "<task or dataset>"`
   - `labmate literature-search "<method or task query>" --max-results 5`
   - `labmate docs-fetch "<framework API or concept>" --max-results 3`
   - `labmate github-find-examples "<implementation pattern>" --max-results 3`
4. Summarize command outputs, URLs, dataset risks, and metric/protocol assumptions.
5. Only recommend implementation steps that are grounded in retrieved evidence.

Suggested Codex prompt:

```text
/goal Follow program.md. Research papers, verify dataset schema and current docs, then implement only evidence-backed ML changes.
Spawn ml_researcher for the research pass and wait for its summary before editing.
```
