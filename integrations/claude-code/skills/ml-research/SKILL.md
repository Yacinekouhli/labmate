---
description: Use Labmate to research ML papers, datasets, Kaggle competitions, benchmarks, docs, and code examples before implementation.
---

# ML Research

Use this skill when the user asks for evidence-backed ML implementation work,
Kaggle competition progress, model training changes, dataset inspection,
benchmark research, or paper-driven experiments.

Workflow:

1. Read `program.md` if present.
2. Ask the `ml-researcher` subagent to gather evidence when the research pass
   would add noisy context to the main conversation.
3. Run focused Labmate tools:
   - `labmate kaggle start <competition-url-or-slug> --workspace <workspace>`
   - `labmate project-scan <project-root>`
   - `labmate experiment-summary <project-root-or-results.tsv>`
   - `labmate research-brief <dataset-path>`
   - `labmate dataset-inspect <dataset-path>`
   - `labmate benchmark-lookup "<task or dataset>"`
   - `labmate literature-search "<method or task query>" --max-results 5`
   - `labmate docs-fetch "<framework API or concept>" --max-results 3`
   - `labmate github-find-examples "<implementation pattern>" --max-results 3`
4. For Kaggle work, use `/kagglethis` or the `kaggle-researcher` subagent before
   editing model code. Do not submit unless the user explicitly approves the
   exact file and message.
5. Summarize command outputs, URLs, dataset risks, metric/protocol assumptions,
   prior experiments, and the experiment-tracking plan.
6. Only recommend implementation steps that are grounded in retrieved evidence.

Suggested Claude Code prompt:

```text
/goal program.md acceptance criteria are satisfied, the research summary cites sources, dataset schema is verified, and tests pass
/kagglethis <competition-url-or-slug>
```
