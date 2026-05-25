---
name: ml-research
description: Use Labmate to research ML papers, datasets, Kaggle competitions, benchmarks, docs, and code examples before implementation.
---

# ML Research

Use this skill when the user asks for evidence-backed ML implementation work,
Kaggle competition progress, model training changes, dataset inspection,
benchmark research, or paper-driven experiments.

Workflow:

1. Read `program.md` if present.
2. Identify the research questions blocking implementation.
3. Run focused Labmate tools:
   - `labmate kaggle start <competition-url-or-slug> --workspace <workspace>`
   - `labmate kaggle baseline <workspace> --run-name constant_baseline`
   - `labmate kaggle validate-submission submissions/constant_baseline.csv --workspace <workspace>`
   - `labmate project-scan <project-root>`
   - `labmate experiment-summary <project-root-or-results.tsv>`
   - `labmate research-brief <dataset-path>`
   - `labmate dataset-inspect <dataset-path>`
   - `labmate benchmark-lookup "<task or dataset>"`
   - `labmate literature-search "<method or task query>" --max-results 5`
   - `labmate docs-fetch "<framework API or concept>" --max-results 3`
   - `labmate github-find-examples "<implementation pattern>" --max-results 3`
4. For Kaggle work, use `kaggle_researcher` before editing model code, create a
   validated constant baseline when data is ready, and do not submit unless the
   user explicitly approves the exact file and message.
5. Summarize command outputs, URLs, dataset risks, metric/protocol assumptions,
   prior experiments, and the experiment-tracking plan.
6. Only recommend implementation steps that are grounded in retrieved evidence.

Suggested Codex prompt:

```text
/goal Work on Kaggle competition <slug>. Start with labmate kaggle start, spawn kaggle_researcher for the research pass, verify metric/schema/validation, and do not submit without explicit approval.
```
