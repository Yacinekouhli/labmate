---
description: Use Labmate to research ML papers, datasets, Kaggle competitions, benchmarks, docs, and code examples before implementation.
---

# ML Research

Use Labmate to verify research evidence before ML implementation work or
Kaggle competition progress.

1. Read `program.md` if present.
2. Use the `ml-researcher` agent for read-only evidence gathering, or
   `kaggle-researcher` for competition work.
3. Run focused Labmate MCP or CLI tools:
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
4. For Kaggle work, create a validated constant baseline when data is ready and
   do not submit unless the user explicitly approves the exact file and message.
5. Return command outputs, URLs, risks, prior experiments, experiment-tracking
   plan, and implementation recommendations.
