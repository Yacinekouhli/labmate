---
description: Use Labmate to research ML papers, datasets, benchmarks, docs, and code examples before implementation.
---

# ML Research

Use Labmate to verify research evidence before ML implementation work.

1. Read `program.md` if present.
2. Use the `ml-researcher` agent for read-only evidence gathering.
3. Run focused Labmate MCP or CLI tools:
   - `labmate research-brief <dataset-path>`
   - `labmate dataset-inspect <dataset-path>`
   - `labmate benchmark-lookup "<task or dataset>"`
   - `labmate literature-search "<method or task query>" --max-results 5`
   - `labmate docs-fetch "<framework API or concept>" --max-results 3`
   - `labmate github-find-examples "<implementation pattern>" --max-results 3`
4. Return command outputs, URLs, risks, and implementation recommendations.
