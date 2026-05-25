# Program

## Goal

Describe the target result and how it will be measured.

## Scope

Allowed changes:

- ...

Forbidden changes:

- ...

## Evidence Required

Run the relevant Labmate commands before editing model code.

- Kaggle workspace:
  - Command: `labmate kaggle start <competition-url-or-slug> --workspace <workspace>`
  - Capture: competition slug, workspace paths, data/download status, agent
    handoff, submission approval policy, and next actions.
- Kaggle baseline:
  - Command: `labmate kaggle baseline <workspace> --run-name constant_baseline`
  - Capture: submission path, validation status, prediction fill values,
    manifest path, and appended ledger row.
- Submission validation:
  - Command: `labmate kaggle validate-submission <submission> --workspace <workspace>`
  - Capture: expected/actual columns, row counts, ID alignment, and errors.
- Project scan:
  - Command: `labmate project-scan <project-root>`
  - Capture: likely dataset directories, code entrypoints, dependency files,
    existing experiment ledgers, existing agent setup, and recommended next
    Labmate command.
- Existing experiments:
  - Command: `labmate experiment-summary <project-root-or-results.tsv>`
  - Capture: best run, latest run, metric direction, status counts, and ledger
    warnings.
- First-pass brief:
  - Command: `labmate research-brief <dataset-path>`
  - Capture: inferred task, dataset warnings, benchmark URLs, recommended next
    commands, prior experiments, experiment-tracking plan, and implementation
    checklist.
- Dataset schema:
  - Command: `labmate dataset-inspect <dataset-path>`
  - Capture: train/test files, target hints, shared feature columns, missingness,
    leakage warnings, sample rows.
- Benchmark/task:
  - Command: `labmate benchmark-lookup "<task or dataset>"`
  - Capture: metric, protocol, target, pitfalls, baseline suggestions, source URL.
- Papers:
  - Command: `labmate literature-search "<method or task query>" --max-results 5`
  - Optional citation context: `labmate citation-graph <paper-id>`
  - Capture: title, year, authors, source URL, relevance signals.
- Current framework docs:
  - Command: `labmate docs-fetch "<framework API or concept>" --max-results 3`
  - Optional exact page: `labmate docs-fetch "<topic>" --url <official-doc-url>`
  - Capture: official URL, version/page title, relevant snippet.
- Working implementation examples:
  - Command: `labmate github-find-examples "<implementation pattern>" --max-results 3`
  - Capture: repository URL, stars, language, license, warnings.

Summarize evidence in the implementation plan with command outputs and URLs.
For Kaggle competitions, keep submission separate from research and ask for
explicit user approval before running any submit command.

## Metric

Primary metric:

Validation command:

## Output

Write results to `results.tsv` with columns:

```text
timestamp_utc	commit	experiment	model_family	features	validation_strategy	metric	score	score_direction	status	artifacts	notes
```

## Keep Criteria

Keep an experiment only if:

- ...

## Stop Criteria

Stop when:

- ...
