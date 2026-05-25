"""Labmate command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from labmate.contracts import ToolError, ToolFailure, ToolResponse, response_to_json, success
from labmate.init import apply_init, plan_init
from labmate.tools.registry import call_tool, iter_tools


def _print_response(response: ToolResponse) -> int:
    print(response_to_json(response, indent=2))
    return int(response.exit_code)


def _tools_response() -> ToolResponse:
    return success(
        "tools",
        {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "read_only": tool.read_only,
                    "risk": tool.risk,
                    "backends": list(tool.backends),
                    "input_schema": tool.input_schema,
                    "usage_examples": list(tool.usage_examples),
                }
                for tool in iter_tools()
            ]
        },
    )


def _plan_to_result(plan, *, applied=None):
    result = {
        "harness": plan.harness,
        "project_root": str(plan.project_root),
        "goal_prompt": plan.goal_prompt,
        "follow_up_commands": list(plan.follow_up_commands),
        "notes": list(plan.notes),
        "files": [
            {
                "source": str(file.source),
                "destination": str(file.destination),
                "relative_destination": file.relative_destination,
                "action": file.action,
                "reason": file.reason,
            }
            for file in plan.files
        ],
    }
    if applied is not None:
        result["applied"] = {
            "written": [str(path) for path in applied.written],
            "skipped": [str(path) for path in applied.skipped],
        }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labmate")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("tools", help="List registered tool definitions.")

    dataset = subparsers.add_parser("dataset-inspect", help="Inspect a local dataset.")
    dataset.add_argument("path", help="CSV/TSV file or directory to inspect.")
    dataset.add_argument("--backend", default="local", help="Dataset backend. Defaults to local.")
    dataset.add_argument("--sample-size", type=int, default=5)
    dataset.add_argument("--max-profile-rows", type=int, default=250_000)

    project = subparsers.add_parser(
        "project-scan",
        help="Scan a local ML project for datasets and entrypoints.",
    )
    project.add_argument("path", help="Project directory to scan.")
    project.add_argument(
        "--backend", default="local", help="Project scan backend. Defaults to local."
    )
    project.add_argument("--max-depth", type=int, default=4)
    project.add_argument("--max-entries", type=int, default=500)

    experiment = subparsers.add_parser(
        "experiment-summary",
        help="Summarize a local experiment ledger.",
    )
    experiment.add_argument("path", help="Experiment ledger file or project directory.")
    experiment.add_argument(
        "--backend", default="local", help="Experiment summary backend. Defaults to local."
    )
    experiment.add_argument("--max-rows", type=int, default=1_000)

    research = subparsers.add_parser(
        "research-brief",
        help="Create a first-pass ML research brief from a local dataset.",
    )
    research.add_argument("path", help="CSV/TSV file or directory to inspect.")
    research.add_argument(
        "--backend", default="local", help="Research brief backend. Defaults to local."
    )
    research.add_argument("--task-hint", help="Optional caller-supplied task description.")
    research.add_argument("--benchmark-query", help="Optional benchmark lookup query override.")
    research.add_argument("--sample-size", type=int, default=3)
    research.add_argument("--max-profile-rows", type=int, default=250_000)
    research.add_argument("--max-benchmarks", type=int, default=3)

    literature = subparsers.add_parser("literature-search", help="Search ML literature.")
    literature.add_argument("query", help="Paper search query.")
    literature.add_argument(
        "--backend", default="arxiv", help="Literature backend. Defaults to arXiv."
    )
    literature.add_argument("--max-results", type=int, default=10)
    literature.add_argument("--since-year", type=int)

    citation = subparsers.add_parser("citation-graph", help="Inspect paper citations.")
    citation.add_argument("paper_id", help="Paper identifier.")
    citation.add_argument("--backend", default="local")
    citation.add_argument("--max-results", type=int, default=20)
    citation.add_argument("--depth", type=int, default=1)

    benchmark = subparsers.add_parser("benchmark-lookup", help="Find ML benchmark context.")
    benchmark.add_argument("query", help="Benchmark, task, metric, or dataset query.")
    benchmark.add_argument(
        "--backend",
        default="local",
        help="Benchmark backend. Defaults to the local catalog.",
    )
    benchmark.add_argument("--max-results", type=int, default=10)

    docs = subparsers.add_parser("docs-fetch", help="Fetch ML framework documentation.")
    docs.add_argument("query", help="Documentation topic or API name.")
    docs.add_argument(
        "--backend",
        default="official_docs",
        choices=["official_docs", "huggingface", "local"],
        help="Docs backend. Defaults to official docs.",
    )
    docs.add_argument("--url", help="Exact documentation URL to fetch.")
    docs.add_argument("--max-results", type=int, default=5)

    github = subparsers.add_parser(
        "github-find-examples",
        help="Find implementation examples in GitHub repositories.",
    )
    github.add_argument("query", help="Implementation pattern or API to search for.")
    github.add_argument("--repository", help="Optional owner/repo filter.")
    github.add_argument("--max-results", type=int, default=10)

    kaggle = subparsers.add_parser("kaggle", help="Kaggle competition workflows.")
    kaggle_subparsers = kaggle.add_subparsers(dest="kaggle_command")
    kaggle_start = kaggle_subparsers.add_parser(
        "start",
        help="Create or update a Kaggle competition workspace.",
    )
    kaggle_start.add_argument("competition", help="Kaggle competition URL or slug.")
    kaggle_start.add_argument(
        "--workspace",
        help="Local competition workspace. Defaults to the competition slug.",
    )
    kaggle_start.add_argument(
        "--data-dir",
        default="data",
        help="Workspace-relative data directory. Defaults to data/.",
    )
    kaggle_start.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=True,
        help="Try Kaggle CLI download when available. This is the default.",
    )
    kaggle_start.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Skip Kaggle CLI download and inspect existing local data only.",
    )
    kaggle_start.add_argument(
        "--force-download",
        action="store_true",
        help="Pass --force to Kaggle CLI download.",
    )
    kaggle_start.add_argument("--sample-size", type=int, default=3)
    kaggle_start.add_argument("--max-profile-rows", type=int, default=250_000)

    kaggle_baseline = kaggle_subparsers.add_parser(
        "baseline",
        help="Create and log a constant Kaggle baseline submission.",
    )
    kaggle_baseline.add_argument("workspace", help="Local Kaggle workspace.")
    kaggle_baseline.add_argument("--competition", help="Optional Kaggle competition slug.")
    kaggle_baseline.add_argument(
        "--data-dir",
        default="data",
        help="Workspace-relative data directory. Defaults to data/.",
    )
    kaggle_baseline.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "sample_default", "target_mean", "target_mode", "zero"],
        help="Constant prediction strategy.",
    )
    kaggle_baseline.add_argument("--run-name", help="Optional run/artifact name.")
    kaggle_baseline.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing submission artifact with the same run name.",
    )
    kaggle_baseline.add_argument("--sample-size", type=int, default=3)
    kaggle_baseline.add_argument("--max-profile-rows", type=int, default=250_000)

    kaggle_validate = kaggle_subparsers.add_parser(
        "validate-submission",
        help="Validate a submission against the local sample submission.",
    )
    kaggle_validate.add_argument("submission", help="Candidate submission path.")
    kaggle_validate.add_argument("--workspace", required=True, help="Local Kaggle workspace.")
    kaggle_validate.add_argument(
        "--data-dir",
        default="data",
        help="Workspace-relative data directory. Defaults to data/.",
    )

    init = subparsers.add_parser("init", help="Plan or apply agent-harness setup.")
    init.add_argument(
        "harness",
        choices=["codex", "claude", "claude-code", "generic", "mcp", "cursor"],
    )
    init.add_argument("project_root")
    init.add_argument("--apply", action="store_true", help="Write planned files.")
    init.add_argument(
        "--overwrite", action="store_true", help="Plan overwrites for existing files."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "tools":
        return _print_response(_tools_response())

    if args.command == "dataset-inspect":
        return _print_response(
            call_tool(
                "dataset_inspect",
                {
                    "path": args.path,
                    "backend": args.backend,
                    "sample_size": args.sample_size,
                    "max_profile_rows": args.max_profile_rows,
                },
            )
        )

    if args.command == "project-scan":
        return _print_response(
            call_tool(
                "project_scan",
                {
                    "path": args.path,
                    "backend": args.backend,
                    "max_depth": args.max_depth,
                    "max_entries": args.max_entries,
                },
            )
        )

    if args.command == "experiment-summary":
        return _print_response(
            call_tool(
                "experiment_summary",
                {
                    "path": args.path,
                    "backend": args.backend,
                    "max_rows": args.max_rows,
                },
            )
        )

    if args.command == "research-brief":
        payload = {
            "path": args.path,
            "backend": args.backend,
            "sample_size": args.sample_size,
            "max_profile_rows": args.max_profile_rows,
            "max_benchmarks": args.max_benchmarks,
        }
        if args.task_hint is not None:
            payload["task_hint"] = args.task_hint
        if args.benchmark_query is not None:
            payload["benchmark_query"] = args.benchmark_query
        return _print_response(call_tool("research_brief", payload))

    if args.command == "literature-search":
        payload = {
            "query": args.query,
            "backend": args.backend,
            "max_results": args.max_results,
        }
        if args.since_year is not None:
            payload["since_year"] = args.since_year
        return _print_response(call_tool("literature_search", payload))

    if args.command == "citation-graph":
        return _print_response(
            call_tool(
                "citation_graph",
                {
                    "paper_id": args.paper_id,
                    "backend": args.backend,
                    "max_results": args.max_results,
                    "depth": args.depth,
                },
            )
        )

    if args.command == "benchmark-lookup":
        return _print_response(
            call_tool(
                "benchmark_lookup",
                {
                    "query": args.query,
                    "backend": args.backend,
                    "max_results": args.max_results,
                },
            )
        )

    if args.command == "docs-fetch":
        payload = {
            "query": args.query,
            "backend": args.backend,
            "max_results": args.max_results,
        }
        if args.url is not None:
            payload["url"] = args.url
        return _print_response(call_tool("docs_fetch", payload))

    if args.command == "github-find-examples":
        payload = {
            "query": args.query,
            "backend": "github",
            "max_results": args.max_results,
        }
        if args.repository is not None:
            payload["repository"] = args.repository
        return _print_response(call_tool("github_find_examples", payload))

    if args.command == "kaggle" and args.kaggle_command == "start":
        payload = {
            "competition": args.competition,
            "backend": "local",
            "data_dir": args.data_dir,
            "download": args.download,
            "force_download": args.force_download,
            "sample_size": args.sample_size,
            "max_profile_rows": args.max_profile_rows,
        }
        if args.workspace is not None:
            payload["workspace"] = args.workspace
        return _print_response(call_tool("kaggle_start", payload))

    if args.command == "kaggle" and args.kaggle_command == "baseline":
        payload = {
            "workspace": args.workspace,
            "backend": "local",
            "data_dir": args.data_dir,
            "strategy": args.strategy,
            "overwrite": args.overwrite,
            "sample_size": args.sample_size,
            "max_profile_rows": args.max_profile_rows,
        }
        if args.competition is not None:
            payload["competition"] = args.competition
        if args.run_name is not None:
            payload["run_name"] = args.run_name
        return _print_response(call_tool("kaggle_baseline", payload))

    if args.command == "kaggle" and args.kaggle_command == "validate-submission":
        return _print_response(
            call_tool(
                "kaggle_validate_submission",
                {
                    "submission": args.submission,
                    "workspace": args.workspace,
                    "backend": "local",
                    "data_dir": args.data_dir,
                },
            )
        )

    if args.command == "init":
        try:
            plan = plan_init(args.harness, args.project_root, overwrite=args.overwrite)
            applied = apply_init(plan) if args.apply else None
        except Exception as exc:
            return _print_response(
                ToolFailure(
                    tool="init",
                    error=ToolError(
                        code="init_failed",
                        message=str(exc),
                    ),
                )
            )
        return _print_response(success("init", _plan_to_result(plan, applied=applied)))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
