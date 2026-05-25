"""Shared tool registry for CLI and MCP surfaces."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from labmate.contracts import ExitCode, JsonObject, JsonValue, ToolResponse, failure, success
from labmate.tools.benchmarks import LocalBenchmarkBackend, lookup_benchmarks
from labmate.tools.datasets import DatasetInspectionError, inspect_local_dataset
from labmate.tools.docs import OfficialDocsBackend, fetch_docs
from labmate.tools.github import GitHubRepositorySearchBackend, find_github_examples
from labmate.tools.literature import (
    ArxivSearchBackend,
    citation_graph,
    default_local_corpus_backend,
    search_literature,
)
from labmate.tools.workflows import build_research_brief

ToolRisk = Literal["read_only", "mutating"]
ToolHandler = Callable[[Mapping[str, JsonValue]], ToolResponse]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    read_only: bool
    backends: tuple[str, ...]
    input_schema: JsonObject
    handler: ToolHandler
    usage_examples: tuple[JsonObject, ...] = ()
    risk: ToolRisk = "read_only"


def _string_schema(description: str, *, min_length: int = 1) -> JsonObject:
    return {
        "type": "string",
        "minLength": min_length,
        "description": description,
    }


def _integer_schema(
    description: str,
    *,
    minimum: int,
    maximum: int | None = None,
    default: int | None = None,
) -> JsonObject:
    schema: JsonObject = {
        "type": "integer",
        "minimum": minimum,
        "description": description,
    }
    if maximum is not None:
        schema["maximum"] = maximum
    if default is not None:
        schema["default"] = default
    return schema


def _object_schema(
    properties: Mapping[str, JsonObject],
    *,
    required: tuple[str, ...],
) -> JsonObject:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _backend_schema(backends: tuple[str, ...]) -> JsonObject:
    return {
        "type": "string",
        "enum": list(backends),
        "description": "Backend to use for this tool.",
    }


def _cli_example(command: str, description: str) -> JsonObject:
    return {
        "surface": "cli",
        "description": description,
        "command": command,
    }


def _mcp_example(arguments: Mapping[str, JsonValue], description: str) -> JsonObject:
    return {
        "surface": "mcp",
        "description": description,
        "arguments": dict(arguments),
    }


def _as_str(arguments: Mapping[str, JsonValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _as_optional_str(arguments: Mapping[str, JsonValue], name: str, default: str) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _as_int(arguments: Mapping[str, JsonValue], name: str, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_optional_int(arguments: Mapping[str, JsonValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_maybe_str(arguments: Mapping[str, JsonValue], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _dataset_inspect_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    try:
        path = _as_str(arguments, "path")
        backend = _as_optional_str(arguments, "backend", "local")
        sample_size = _as_int(arguments, "sample_size", 5)
        max_profile_rows = _as_int(arguments, "max_profile_rows", 250_000)
        if backend != "local":
            return failure(
                "dataset_inspect",
                code="backend_not_implemented",
                message=f"Dataset backend {backend!r} is not implemented yet.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={"backend": backend},
            )
        result = inspect_local_dataset(
            path,
            sample_size=sample_size,
            max_profile_rows=max_profile_rows,
        )
    except DatasetInspectionError as exc:
        return failure(
            "dataset_inspect",
            code="dataset_inspection_error",
            message=str(exc),
            exit_code=ExitCode.TOOL_ERROR,
        )
    except ValueError as exc:
        return failure(
            "dataset_inspect",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )

    return success(
        "dataset_inspect",
        result,
        metadata={"backend": "local"},
    )


def _literature_search_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    try:
        query = _as_str(arguments, "query")
        backend_name = _as_optional_str(arguments, "backend", "arxiv")
        max_results = _as_int(arguments, "max_results", 10)
        since_year = _as_optional_int(arguments, "since_year")
        if backend_name != "arxiv":
            return failure(
                "literature_search",
                code="backend_not_implemented",
                message=f"Literature backend {backend_name!r} is not implemented yet.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={"backend": backend_name},
            )
        result = search_literature(
            query,
            backend=ArxivSearchBackend(),
            max_results=max_results,
            since_year=since_year,
        )
    except ValueError as exc:
        return failure(
            "literature_search",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )
    except OSError as exc:
        return failure(
            "literature_search",
            code="backend_unavailable",
            message=str(exc),
            exit_code=ExitCode.BACKEND_UNAVAILABLE,
            retryable=True,
            details={"backend": "arxiv"},
        )

    return success(
        "literature_search",
        result.to_dict(),
        metadata={"backend": backend_name},
    )


def _citation_graph_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    try:
        paper_id = _as_str(arguments, "paper_id")
        backend_name = _as_optional_str(arguments, "backend", "local")
        max_results = _as_int(arguments, "max_results", 20)
        depth = _as_int(arguments, "depth", 1)
        if backend_name != "local":
            return failure(
                "citation_graph",
                code="backend_not_implemented",
                message=f"Citation backend {backend_name!r} is not implemented yet.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={
                    "paper_id": paper_id,
                    "backend": backend_name,
                    "max_results": max_results,
                    "depth": depth,
                },
            )

        result = citation_graph(
            paper_id,
            backend=default_local_corpus_backend(),
            max_results=max_results,
            depth=depth,
        )
    except ValueError as exc:
        return failure(
            "citation_graph",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )
    except KeyError as exc:
        return failure(
            "citation_graph",
            code="paper_not_found",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
            details={"paper_id": paper_id, "backend": backend_name},
        )

    return success(
        "citation_graph",
        result.to_dict(),
        metadata={"backend": backend_name},
    )


def _benchmark_lookup_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    try:
        query = _as_str(arguments, "query")
        backend_name = _as_optional_str(arguments, "backend", "local")
        max_results = _as_int(arguments, "max_results", 10)
        if backend_name != "local":
            return failure(
                "benchmark_lookup",
                code="backend_not_implemented",
                message=f"Benchmark backend {backend_name!r} is not implemented yet.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={"backend": backend_name},
            )

        result = lookup_benchmarks(
            query,
            backend=LocalBenchmarkBackend(),
            max_results=max_results,
        )
    except ValueError as exc:
        return failure(
            "benchmark_lookup",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )

    return success(
        "benchmark_lookup",
        result.to_dict(),
        metadata={"backend": backend_name},
    )


def _docs_fetch_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    backend_name = "official_docs"
    try:
        query = _as_str(arguments, "query")
        backend_name = _as_optional_str(arguments, "backend", "official_docs")
        url = _as_maybe_str(arguments, "url")
        max_results = _as_int(arguments, "max_results", 5)
        if backend_name not in DOCS_BACKENDS:
            return failure(
                "docs_fetch",
                code="backend_not_implemented",
                message=f"Docs backend {backend_name!r} is not implemented.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={"backend": backend_name},
            )
        if backend_name == "local" and url is not None:
            return failure(
                "docs_fetch",
                code="invalid_arguments",
                message="The local docs backend searches the built-in catalog only; omit url.",
                exit_code=ExitCode.USAGE_ERROR,
                details={"backend": backend_name},
            )

        result = fetch_docs(
            query,
            backend=OfficialDocsBackend(name=backend_name),
            url=url,
            max_results=max_results,
        )
    except ValueError as exc:
        return failure(
            "docs_fetch",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )
    except OSError as exc:
        return failure(
            "docs_fetch",
            code="backend_unavailable",
            message=str(exc),
            exit_code=ExitCode.BACKEND_UNAVAILABLE,
            retryable=True,
            details={"backend": backend_name},
        )

    return success(
        "docs_fetch",
        result.to_dict(),
        metadata={"backend": backend_name},
    )


def _github_find_examples_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    try:
        query = _as_str(arguments, "query")
        backend_name = _as_optional_str(arguments, "backend", "github")
        repository = _as_maybe_str(arguments, "repository")
        max_results = _as_int(arguments, "max_results", 10)
        if backend_name != "github":
            return failure(
                "github_find_examples",
                code="backend_not_implemented",
                message=f"GitHub examples backend {backend_name!r} is not implemented.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={"backend": backend_name},
            )

        result = find_github_examples(
            query,
            backend=GitHubRepositorySearchBackend(),
            repository=repository,
            max_results=max_results,
        )
    except ValueError as exc:
        return failure(
            "github_find_examples",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )
    except OSError as exc:
        return failure(
            "github_find_examples",
            code="backend_unavailable",
            message=str(exc),
            exit_code=ExitCode.BACKEND_UNAVAILABLE,
            retryable=True,
            details={"backend": "github"},
        )

    return success(
        "github_find_examples",
        result.to_dict(),
        metadata={"backend": "github"},
    )


def _research_brief_handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
    try:
        path = _as_str(arguments, "path")
        backend_name = _as_optional_str(arguments, "backend", "local")
        task_hint = _as_maybe_str(arguments, "task_hint")
        benchmark_query = _as_maybe_str(arguments, "benchmark_query")
        sample_size = _as_int(arguments, "sample_size", 3)
        max_profile_rows = _as_int(arguments, "max_profile_rows", 250_000)
        max_benchmarks = _as_int(arguments, "max_benchmarks", 3)
        if backend_name != "local":
            return failure(
                "research_brief",
                code="backend_not_implemented",
                message=f"Research brief backend {backend_name!r} is not implemented yet.",
                exit_code=ExitCode.BACKEND_UNAVAILABLE,
                details={"backend": backend_name},
            )

        result = build_research_brief(
            path,
            task_hint=task_hint,
            benchmark_query=benchmark_query,
            sample_size=sample_size,
            max_profile_rows=max_profile_rows,
            max_benchmarks=max_benchmarks,
        )
    except DatasetInspectionError as exc:
        return failure(
            "research_brief",
            code="dataset_inspection_error",
            message=str(exc),
            exit_code=ExitCode.TOOL_ERROR,
        )
    except ValueError as exc:
        return failure(
            "research_brief",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )

    return success(
        "research_brief",
        result,
        metadata={"backend": "local"},
    )


def _not_implemented_handler(tool_name: str) -> ToolHandler:
    def handler(arguments: Mapping[str, JsonValue]) -> ToolResponse:
        backend = arguments.get("backend")
        details: JsonObject = {}
        if isinstance(backend, str):
            details["backend"] = backend
        return failure(
            tool_name,
            code="tool_not_implemented",
            message=f"{tool_name} is registered but its handler is not implemented yet.",
            exit_code=ExitCode.BACKEND_UNAVAILABLE,
            details=details,
        )

    return handler


DATASET_BACKENDS = ("local", "huggingface", "kaggle", "openml", "uci")
LITERATURE_BACKENDS = ("arxiv", "semantic_scholar", "openalex", "core")
CITATION_BACKENDS = ("local", "semantic_scholar", "openalex")
DOCS_BACKENDS = ("official_docs", "huggingface", "local")
BENCHMARK_BACKENDS = ("local", "papers_with_code", "openml")
RESEARCH_BRIEF_BACKENDS = ("local",)

_TOOLS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="literature_search",
        description="Search for ML papers with source attribution.",
        read_only=True,
        backends=LITERATURE_BACKENDS,
        input_schema=_object_schema(
            {
                "query": _string_schema("Paper search query."),
                "backend": _backend_schema(LITERATURE_BACKENDS),
                "max_results": _integer_schema(
                    "Maximum number of papers to return.",
                    minimum=1,
                    maximum=50,
                    default=10,
                ),
                "since_year": _integer_schema(
                    "Optional minimum publication year.",
                    minimum=1900,
                ),
            },
            required=("query",),
        ),
        handler=_literature_search_handler,
        usage_examples=(
            _cli_example(
                'labmate literature-search "tabular classification baseline" --max-results 5',
                "Find papers relevant to a modeling approach or task.",
            ),
            _mcp_example(
                {"query": "tabular classification baseline", "max_results": 5},
                "Search for papers from an MCP client.",
            ),
        ),
    ),
    ToolDefinition(
        name="citation_graph",
        description="Inspect citations, references, and related work for a paper.",
        read_only=True,
        backends=CITATION_BACKENDS,
        input_schema=_object_schema(
            {
                "paper_id": _string_schema("Paper identifier, such as DOI, arXiv ID, or S2 ID."),
                "backend": _backend_schema(CITATION_BACKENDS),
                "max_results": _integer_schema(
                    "Maximum references or citations to return.",
                    minimum=1,
                    maximum=100,
                    default=20,
                ),
                "depth": _integer_schema(
                    "Citation traversal depth. The initial implementation supports depth 1.",
                    minimum=1,
                    maximum=2,
                    default=1,
                ),
            },
            required=("paper_id",),
        ),
        handler=_citation_graph_handler,
        usage_examples=(
            _cli_example(
                "labmate citation-graph arxiv:1603.02754 --max-results 3",
                "Inspect local citation context for a known paper.",
            ),
            _mcp_example(
                {"paper_id": "arxiv:1603.02754", "max_results": 3},
                "Fetch citation context from an MCP client.",
            ),
        ),
    ),
    ToolDefinition(
        name="dataset_inspect",
        description="Inspect dataset schema, splits, sample rows, licenses, and task fit.",
        read_only=True,
        backends=DATASET_BACKENDS,
        input_schema=_object_schema(
            {
                "path": _string_schema("Local CSV/TSV file or dataset directory to inspect."),
                "backend": _backend_schema(DATASET_BACKENDS),
                "sample_size": _integer_schema(
                    "Number of sample rows to include per file.",
                    minimum=0,
                    maximum=100,
                    default=5,
                ),
                "max_profile_rows": _integer_schema(
                    "Maximum rows to profile per tabular file.",
                    minimum=1,
                    default=250_000,
                ),
            },
            required=("path",),
        ),
        handler=_dataset_inspect_handler,
        usage_examples=(
            _cli_example(
                "labmate dataset-inspect data/ --sample-size 5",
                "Inspect a local Kaggle-style data directory.",
            ),
            _mcp_example(
                {"path": "data/", "sample_size": 5},
                "Inspect a dataset through MCP.",
            ),
        ),
    ),
    ToolDefinition(
        name="research_brief",
        description=(
            "Create a first-pass ML research brief from local dataset and benchmark context."
        ),
        read_only=True,
        backends=RESEARCH_BRIEF_BACKENDS,
        input_schema=_object_schema(
            {
                "path": _string_schema("Local CSV/TSV file or dataset directory to inspect."),
                "backend": _backend_schema(RESEARCH_BRIEF_BACKENDS),
                "task_hint": _string_schema("Optional caller-supplied task description."),
                "benchmark_query": _string_schema("Optional benchmark lookup query override."),
                "sample_size": _integer_schema(
                    "Number of sample rows to inspect per file.",
                    minimum=0,
                    maximum=50,
                    default=3,
                ),
                "max_profile_rows": _integer_schema(
                    "Maximum rows to profile per tabular file.",
                    minimum=1,
                    default=250_000,
                ),
                "max_benchmarks": _integer_schema(
                    "Maximum local benchmark references to include.",
                    minimum=1,
                    maximum=10,
                    default=3,
                ),
            },
            required=("path",),
        ),
        handler=_research_brief_handler,
        usage_examples=(
            _cli_example(
                "labmate research-brief data/ --max-benchmarks 3",
                "Create a first-pass brief before editing model code.",
            ),
            _mcp_example(
                {"path": "data/", "max_benchmarks": 3},
                "Build a research brief through MCP.",
            ),
        ),
    ),
    ToolDefinition(
        name="benchmark_lookup",
        description="Find benchmark tasks, datasets, metrics, papers, code, and results.",
        read_only=True,
        backends=BENCHMARK_BACKENDS,
        input_schema=_object_schema(
            {
                "query": _string_schema("Benchmark, task, metric, or dataset query."),
                "backend": _backend_schema(BENCHMARK_BACKENDS),
                "max_results": _integer_schema(
                    "Maximum benchmark results to return.",
                    minimum=1,
                    maximum=50,
                    default=10,
                ),
            },
            required=("query",),
        ),
        handler=_benchmark_lookup_handler,
        usage_examples=(
            _cli_example(
                'labmate benchmark-lookup "tabular classification auc" --max-results 3',
                "Find metric, protocol, and baseline context for a task.",
            ),
            _mcp_example(
                {"query": "tabular classification auc", "max_results": 3},
                "Look up benchmark context through MCP.",
            ),
        ),
    ),
    ToolDefinition(
        name="docs_fetch",
        description="Fetch current ML framework documentation and examples.",
        read_only=True,
        backends=DOCS_BACKENDS,
        input_schema=_object_schema(
            {
                "query": _string_schema("Documentation topic or API name."),
                "backend": _backend_schema(DOCS_BACKENDS),
                "url": _string_schema("Optional exact documentation URL.", min_length=1),
                "max_results": _integer_schema(
                    "Maximum documentation references to return.",
                    minimum=1,
                    maximum=20,
                    default=5,
                ),
            },
            required=("query",),
        ),
        handler=_docs_fetch_handler,
        usage_examples=(
            _cli_example(
                'labmate docs-fetch "sklearn ColumnTransformer pipeline" --max-results 3',
                "Find official framework documentation for an implementation detail.",
            ),
            _mcp_example(
                {"query": "sklearn ColumnTransformer pipeline", "max_results": 3},
                "Fetch framework docs through MCP.",
            ),
        ),
    ),
    ToolDefinition(
        name="github_find_examples",
        description="Find implementation examples in GitHub repositories.",
        read_only=True,
        backends=("github",),
        input_schema=_object_schema(
            {
                "query": _string_schema("Implementation pattern or API to search for."),
                "repository": _string_schema("Optional owner/repo filter.", min_length=1),
                "max_results": _integer_schema(
                    "Maximum examples to return.",
                    minimum=1,
                    maximum=50,
                    default=10,
                ),
            },
            required=("query",),
        ),
        handler=_github_find_examples_handler,
        usage_examples=(
            _cli_example(
                'labmate github-find-examples "sklearn pipeline kaggle" --max-results 3',
                "Find public repository examples for an implementation pattern.",
            ),
            _mcp_example(
                {"query": "sklearn pipeline kaggle", "max_results": 3},
                "Find implementation examples through MCP.",
            ),
        ),
    ),
)


def iter_tools() -> Iterable[ToolDefinition]:
    return iter(_TOOLS)


def get_tool(name: str) -> ToolDefinition:
    for tool in _TOOLS:
        if tool.name == name:
            return tool
    raise KeyError(name)


def call_tool(name: str, arguments: Mapping[str, JsonValue]) -> ToolResponse:
    return get_tool(name).handler(arguments)
