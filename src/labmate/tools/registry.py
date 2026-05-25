"""Shared tool registry for CLI and MCP surfaces."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from labmate.contracts import ExitCode, JsonObject, JsonValue, ToolResponse, failure, success
from labmate.tools.datasets import DatasetInspectionError, inspect_local_dataset
from labmate.tools.literature import ArxivSearchBackend, search_literature

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
        backend_name = _as_optional_str(arguments, "backend", "semantic_scholar")
        max_results = _as_int(arguments, "max_results", 20)
        depth = _as_int(arguments, "depth", 1)
    except ValueError as exc:
        return failure(
            "citation_graph",
            code="invalid_arguments",
            message=str(exc),
            exit_code=ExitCode.USAGE_ERROR,
        )

    return failure(
        "citation_graph",
        code="backend_not_implemented",
        message=(
            "Citation graph execution is not implemented yet. "
            "Add a Semantic Scholar or OpenAlex backend before using this tool."
        ),
        exit_code=ExitCode.BACKEND_UNAVAILABLE,
        details={
            "paper_id": paper_id,
            "backend": backend_name,
            "max_results": max_results,
            "depth": depth,
        },
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
CITATION_BACKENDS = ("semantic_scholar", "openalex")

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
    ),
    ToolDefinition(
        name="benchmark_lookup",
        description="Find benchmark tasks, datasets, metrics, papers, code, and results.",
        read_only=True,
        backends=("papers_with_code", "openml", "local"),
        input_schema=_object_schema(
            {
                "query": _string_schema("Benchmark, task, metric, or dataset query."),
                "backend": _backend_schema(("papers_with_code", "openml", "local")),
                "max_results": _integer_schema(
                    "Maximum benchmark results to return.",
                    minimum=1,
                    maximum=50,
                    default=10,
                ),
            },
            required=("query",),
        ),
        handler=_not_implemented_handler("benchmark_lookup"),
    ),
    ToolDefinition(
        name="docs_fetch",
        description="Fetch current ML framework documentation and examples.",
        read_only=True,
        backends=("official_docs", "huggingface", "local"),
        input_schema=_object_schema(
            {
                "query": _string_schema("Documentation topic or API name."),
                "backend": _backend_schema(("official_docs", "huggingface", "local")),
                "url": _string_schema("Optional exact documentation URL.", min_length=1),
            },
            required=("query",),
        ),
        handler=_not_implemented_handler("docs_fetch"),
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
        handler=_not_implemented_handler("github_find_examples"),
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
