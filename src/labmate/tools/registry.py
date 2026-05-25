"""Shared tool registry for CLI and MCP surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    read_only: bool
    backends: tuple[str, ...]


_TOOLS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="literature_search",
        description="Search for ML papers with source attribution.",
        read_only=True,
        backends=("arxiv", "semantic_scholar", "openalex", "core"),
    ),
    ToolDefinition(
        name="citation_graph",
        description="Inspect citations, references, and related work for a paper.",
        read_only=True,
        backends=("semantic_scholar", "openalex"),
    ),
    ToolDefinition(
        name="dataset_inspect",
        description="Inspect dataset schema, splits, sample rows, licenses, and task fit.",
        read_only=True,
        backends=("huggingface", "kaggle", "openml", "uci", "local"),
    ),
    ToolDefinition(
        name="benchmark_lookup",
        description="Find benchmark tasks, datasets, metrics, papers, code, and results.",
        read_only=True,
        backends=("papers_with_code", "openml", "local"),
    ),
    ToolDefinition(
        name="docs_fetch",
        description="Fetch current ML framework documentation and examples.",
        read_only=True,
        backends=("official_docs", "huggingface", "local"),
    ),
    ToolDefinition(
        name="github_find_examples",
        description="Find implementation examples in GitHub repositories.",
        read_only=True,
        backends=("github",),
    ),
)


def iter_tools() -> Iterable[ToolDefinition]:
    return iter(_TOOLS)


def get_tool(name: str) -> ToolDefinition:
    for tool in _TOOLS:
        if tool.name == name:
            return tool
    raise KeyError(name)
