from labmate.tools.registry import get_tool, iter_tools


def test_registry_contains_initial_tools() -> None:
    tools = list(iter_tools())

    assert tools
    assert all(tool.risk in {"read_only", "mutating"} for tool in tools)
    assert all(tool.input_schema["type"] == "object" for tool in tools)
    assert all(tool.usage_examples for tool in tools)
    assert all(
        {"cli", "mcp"} <= {str(example["surface"]) for example in tool.usage_examples}
        for tool in tools
    )
    assert all(callable(tool.handler) for tool in tools)
    assert {tool.name for tool in tools} >= {
        "literature_search",
        "citation_graph",
        "dataset_inspect",
        "project_scan",
        "experiment_summary",
        "research_brief",
        "benchmark_lookup",
        "docs_fetch",
        "github_find_examples",
        "kaggle_start",
    }

    kaggle_start = get_tool("kaggle_start")
    assert kaggle_start.read_only is False
    assert kaggle_start.risk == "mutating"


def test_get_tool_returns_named_tool() -> None:
    tool = get_tool("dataset_inspect")

    assert tool.name == "dataset_inspect"
    assert "huggingface" in tool.backends
