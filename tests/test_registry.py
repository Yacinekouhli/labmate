from labmate.tools.registry import get_tool, iter_tools


def test_registry_contains_initial_read_only_tools() -> None:
    tools = list(iter_tools())

    assert tools
    assert all(tool.read_only for tool in tools)
    assert all(tool.input_schema["type"] == "object" for tool in tools)
    assert all(callable(tool.handler) for tool in tools)
    assert {tool.name for tool in tools} >= {
        "literature_search",
        "citation_graph",
        "dataset_inspect",
        "benchmark_lookup",
        "docs_fetch",
        "github_find_examples",
    }


def test_get_tool_returns_named_tool() -> None:
    tool = get_tool("dataset_inspect")

    assert tool.name == "dataset_inspect"
    assert "huggingface" in tool.backends
