import json

import pytest

from labmate.mcp_server import (
    MCP_TRANSPORT_UNAVAILABLE_MESSAGE,
    export_mcp_tools_json,
    list_mcp_tools,
    main,
    tool_to_mcp_metadata,
)
from labmate.tools.registry import get_tool, iter_tools


def test_mcp_tools_are_generated_from_registry() -> None:
    registry_names = {tool.name for tool in iter_tools()}
    mcp_tools = list_mcp_tools()

    assert {tool["name"] for tool in mcp_tools} == registry_names


def test_mcp_metadata_preserves_registry_tool_attributes() -> None:
    registry_tool = get_tool("dataset_inspect")
    mcp_tool = tool_to_mcp_metadata(registry_tool)

    assert mcp_tool["name"] == "dataset_inspect"
    assert mcp_tool["description"] == registry_tool.description
    assert mcp_tool["annotations"] == {"readOnlyHint": True}
    assert mcp_tool["_meta"]["labmate/backends"] == list(registry_tool.backends)


def test_mcp_metadata_exposes_tool_specific_input_schema() -> None:
    mcp_tool = tool_to_mcp_metadata(get_tool("literature_search"))
    schema = mcp_tool["inputSchema"]

    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["backend"]["enum"] == [
        "arxiv",
        "semantic_scholar",
        "openalex",
        "core",
    ]
    assert schema["properties"]["max_results"]["maximum"] == 50

    dataset_tool = tool_to_mcp_metadata(get_tool("dataset_inspect"))
    dataset_schema = dataset_tool["inputSchema"]
    assert dataset_schema["required"] == ["path"]
    assert "sample_size" in dataset_schema["properties"]


def test_export_mcp_tools_json_returns_tools_object() -> None:
    payload = json.loads(export_mcp_tools_json(indent=None))

    assert "tools" in payload
    schemas_by_name = {tool["name"]: tool["inputSchema"] for tool in payload["tools"]}
    assert schemas_by_name["literature_search"]["required"] == ["query"]
    assert schemas_by_name["dataset_inspect"]["required"] == ["path"]


def test_mcp_entrypoint_can_print_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--list-tools"])

    payload = json.loads(capsys.readouterr().out)
    assert {tool["name"] for tool in payload["tools"]} == {tool.name for tool in iter_tools()}


def test_mcp_entrypoint_refuses_to_start_transport() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert str(exc_info.value) == MCP_TRANSPORT_UNAVAILABLE_MESSAGE
