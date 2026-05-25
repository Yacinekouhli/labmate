import asyncio
import json

from mcp import types

import labmate.mcp_server as mcp_server
from labmate.mcp_server import (
    call_mcp_tool,
    create_mcp_server,
    export_mcp_tools_json,
    iter_mcp_tool_definitions,
    list_mcp_tools,
    tool_to_mcp_metadata,
    tool_to_mcp_tool,
)
from labmate.tools.registry import get_tool, iter_tools


def test_mcp_tools_are_generated_from_registry() -> None:
    registry_tools = list(iter_tools())
    mcp_tools = list_mcp_tools()

    assert [tool["name"] for tool in mcp_tools] == [tool.name for tool in registry_tools]
    assert mcp_tools == [tool_to_mcp_metadata(tool) for tool in registry_tools]


def test_mcp_tool_exposure_is_read_only_registry_subset() -> None:
    assert [tool.name for tool in iter_mcp_tool_definitions()] == [
        tool.name for tool in iter_tools() if tool.read_only
    ]


def test_mcp_metadata_preserves_registry_tool_attributes() -> None:
    registry_tool = get_tool("dataset_inspect")
    mcp_tool = tool_to_mcp_metadata(registry_tool)

    assert mcp_tool["name"] == "dataset_inspect"
    assert mcp_tool["description"] == registry_tool.description
    assert mcp_tool["inputSchema"] == registry_tool.input_schema
    assert mcp_tool["annotations"] == {"readOnlyHint": True}
    assert mcp_tool["_meta"]["labmate/backends"] == list(registry_tool.backends)
    assert mcp_tool["_meta"]["labmate/risk"] == registry_tool.risk


def test_mcp_tool_model_uses_registry_input_schema() -> None:
    registry_tool = get_tool("dataset_inspect")
    mcp_tool = tool_to_mcp_tool(registry_tool)

    assert mcp_tool.name == "dataset_inspect"
    assert mcp_tool.description == registry_tool.description
    assert mcp_tool.inputSchema == registry_tool.input_schema
    assert mcp_tool.annotations is not None
    assert mcp_tool.annotations.readOnlyHint is True
    assert mcp_tool.meta == {
        "labmate/backends": list(registry_tool.backends),
        "labmate/risk": registry_tool.risk,
    }


def test_export_mcp_tools_json_returns_tools_object() -> None:
    payload = json.loads(export_mcp_tools_json(indent=None))

    assert payload == {"tools": list_mcp_tools()}


def test_mcp_entrypoint_can_print_metadata(capsys) -> None:
    mcp_server.main(["--list-tools"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"tools": list_mcp_tools()}


def test_dataset_inspect_can_be_called_through_mcp_helper(tmp_path) -> None:
    dataset = tmp_path / "train.csv"
    dataset.write_text("id,target\n1,0\n2,1\n", encoding="utf-8")

    result = call_mcp_tool("dataset_inspect", {"path": str(dataset), "sample_size": 1})

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["schema_version"] == "labmate.tool.v1"
    assert result.structuredContent["ok"] is True
    assert result.structuredContent["tool"] == "dataset_inspect"
    assert result.structuredContent["result"]["file_name"] == "train.csv"
    assert result.structuredContent["result"]["sample_rows"] == [{"id": "1", "target": "0"}]

    text_payload = json.loads(result.content[0].text)
    assert text_payload == result.structuredContent


def test_dataset_inspect_can_be_called_through_registered_server_handler(tmp_path) -> None:
    (tmp_path / "train.csv").write_text("id,feature,target\n1,10,0\n2,11,1\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,feature\n3,12\n4,13\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,target\n3,0\n4,0\n", encoding="utf-8")

    async def call_tool() -> types.CallToolResult:
        server = create_mcp_server()
        handler = server.request_handlers[types.CallToolRequest]
        result = await handler(
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="dataset_inspect",
                    arguments={"path": str(tmp_path), "sample_size": 1},
                )
            )
        )
        return result.root

    result = asyncio.run(call_tool())

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["schema_version"] == "labmate.tool.v1"
    assert result.structuredContent["tool"] == "dataset_inspect"
    assert result.structuredContent["result"]["kind"] == "local_dataset_directory"
    assert result.structuredContent["result"]["relations"]["train_file"] == "train.csv"
    assert result.structuredContent["result"]["relations"]["test_file"] == "test.csv"
    assert (
        result.structuredContent["result"]["relations"]["sample_submission_file"]
        == "sample_submission.csv"
    )


def test_registered_server_handler_returns_contract_failure_for_invalid_input() -> None:
    async def call_tool() -> types.CallToolResult:
        server = create_mcp_server()
        handler = server.request_handlers[types.CallToolRequest]
        result = await handler(
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="dataset_inspect",
                    arguments={},
                )
            )
        )
        return result.root

    result = asyncio.run(call_tool())

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["schema_version"] == "labmate.tool.v1"
    assert result.structuredContent["ok"] is False
    assert result.structuredContent["tool"] == "dataset_inspect"
    assert result.structuredContent["error"]["code"] == "invalid_arguments"
    assert result.structuredContent["error"]["details"]["validator"] == "required"

    text_payload = json.loads(result.content[0].text)
    assert text_payload == result.structuredContent


def test_unknown_mcp_tool_returns_contract_failure() -> None:
    result = call_mcp_tool("unknown_tool", {})

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["schema_version"] == "labmate.tool.v1"
    assert result.structuredContent["ok"] is False
    assert result.structuredContent["tool"] == "unknown_tool"
    assert result.structuredContent["error"]["code"] == "unknown_mcp_tool"


def test_create_mcp_server_registers_registry_tools() -> None:
    async def list_tools() -> list[types.Tool]:
        server = create_mcp_server()
        handler = server.request_handlers[types.ListToolsRequest]
        result = await handler(types.ListToolsRequest())
        return result.root.tools

    tools = asyncio.run(list_tools())

    assert [tool.name for tool in tools] == [tool.name for tool in iter_tools()]
    assert [tool.inputSchema for tool in tools] == [tool.input_schema for tool in iter_tools()]


def test_mcp_entrypoint_starts_stdio_server(monkeypatch) -> None:
    started = {}

    def fake_run_stdio_server(server) -> None:
        started["name"] = server.name
        started["has_list_tools"] = types.ListToolsRequest in server.request_handlers
        started["has_call_tool"] = types.CallToolRequest in server.request_handlers

    monkeypatch.setattr(mcp_server, "run_stdio_server", fake_run_stdio_server)

    mcp_server.main([])

    assert started == {
        "name": "labmate",
        "has_list_tools": True,
        "has_call_tool": True,
    }
