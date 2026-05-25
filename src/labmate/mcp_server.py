"""MCP stdio server for Labmate tools."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import jsonschema
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from labmate.contracts import ExitCode, JsonValue, failure, response_to_json
from labmate.tools.registry import ToolDefinition, get_tool, iter_tools
from labmate.tools.registry import call_tool as call_registry_tool


def iter_mcp_tool_definitions() -> Iterable[ToolDefinition]:
    """Yield registry tools exposed through the MCP server."""

    return (tool for tool in iter_tools() if tool.read_only)


def _mcp_tool_names() -> set[str]:
    return {tool.name for tool in iter_mcp_tool_definitions()}


def tool_to_mcp_metadata(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a registry tool into an MCP-compatible tool descriptor."""

    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
        "annotations": {
            "readOnlyHint": tool.read_only,
        },
        "_meta": {
            "labmate/backends": list(tool.backends),
            "labmate/risk": tool.risk,
            "labmate/usage_examples": list(tool.usage_examples),
        },
    }


def tool_to_mcp_tool(tool: ToolDefinition) -> types.Tool:
    """Convert a registry tool into the official MCP SDK tool model."""

    metadata = tool_to_mcp_metadata(tool)
    return types.Tool(
        name=cast(str, metadata["name"]),
        description=cast(str, metadata["description"]),
        inputSchema=cast(dict[str, Any], metadata["inputSchema"]),
        annotations=types.ToolAnnotations(
            **cast(dict[str, Any], metadata["annotations"]),
        ),
        _meta=cast(dict[str, Any], metadata["_meta"]),
    )


def list_mcp_tools() -> list[dict[str, Any]]:
    """Return MCP-compatible metadata generated from the shared registry."""

    return [tool_to_mcp_metadata(tool) for tool in iter_mcp_tool_definitions()]


def export_mcp_tools_json(*, indent: int | None = 2) -> str:
    """Serialize generated MCP tool metadata for installers and tests."""

    return json.dumps({"tools": list_mcp_tools()}, indent=indent, sort_keys=True)


def _validate_tool_arguments(
    tool: ToolDefinition,
    arguments: Mapping[str, JsonValue],
) -> types.CallToolResult | None:
    try:
        jsonschema.validate(instance=dict(arguments), schema=tool.input_schema)
    except jsonschema.ValidationError as exc:
        response = failure(
            tool.name,
            code="invalid_arguments",
            message=f"Invalid MCP arguments for {tool.name}: {exc.message}",
            exit_code=ExitCode.USAGE_ERROR,
            details={
                "path": list(exc.path),
                "schema_path": list(exc.schema_path),
                "validator": str(exc.validator),
            },
        )
        payload = response.to_dict()
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=response_to_json(response, indent=2),
                )
            ],
            structuredContent=payload,
            isError=True,
        )

    return None


def call_mcp_tool(name: str, arguments: Mapping[str, JsonValue]) -> types.CallToolResult:
    """Call a registry tool and return its Labmate contract through MCP."""

    if name not in _mcp_tool_names():
        response = failure(
            name,
            code="unknown_mcp_tool",
            message=f"{name!r} is not exposed by the Labmate MCP server.",
            exit_code=ExitCode.USAGE_ERROR,
        )
    else:
        tool = get_tool(name)
        validation_failure = _validate_tool_arguments(tool, arguments)
        if validation_failure is not None:
            return validation_failure
        response = call_registry_tool(name, arguments)

    payload = response.to_dict()
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=response_to_json(response, indent=2),
            )
        ],
        structuredContent=payload,
        isError=not bool(payload["ok"]),
    )


def create_mcp_server() -> Server:
    """Create the stdio MCP server without starting a transport."""

    server = Server("labmate")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [tool_to_mcp_tool(tool) for tool in iter_mcp_tool_definitions()]

    @server.call_tool(validate_input=False)
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return call_mcp_tool(name, cast(dict[str, JsonValue], arguments))

    return server


async def _run_stdio_server_async(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_stdio_server(server: Server | None = None) -> None:
    """Run the Labmate MCP server over stdio."""

    asyncio.run(_run_stdio_server_async(server or create_mcp_server()))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="labmate-mcp")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print generated MCP tool metadata without starting a transport.",
    )

    args = parser.parse_args(argv)
    if args.list_tools:
        print(export_mcp_tools_json())
        return

    run_stdio_server(create_mcp_server())
