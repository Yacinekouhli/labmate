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

    return iter_tools()


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


def list_mcp_prompts() -> list[dict[str, Any]]:
    """Return MCP prompt metadata for host slash-command discovery."""

    return [
        {
            "name": "kagglethis",
            "title": "Start a Kaggle competition workflow",
            "description": (
                "Create or update a Labmate Kaggle workspace and guide the host agent "
                "through research, baseline, logging, and approval-gated submission."
            ),
            "arguments": [
                {
                    "name": "competition",
                    "description": "Kaggle competition URL or slug.",
                    "required": True,
                },
                {
                    "name": "workspace",
                    "description": "Optional local workspace path.",
                    "required": False,
                },
            ],
        }
    ]


def prompt_to_mcp_prompt(prompt: Mapping[str, Any]) -> types.Prompt:
    """Convert Labmate prompt metadata into an MCP prompt model."""

    return types.Prompt(
        name=str(prompt["name"]),
        title=str(prompt["title"]),
        description=str(prompt["description"]),
        arguments=[
            types.PromptArgument(
                name=str(argument["name"]),
                description=str(argument["description"]),
                required=bool(argument["required"]),
            )
            for argument in prompt["arguments"]
        ],
    )


def export_mcp_prompts_json(*, indent: int | None = 2) -> str:
    """Serialize generated MCP prompt metadata for installers and tests."""

    return json.dumps({"prompts": list_mcp_prompts()}, indent=indent, sort_keys=True)


def get_mcp_prompt(name: str, arguments: Mapping[str, str] | None) -> types.GetPromptResult:
    """Return a prompt body for host agents that support MCP slash commands."""

    if name != "kagglethis":
        return types.GetPromptResult(
            description=f"Unknown Labmate prompt: {name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Unknown Labmate prompt {name!r}.",
                    ),
                )
            ],
        )

    arguments = arguments or {}
    competition = str(arguments.get("competition", "")).strip()
    workspace = str(arguments.get("workspace", "")).strip()
    if not competition:
        prompt_text = "Ask the user for a Kaggle competition URL or slug, then rerun this prompt."
    else:
        workspace_clause = f" using workspace `{workspace}`" if workspace else ""
        tool_arguments = {"competition": competition}
        if workspace:
            tool_arguments["workspace"] = workspace
        tool_arguments_json = json.dumps(tool_arguments)
        prompt_header = (
            f"Start a Labmate Kaggle competition workflow for `{competition}`{workspace_clause}."
        )
        prompt_text = f"""{prompt_header}

1. Call the Labmate MCP tool `kaggle_start` with arguments:
   `{tool_arguments_json}`.
2. If the result says data is missing and Kaggle MCP tools are available, use the Kaggle MCP
   competition details and download tools. If not, explain the exact auth/tooling blocker.
3. Inspect the dataset and results ledger reported by Labmate before editing model code.
4. Use or delegate to the `kaggle-researcher` subagent for competition rules, metric,
   leakage risks, validation strategy, baseline plan, prior submissions, and next experiments.
5. Implement the smallest reproducible baseline only after target, metric, split, and submission
   format are verified.
6. Log every experiment in `results.tsv`.
7. Do not submit to Kaggle unless the user explicitly approves the exact submission file and
   message."""

    return types.GetPromptResult(
        description="Start a Kaggle competition workflow with Labmate.",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            )
        ],
    )


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

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [prompt_to_mcp_prompt(prompt) for prompt in list_mcp_prompts()]

    @server.get_prompt()
    async def handle_get_prompt(
        name: str,
        arguments: dict[str, str] | None,
    ) -> types.GetPromptResult:
        return get_mcp_prompt(name, arguments)

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
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="Print generated MCP prompt metadata without starting a transport.",
    )

    args = parser.parse_args(argv)
    if args.list_tools:
        print(export_mcp_tools_json())
        return
    if args.list_prompts:
        print(export_mcp_prompts_json())
        return

    run_stdio_server(create_mcp_server())
