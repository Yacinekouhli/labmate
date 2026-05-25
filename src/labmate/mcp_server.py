"""Dependency-light MCP adapter boundary.

This module intentionally does not start a real MCP transport yet. The first
slice keeps MCP as a generated surface over the shared Labmate tool registry,
so Codex, Claude Code, CLI commands, and tests can converge on one contract.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from labmate.tools.registry import ToolDefinition, iter_tools

MCP_TRANSPORT_UNAVAILABLE_MESSAGE = (
    "labmate-mcp can export tool metadata, but a real MCP transport dependency "
    "has not been selected yet. Use --list-tools for the generated MCP tool list."
)


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
        },
    }


def list_mcp_tools() -> list[dict[str, Any]]:
    """Return MCP-compatible tool metadata generated from the shared registry."""

    return [tool_to_mcp_metadata(tool) for tool in iter_tools()]


def export_mcp_tools_json(*, indent: int | None = 2) -> str:
    """Serialize the generated MCP tool metadata for installers and tests."""

    return json.dumps({"tools": list_mcp_tools()}, indent=indent, sort_keys=True)


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

    raise SystemExit(MCP_TRANSPORT_UNAVAILABLE_MESSAGE)
