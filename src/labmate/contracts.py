"""Typed JSON contracts shared by Labmate CLI and MCP tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeAlias

SCHEMA_VERSION = "labmate.tool.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class ExitCode(IntEnum):
    """Process exit codes used by Labmate CLI tool commands."""

    OK = 0
    USAGE_ERROR = 2
    TOOL_ERROR = 10
    BACKEND_UNAVAILABLE = 11
    AUTH_REQUIRED = 12
    RATE_LIMITED = 13
    INTERNAL_ERROR = 70


@dataclass(frozen=True)
class ToolError:
    """Structured error payload for a failed tool response."""

    code: str
    message: str
    retryable: bool = False
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ToolSuccess:
    """Successful tool response with machine-readable output."""

    tool: str
    result: Mapping[str, JsonValue]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def exit_code(self) -> ExitCode:
        return ExitCode.OK

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "tool": self.tool,
            "exit_code": int(self.exit_code),
            "result": dict(self.result),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ToolFailure:
    """Failed tool response with structured error details."""

    tool: str
    error: ToolError
    exit_code: ExitCode = ExitCode.TOOL_ERROR
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "tool": self.tool,
            "exit_code": int(self.exit_code),
            "error": self.error.to_dict(),
            "metadata": dict(self.metadata),
        }


ToolResponse: TypeAlias = ToolSuccess | ToolFailure


def success(
    tool: str,
    result: Mapping[str, JsonValue],
    *,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ToolSuccess:
    return ToolSuccess(tool=tool, result=result, metadata=metadata or {})


def failure(
    tool: str,
    *,
    code: str,
    message: str,
    exit_code: ExitCode = ExitCode.TOOL_ERROR,
    retryable: bool = False,
    details: Mapping[str, JsonValue] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ToolFailure:
    return ToolFailure(
        tool=tool,
        error=ToolError(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        ),
        exit_code=exit_code,
        metadata=metadata or {},
    )


def response_to_json(response: ToolResponse, *, indent: int | None = None) -> str:
    """Serialize a response deterministically for logs, fixtures, and agents."""

    return json.dumps(response.to_dict(), indent=indent, sort_keys=True)
