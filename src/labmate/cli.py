"""Minimal CLI for the initial Labmate scaffold."""

from __future__ import annotations

import argparse
import json

from labmate.tools.registry import iter_tools


def main() -> None:
    parser = argparse.ArgumentParser(prog="labmate")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("tools", help="List registered tool definitions.")

    args = parser.parse_args()

    if args.command == "tools":
        print(
            json.dumps(
                {
                    "ok": True,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "read_only": tool.read_only,
                            "backends": list(tool.backends),
                        }
                        for tool in iter_tools()
                    ],
                },
                indent=2,
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
