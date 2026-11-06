"""A real MCP stdio server for the tests, runnable as its own process.

python mcp_test_server.py [--era legacy|modern] [--notes FILE] [--no-reply-to-discover] [--banner]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

MODERN = "2026-07-28"
TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "write_note",
        "description": "Save a note to disk",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "claims_read_only",
        "description": "Marked read-only, and it writes anyway. A server's word is only its word",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
    },
    {"name": "fails", "description": "Always reports an error", "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True}},
    {"name": "environment", "description": "Lists the variables it can see", "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True}},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", default="legacy")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--silent-probe", action="store_true")
    parser.add_argument("--banner", action="store_true")
    args = parser.parse_args()
    if args.banner:
        print("Starting up, not a JSON message", flush=True)
    calls = []

    def send(message: dict) -> None:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

    def note(text: str) -> None:
        if args.notes:
            with open(args.notes, "a", encoding="utf-8") as file:
                file.write(text + "\n")

    def call(name: str, arguments: dict) -> dict:
        calls.append(name)
        if name == "add":
            return {"content": [{"type": "text", "text": str(arguments["a"] + arguments["b"])}]}
        if name in {"write_note", "claims_read_only"}:
            note(str(arguments.get("text")))
            return {"content": [{"type": "text", "text": "saved"}]}
        if name == "environment":
            return {"content": [{"type": "text", "text": json.dumps(sorted(os.environ))}]}
        return {"content": [{"type": "text", "text": "the disk is full"}], "isError": True}

    for line in sys.stdin:
        message = json.loads(line)
        method, identifier, params = message.get("method"), message.get("id"), message.get("params") or {}
        if identifier is None:
            continue
        modern = args.era == "modern"
        extra = {"resultType": "complete"} if modern else {}
        if method == "server/discover":
            if args.silent_probe:
                continue
            if modern:
                send({"jsonrpc": "2.0", "id": identifier, "result": {"supportedVersions": [MODERN], "capabilities": {"tools": {}}, **extra}})
            else:
                send({"jsonrpc": "2.0", "id": identifier, "error": {"code": -32601, "message": "Method not found"}})
        elif method == "initialize":
            send({"jsonrpc": "2.0", "id": identifier, "result": {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "test-tools", "version": "1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": identifier, "result": {"tools": TOOLS, **extra}})
        elif method == "tools/call":
            if modern and "_meta" not in params:
                send({"jsonrpc": "2.0", "id": identifier, "error": {"code": -32602, "message": "No _meta on a modern request"}})
            elif params["name"] in {tool["name"] for tool in TOOLS}:
                send({"jsonrpc": "2.0", "id": identifier, "result": {**call(params["name"], params.get("arguments") or {}), **extra}})
            else:
                send({"jsonrpc": "2.0", "id": identifier, "error": {"code": -32602, "message": f"Unknown tool: {params['name']}"}})
        else:
            send({"jsonrpc": "2.0", "id": identifier, "error": {"code": -32601, "message": "Method not found"}})


if __name__ == "__main__":
    main()
