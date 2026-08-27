"""Tool schema + dispatch table.

A single source of truth: the same definitions drive both what is advertised
to the model (``TOOL_SCHEMAS``) and what is executed locally (``execute``).
"""
from __future__ import annotations

from pathlib import Path

from . import bash, files

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the working directory and return its output. "
                "Use for building, running, testing, and listing. Note: the shell is "
                "bash on this system."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "shell command"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's content, optionally a line range. Large files should be read in chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-based first line"},
                    "end_line": {"type": "integer", "description": "1-based last line"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Prefer reading the file first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a directory matching a glob pattern (e.g. **/*.py).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "directory, default ."},
                    "pattern": {"type": "string", "description": "glob pattern, default *"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search files under a directory for a regex pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex"},
                    "path": {"type": "string", "description": "directory, default ."},
                },
                "required": ["pattern"],
            },
        },
    },
]


def execute(workspace: str | Path, name: str, arguments: dict) -> str:
    """Dispatch one parsed tool call to its local implementation."""
    workspace = Path(workspace)

    if name == "bash":
        cmd = arguments.get("command")
        if not isinstance(cmd, str):
            return "[error] bash requires a 'command' string"
        return bash.run(cmd, cwd=workspace)

    if name == "read_file":
        path = arguments.get("path")
        if not isinstance(path, str):
            return "[error] read_file requires a 'path' string"
        return files.read_file(
            workspace, path, arguments.get("start_line"), arguments.get("end_line")
        )

    if name == "write_file":
        path, content = arguments.get("path"), arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return "[error] write_file requires string 'path' and 'content'"
        return files.write_file(workspace, path, content)

    if name == "list_files":
        return files.list_files(
            workspace, arguments.get("path", "."), arguments.get("pattern", "*")
        )

    if name == "search":
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str):
            return "[error] search requires a 'pattern' string"
        return files.search(workspace, pattern, arguments.get("path", "."))

    return f"[error] unknown tool: {name}"
