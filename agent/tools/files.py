"""File tools: read_file, write_file, list_files, search.

All paths are confined to the workspace by :func:`security.resolve_in_workspace`,
so ``../`` and symlink escapes are rejected before any I/O happens.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import security

MAX_READ_CHARS = 6000
_MAX_SEARCH_FILE_SIZE = 1_000_000  # skip huge binaries/logs
_MAX_SEARCH_HITS = 60


def read_file(
    workspace: str | Path,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    p = security.resolve_in_workspace(workspace, path)
    if not p.is_file():
        return f"[error] not a file: {path}"

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = start_line if start_line else 1
    end = end_line if end_line else len(lines)
    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return f"[error] invalid line range {start_line}-{end_line} (file has {len(lines)} lines)"

    numbered = [f"{i}: {ln}" for i, ln in enumerate(lines[start - 1:end], start=start)]
    body = "\n".join(numbered)
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS] + f"\n…(truncated, file has {len(lines)} lines)"
    return f"{path} (lines {start}-{end} of {len(lines)}):\n{body}"


def write_file(workspace: str | Path, path: str, content: str) -> str:
    p = security.resolve_in_workspace(workspace, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {p} ({len(content)} chars)"


def list_files(workspace: str | Path, path: str = ".", pattern: str = "*") -> str:
    base = security.resolve_in_workspace(workspace, path)
    matches = sorted(m.relative_to(base).as_posix() for m in base.glob(pattern))
    if not matches:
        return f"(no files matching {pattern!r} under {path})"
    if len(matches) > 200:
        matches = matches[:200] + ["…(more)"]
    return "\n".join(matches)


def search(workspace: str | Path, pattern: str, path: str = ".") -> str:
    base = security.resolve_in_workspace(workspace, path)
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"[error] invalid regex: {exc}"

    hits: list[str] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if p.stat().st_size > _MAX_SEARCH_FILE_SIZE:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = p.relative_to(base).as_posix()
                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                if len(hits) >= _MAX_SEARCH_HITS:
                    break
        if len(hits) >= _MAX_SEARCH_HITS:
            break

    if not hits:
        return f"(no matches for {pattern!r} under {path})"
    return "\n".join(hits)
