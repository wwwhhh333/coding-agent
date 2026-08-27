"""bash tool: subprocess execution with timeout and output truncation.

Commands run in a fresh, stateless subprocess rooted at the workspace. The
safety boundary lives in :mod:`security` (deny-list + cwd confinement).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import security

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_CHARS = 3000  # matches the observation-truncation limit


def run(
    command: str,
    cwd: str | Path,
    timeout: int = DEFAULT_TIMEOUT,
    max_output: int = MAX_OUTPUT_CHARS,
) -> str:
    """Execute a command in bash and return its (truncated) combined output."""
    reason = security.dangerous_command(command)
    if reason:
        return f"[blocked] command refused: {reason}"

    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] command exceeded {timeout}s"
    except FileNotFoundError:
        return "[error] bash not found on PATH"

    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.strip()
    if not out:
        return f"(no output, exit code {proc.returncode})"

    if len(out) > max_output:
        out = out[:max_output] + f"\n…(truncated, total {len(out)} chars)"
    return out
