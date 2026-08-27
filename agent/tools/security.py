"""Path isolation and dangerous-command blocking (ADR-0004).

Deliberately not a sandbox: the agent runs as the current user on their own
machine, so a full sandbox would be protecting the user from themselves.
Instead we apply policy guards at the tool layer — workspace confinement for
file/command access plus a deny-list for destructive commands.
"""
from __future__ import annotations

import re
from pathlib import Path

# (compiled pattern, human-readable reason)
_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # destructive filesystem
    # requires BOTH an -r and an -f flag (any order/combination) before the
    # command separator; catches rm -rf, -fr, and split "-r -f" alike.
    (re.compile(r"\brm\b(?=[^;\n|]*-[a-z]*r)(?=[^;\n|]*-[a-z]*f)[^;\n|]*",
                re.IGNORECASE), "recursive force delete"),
    (re.compile(r"\bmkfs\b"), "filesystem creation"),
    (re.compile(r"\bdd\s+if="), "raw device write"),
    (re.compile(r"\bformat\s+"), "format"),
    (re.compile(r"\bdiskpart\b"), "diskpart"),
    (re.compile(r"\bregedit\b"), "registry editor"),
    (re.compile(r"\brmdir\s+/s\b", re.IGNORECASE), "recursive directory delete"),
    (re.compile(r"\bdel\s+/[sfq]\b", re.IGNORECASE), "bulk file delete"),
    # system control
    (re.compile(r"\bshutdown\b"), "shutdown"),
    (re.compile(r"\breboot\b"), "reboot"),
    (re.compile(r"\bpoweroff\b"), "poweroff"),
    # privilege escalation
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\bsu\s+"), "switch user"),
    # runaway resource use
    (re.compile(r":\(\)\s*\{"), "fork bomb"),
    # git history rewrite (forbidden by the assessment and the git spec)
    (re.compile(r"\bgit\s+push\b[^\n]*\s(--force|-f)\b"), "force push"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "hard reset"),
    (re.compile(r"\bgit\s+clean\s+-[a-z]*f\b"), "git clean -f"),
    (re.compile(r"\bgit\s+filter-repo\b"), "history rewrite"),
]


def dangerous_command(command: str) -> str | None:
    """Return a reason string if the command is blocked, else ``None``."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def resolve_in_workspace(workspace: str | Path, target: str | Path) -> Path:
    """Resolve ``target`` to an absolute path guaranteed to be inside ``workspace``.

    Raises ``ValueError`` when the target escapes the workspace (via ``..``,
    absolute paths, or symlinks), enforcing the isolation boundary.
    """
    ws = Path(workspace).resolve()
    raw = Path(target)
    candidate = raw if raw.is_absolute() else ws / raw
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ws):
        raise ValueError(f"path escapes workspace: {target}")
    return resolved
