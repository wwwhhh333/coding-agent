"""Terminal output (ANSI) and JSONL debug logging.

``RunLogger`` is what the CLI wires into the loop's ``on_chunk`` / ``on_step``
callbacks: it renders live, human-readable progress to the terminal and, when a
log file is given, appends every step event as one JSON line so a whole run is
replayable and diffable later.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from typing import Any, TextIO

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIMMED = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"

LOG_RESULT_LIMIT = 2000  # keep the JSONL log from ballooning with huge outputs


def supports_color(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except Exception:
        return False


def compact(obj: Any, limit: int = 400) -> str:
    """Render ``obj`` as a short one-line string for terminal display."""
    if isinstance(obj, dict):
        parts = [f"{k}={compact(v, limit // 2)}" for k, v in obj.items()]
        text = ", ".join(parts)
    elif isinstance(obj, str):
        text = obj.replace("\n", " ")
    else:
        text = str(obj)
    if len(text) > limit:
        text = text[:limit] + " …"
    return text


class RunLogger:
    """Write step events to the terminal and (optionally) a JSONL file."""

    def __init__(
        self,
        log_path: str | None = None,
        *,
        stream: TextIO = sys.stdout,
    ) -> None:
        self.stream = stream
        self.color = supports_color(stream)
        self.start = time.time()
        self._log_file = open(log_path, "a", encoding="utf-8") if log_path else None

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _color(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def _line(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()

    def _record(self, step: int, kind: str, payload: Any) -> None:
        if not self._log_file:
            return
        if kind == "tool" and isinstance(payload, dict) and isinstance(payload.get("result"), str):
            payload = {**payload, "result": payload["result"][:LOG_RESULT_LIMIT]}
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed": round(time.time() - self.start, 2),
            "step": step,
            "kind": kind,
            "payload": payload,
        }
        self._log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log_file.flush()

    def chunk(self, text: str) -> None:
        """Streamed model text — printed raw so tokens appear live."""
        self.stream.write(text)
        self.stream.flush()

    def event(self, step: int, kind: str, payload: Any) -> None:
        """One loop event: ``tool``, ``tool_error`` or ``compacted``."""
        self._record(step, kind, payload)
        if kind == "tool":
            name = payload.get("name", "?")
            args = compact(payload.get("arguments", {}), limit=200)
            self._line(self._color(CYAN, f"  step {step} > {name}") + f" ({args})")
            result = payload.get("result")
            if result:
                self._line("    " + self._color(DIMMED, compact(result)))
        elif kind == "tool_error":
            self._line(self._color(YELLOW, f"  step {step} ! {payload}"))
        elif kind == "compacted":
            self._line(self._color(MAGENTA, f"  step {step} ~ {payload}"))

    def summary(self, text: str, steps: int, reason: str, elapsed: float) -> None:
        """Render the end-of-run status line, color-coded by stop reason."""
        color = GREEN if reason == "completed" else YELLOW if reason in (
            "max_steps", "dead_loop", "interrupted") else RED
        self._line("")
        self._line(self._color(BOLD, "=== 运行结束 ==="))
        self._line(f"停止原因: {self._color(color, reason)}")
        self._line(f"步骤数:   {steps}")
        self._line(f"耗时:     {elapsed:.1f}s")
        if text:
            self._line("")
            self._line(self._color(BOLD, "最终回答:"))
            self._line(text)
