"""CLI entry point.

Runs the agent on a task from the terminal with live streaming output and a
JSONL debug log. Provider settings come from the environment; ``--model`` and
``--base-url`` optionally override ``AGENT_MODEL`` / ``AGENT_BASE_URL``.

Example:
    python -m agent.main --task "列出当前目录的文件" --dir .
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from . import llm
from . import logging as agent_logging
from .loop import run_agent

_EXIT_CODE = {"completed": 0, "max_steps": 1, "dead_loop": 1,
              "interrupted": 130, "error": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agent.main",
        description="Autonomous coding agent CLI.",
    )
    parser.add_argument("--task", required=True, help="the task to accomplish")
    parser.add_argument("--dir", default=".", help="working directory (default: current)")
    parser.add_argument("--max-steps", type=int, default=30, help="step budget (default: 30)")
    parser.add_argument("--model", help="override AGENT_MODEL")
    parser.add_argument("--base-url", help="override AGENT_BASE_URL")
    parser.add_argument("--no-log", action="store_true",
                        help="skip writing the JSONL run log")
    return parser.parse_args(argv)


def _default_log_path(workspace: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(Path(workspace) / f"agent-run-{stamp}.jsonl")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    workspace = str(Path(args.dir).resolve())
    cfg = llm.load_config()
    if args.model:
        cfg = llm.LLMConfig(cfg.api_key, cfg.base_url, args.model)
    if args.base_url:
        cfg = llm.LLMConfig(cfg.api_key, args.base_url.rstrip("/"), cfg.model)

    logger = agent_logging.RunLogger(
        None if args.no_log else _default_log_path(workspace)
    )
    started = time.time()

    try:
        result = run_agent(
            cfg,
            workspace,
            args.task,
            max_steps=args.max_steps,
            on_chunk=logger.chunk,
            on_step=logger.event,
        )
    finally:
        logger.close()

    logger.summary(result.text, result.steps, result.stopped_reason,
                   time.time() - started)
    return _EXIT_CODE.get(result.stopped_reason, 1)


if __name__ == "__main__":
    sys.exit(main())
