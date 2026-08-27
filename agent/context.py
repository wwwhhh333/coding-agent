"""Context management (ADR-0002): observation truncation + summarization compaction.

Two tiers:
1. Per-message: any tool observation longer than ``MAX_OBSERVATION_CHARS`` is
   truncated with a marker, killing the most common context blow-up.
2. Whole-history: when the estimated token count crosses a budget ratio of the
   model window, the middle of the history is summarized into one compact
   message, while the pinned head (system + original task) and the recent tail
   survive verbatim — preserving intent, sacrificing mid-task detail.
"""
from __future__ import annotations

import json
from typing import Any, Callable

MAX_OBSERVATION_CHARS = 6000
CHARS_PER_TOKEN = 2.0  # rough heuristic, no offline tokenizer available
BUDGET_RATIO = 0.6
KEEP_HEAD = 2  # system prompt + pinned original user task
KEEP_TAIL = 8  # most recent messages preserved verbatim

COMPACT_MARKER = (
    "[context compacted] 以下是此前对话的摘要（含任务进度与关键事实），"
    "继续执行时无需重复查看原始过程：\n"
)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_history_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def truncate_observations(messages: list[dict]) -> list[dict]:
    """Return a copy with every tool observation capped at the limit."""
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            content = msg["content"]
            if len(content) > MAX_OBSERVATION_CHARS:
                msg = {
                    **msg,
                    "content": content[:MAX_OBSERVATION_CHARS]
                    + f"\n…(truncated, total {len(content)} chars)",
                }
        out.append(msg)
    return out


def needs_compaction(messages: list[dict], window_tokens: int) -> bool:
    return estimate_history_tokens(messages) >= int(window_tokens * BUDGET_RATIO)


def compact_history(
    messages: list[dict],
    summarizer: Callable[[list[dict]], str],
    window_tokens: int = 64_000,
) -> list[dict]:
    """Compact an over-budget history; returns the original when not needed."""
    if not needs_compaction(messages, window_tokens):
        return messages
    if len(messages) <= KEEP_HEAD + KEEP_TAIL:
        return messages

    head = messages[:KEEP_HEAD]
    tail = messages[-KEEP_TAIL:]
    middle = messages[KEEP_HEAD:-KEEP_TAIL]

    summary = summarizer(middle).strip()
    if not summary:
        return messages  # summarizer failed — keep history intact rather than corrupt it

    return head + [{"role": "user", "content": COMPACT_MARKER + summary}] + tail


def make_summarizer(cfg: Any) -> Callable[[list[dict]], str]:
    """Build a summarizer that compresses a message slice via the LLM."""
    from . import llm

    def summarize(messages: list[dict]) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是任务进度摘要器。把下面的对话压缩成简洁的中文摘要，"
                    "保留：当前任务目标、已完成的关键步骤、重要的文件路径/命令/结果、"
                    "待办事项。只输出摘要本身，不要评价。"
                ),
            },
            {"role": "user", "content": json.dumps(messages, ensure_ascii=False)},
        ]
        resp = llm.chat_complete(cfg, prompt, stream=False)
        return resp.choices[0].message.content or ""

    return summarize
