"""Agent main loop (ADR-0001).

The loop is deliberately small because every moving part lives in a dedicated
module: parsing (messages), client/retries (llm), context management
(context), and tools (tools.registry). This module only orchestrates them and
decides when to stop.

Stop conditions:
- primary: the model returns text with no tool calls (completion)
- safety nets: max steps, dead-loop detection, user interrupt, fatal API error
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from . import context, llm, messages
from .tools import registry

DEFAULT_MAX_STEPS = 30
DEAD_LOOP_THRESHOLD = 3
DEFAULT_WINDOW_TOKENS = 64_000


@dataclass
class AgentResult:
    text: str
    steps: int
    stopped_reason: str  # completed | max_steps | dead_loop | interrupted | error


def build_system_prompt(workspace: str) -> str:
    return f"""你是运行在「{workspace}」目录下的自主编程智能体（coding agent）。
用户会给你一个编程任务，你要通过反复调用工具来完成它。

工作准则：
1. 修改任何文件之前，先读取它，避免盲目覆盖。
2. 每完成一处实质改动，用 bash 运行验证（写代码不跑等于没写）。
3. 小步推进：一次只改一处，改完即验。
4. 报错时先读取完整错误信息再行动，不要靠猜。
5. 需求不明确时，先询问用户，不要臆断。
6. 完成所有工作后，用一段话向用户总结你做了什么、结果如何。

工具使用策略：
- 列目录、找文件用 list_files / search，不要用 bash 反复 cat。
- 大文件用 read_file 分段读取。
- 危险命令（删除、格式化等）会被拦截并给出原因，请改用安全做法。

你的每次回复要么是一段普通文本（表示任务完成，即最终回答），
要么是工具调用（继续工作）。"""


def run_agent(
    cfg: llm.LLMConfig,
    workspace: str,
    task: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    on_chunk: Callable[[str], None] | None = None,
    on_step: Callable[[int, str, Any], None] | None = None,
) -> AgentResult:
    """Run the agent on ``task`` until it finishes or a safety net fires.

    ``on_chunk`` receives streamed model text for live display.
    ``on_step(step_no, kind, payload)`` receives structured events
    (``tool``, ``tool_error``, ``compacted``) for logging.
    """
    summarizer = context.make_summarizer(cfg)
    history: list[dict] = [
        messages.system(build_system_prompt(workspace)),
        messages.user(task),
    ]
    tool_schemas = registry.TOOL_SCHEMAS

    last_call_key: str | None = None
    repeat_count = 0

    try:
        for step in range(1, max_steps + 1):
            if context.needs_compaction(history, window_tokens):
                history = context.compact_history(history, summarizer, window_tokens)
                if on_step:
                    on_step(step, "compacted", "上下文已压缩")

            history = context.truncate_observations(history)

            stream = llm.chat_complete(cfg, history, tools=tool_schemas, stream=True)
            acc = messages.new_accumulator()
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                acc = messages.accumulate_stream(acc, delta)
                if on_chunk and delta.content:
                    on_chunk(delta.content)

            history.append(acc)

            if messages.is_final_answer(acc):
                return AgentResult(
                    text=acc.get("content") or "", steps=step, stopped_reason="completed"
                )

            for tc in messages.tool_calls_from_message(acc):
                name = tc["function"]["name"]
                call_id = tc.get("id") or f"call_{step}"
                parsed, reason = messages.parse_arguments(tc["function"]["arguments"])
                if parsed is None:
                    history.append(messages.tool_error(call_id, reason))
                    if on_step:
                        on_step(step, "tool_error", f"{name}: {reason}")
                    continue

                call_key = json.dumps([name, parsed], sort_keys=True)
                if call_key == last_call_key:
                    repeat_count += 1
                else:
                    last_call_key, repeat_count = call_key, 1
                if repeat_count >= DEAD_LOOP_THRESHOLD:
                    return AgentResult(
                        text="检测到重复调用同一工具且无进展，已终止。",
                        steps=step,
                        stopped_reason="dead_loop",
                    )

                result = registry.execute(workspace, name, parsed)
                history.append(messages.tool(call_id, result))
                if on_step:
                    on_step(step, "tool", {"name": name, "arguments": parsed, "result": result})

        return AgentResult(
            text="达到最大步数上限，已停止。", steps=max_steps, stopped_reason="max_steps"
        )

    except KeyboardInterrupt:
        return AgentResult(
            text="用户中断，已停止。", steps=step if "step" in locals() else 0,
            stopped_reason="interrupted",
        )
    except RuntimeError as exc:
        return AgentResult(text=f"API 错误：{exc}", steps=step if "step" in locals() else 0,
                           stopped_reason="error")
