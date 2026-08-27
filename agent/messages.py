"""Message construction and model-output parsing.

Implements the two "output parsing" responsibilities from the assessment:
- extract structured tool calls from the model's response (ADR-0001)
- recover from malformed tool arguments by sending the model a corrective
  observation instead of crashing (ADR-0003)

All messages are plain dicts in the OpenAI wire format, so the same objects
are both sent to the API and stored in the conversation history.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# message builders (OpenAI wire format)
# ---------------------------------------------------------------------------

def system(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


def user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def assistant(content: str | None, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool(tool_call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ---------------------------------------------------------------------------
# completion signal
# ---------------------------------------------------------------------------

def is_final_answer(msg: dict[str, Any]) -> bool:
    """The model is done when it produced text and no tool calls."""
    return bool(msg.get("content")) and not msg.get("tool_calls")


# ---------------------------------------------------------------------------
# parsing an assistant response into wire-format tool calls
# ---------------------------------------------------------------------------

def tool_calls_from_message(msg: Any) -> list[dict]:
    """Convert an SDK assistant message's tool_calls into wire-format dicts.

    Accepts either an SDK ``ChatCompletionMessage`` (has ``tool_calls`` with
    ``id``/``function`` attributes) or an already-wire-format dict.
    """
    raw = getattr(msg, "tool_calls", None)
    if not raw:
        return []

    out: list[dict] = []
    for tc in raw:
        if isinstance(tc, dict):
            out.append(tc)
            continue
        out.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )
    return out


def parse_arguments(arguments: str) -> tuple[dict | None, str | None]:
    """Return ``(parsed, None)`` on success or ``(None, reason)`` on failure."""
    if not arguments or not arguments.strip():
        return None, "arguments were empty"
    try:
        obj = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return None, f"arguments were not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
    if not isinstance(obj, dict):
        return None, f"arguments must be a JSON object, got {type(obj).__name__}"
    return obj, None


def tool_error(tool_call_id: str, reason: str) -> dict[str, Any]:
    """A corrective observation for the model when a call cannot be executed."""
    return tool(tool_call_id, f"[tool error] could not execute: {reason}")


# ---------------------------------------------------------------------------
# streaming accumulation (signature feature: live token output)
# ---------------------------------------------------------------------------

def accumulate_stream(acc: dict[str, Any], delta: Any) -> dict[str, Any]:
    """Fold one SDK stream delta into an accumulating assistant message.

    ``delta.tool_calls`` arrive in fragments keyed by index; the index must
    match the one in the tool call's wire format we emit, so we keep it.
    """
    content = getattr(delta, "content", None)
    if content:
        acc["content"] = (acc.get("content") or "") + content

    raw_calls = getattr(delta, "tool_calls", None)
    if not raw_calls:
        return acc

    calls = acc.get("tool_calls")
    if not calls:
        calls = []
        acc["tool_calls"] = calls
    for part in raw_calls:
        idx = part.index
        while len(calls) <= idx:
            calls.append({"id": None, "type": "function",
                          "function": {"name": None, "arguments": ""}})
        slot = calls[idx]
        if part.id:
            slot["id"] = part.id
        fn = part.function
        if fn:
            if fn.name:
                slot["function"]["name"] = fn.name
            if fn.arguments:
                slot["function"]["arguments"] += fn.arguments
    return acc


def new_accumulator() -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": None}
