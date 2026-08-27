"""LLM client wrapper: provider-agnostic, env-var driven, retry with backoff.

Only the OpenAI-compatible HTTP client is used here (explicitly permitted by
the assessment). Everything about *how* the agent loops, parses, and executes
tools lives outside this module. Provider is switched by changing three env
vars, so the agent is not tied to any single model vendor.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import openai

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2, 4, 8)

# Transient conditions worth retrying: network failure, timeout, rate limit,
# and 5xx server errors. 4xx like bad request / auth are handled separately.
_TRANSIENT = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


def load_config() -> LLMConfig:
    """Read provider settings from the environment (never from the repo)."""
    api_key = os.environ.get("AGENT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AGENT_API_KEY is not set. Export it before running, e.g. "
            "`export AGENT_API_KEY=sk-...` on Unix or "
            "`$env:AGENT_API_KEY='sk-...'` in PowerShell."
        )
    return LLMConfig(
        api_key=api_key,
        base_url=os.environ.get("AGENT_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        model=os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
    )


def make_client(cfg: LLMConfig) -> openai.OpenAI:
    return openai.OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120)


def chat_complete(
    cfg: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
    stream: bool = False,
):
    """Call the chat completions endpoint with exponential-backoff retries.

    Raises ``RuntimeError`` with a clear message when the request ultimately
    fails, or when the API key is rejected (never retried).
    """
    client = make_client(cfg)
    kwargs: dict = {"model": cfg.model, "messages": messages, "stream": stream}
    if tools:
        kwargs["tools"] = tools

    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "API key was rejected (401). Check AGENT_API_KEY."
            ) from exc
        except _TRANSIENT as exc:
            last_err = exc
            if attempt < MAX_ATTEMPTS:
                delay = BACKOFF_SECONDS[attempt - 1]
                time.sleep(delay)

    raise RuntimeError(
        f"API request failed after {MAX_ATTEMPTS} attempts: {last_err}"
    ) from last_err
