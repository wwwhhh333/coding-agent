# coding-agent

A minimal, transparent coding agent built from scratch: it talks to a large language model and autonomously reads/writes files and runs commands to complete programming tasks.

- Core loop: native tool calling (OpenAI-compatible `tool_calls`)
- Context management: observation truncation + summarization compaction
- Error handling: errors are observations — the model repairs its own mistakes
- Safety: working-directory isolation, dangerous-command blocking, command timeout
- Model-agnostic: switch providers via environment variables
