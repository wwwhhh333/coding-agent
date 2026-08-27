"""Unit tests for the agent loop (agent.loop) with a mocked LLM stream."""
import unittest
from types import SimpleNamespace as NS

from agent import llm, messages
from agent.loop import DEAD_LOOP_THRESHOLD, run_agent


def _delta(content=None, tool_calls=None):
    return NS(content=content, tool_calls=tool_calls)


def _chunk(d):
    return NS(choices=[NS(delta=d)])


def _tool_part(idx, cid=None, name=None, arguments=None):
    return NS(index=idx, id=cid, function=NS(name=name, arguments=arguments))


def _tool_stream(cid, name, arguments):
    """Emit a tool call in fragments, mirroring real SSE deltas."""
    yield _chunk(_delta(None, [_tool_part(0, cid=cid, name=name, arguments="")]))
    yield _chunk(_delta(None, [_tool_part(0, arguments=arguments)]))


def _text_stream(text):
    yield _chunk(_delta(text))


class MockLLM:
    """Serves queued streams to chat_complete; fails on non-stream calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.stream_calls = 0

    def chat_complete(self, cfg, msgs, tools=None, stream=False):
        if not stream:
            raise AssertionError("non-stream call not expected in loop tests")
        self.stream_calls += 1
        return self._responses.pop(0)


def _run(responses, **kwargs):
    cfg = llm.LLMConfig(api_key="test", base_url="http://localhost",
                        model="test-model")
    mock = MockLLM(responses)
    llm.chat_complete = mock.chat_complete
    result = run_agent(cfg, ".", "test task", **kwargs)
    return result, mock


class TestLoop(unittest.TestCase):
    def test_completes_after_tool_roundtrip(self):
        responses = [
            _tool_stream("c1", "list_files", '{"path": "."}'),
            _text_stream("任务完成"),
        ]
        result, mock = _run(responses, max_steps=10)
        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.text, "任务完成")
        self.assertEqual(mock.stream_calls, 2)

    def test_completes_immediately(self):
        result, mock = _run([_text_stream("直接完成")], max_steps=10)
        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.steps, 1)

    def test_max_steps_safety_net(self):
        # model never finishes, but varies its calls so dead-loop detection
        # does not fire first; the step budget must be the one that stops it.
        responses = [
            _tool_stream("c1", "list_files", '{"path": "."}'),
            _tool_stream("c2", "list_files", '{"pattern": "*.txt"}'),
            _tool_stream("c1", "list_files", '{"path": "."}'),
        ]
        result, mock = _run(responses, max_steps=3)
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(result.steps, 3)

    def test_dead_loop_detection(self):
        responses = [_tool_stream("c1", "list_files", '{"path": "."}')
                     for _ in range(DEAD_LOOP_THRESHOLD + 2)]
        result, mock = _run(responses, max_steps=10)
        self.assertEqual(result.stopped_reason, "dead_loop")
        self.assertEqual(result.steps, DEAD_LOOP_THRESHOLD)

    def test_tool_error_is_observation(self):
        # invalid JSON arguments -> tool_error injected, loop continues
        bad = NS(choices=[NS(delta=_delta(
            None, [_tool_part(0, cid="c1", name="list_files", arguments="{bad")]))])
        responses = [iter([bad]), _text_stream("重试后完成")]
        result, mock = _run(responses, max_steps=10)
        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.steps, 2)

    def test_step_events_emitted(self):
        responses = [
            _tool_stream("c1", "list_files", '{"path": "."}'),
            _text_stream("完成"),
        ]
        events = []

        def on_step(step, kind, payload):
            events.append((step, kind, payload))

        cfg = llm.LLMConfig(api_key="test", base_url="http://localhost",
                            model="test-model")
        mock = MockLLM(responses)
        llm.chat_complete = mock.chat_complete
        run_agent(cfg, ".", "task", max_steps=10, on_step=on_step)

        tools = [e for e in events if e[1] == "tool"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0][2]["name"], "list_files")

    def test_message_helpers_used_for_history(self):
        # sanity: is_final_answer and tool_calls_from_message agree with loop
        final = messages.assistant("done")
        self.assertTrue(messages.is_final_answer(final))
        tool_msg = messages.assistant(None, [{"id": "c1", "type": "function",
                                              "function": {"name": "bash",
                                                           "arguments": "{}"}}])
        self.assertFalse(messages.is_final_answer(tool_msg))


if __name__ == "__main__":
    unittest.main()
