"""Unit tests for message construction and output parsing (agent.messages)."""
import unittest
from types import SimpleNamespace as NS

from agent import messages


class TestBuilders(unittest.TestCase):
    def test_system_and_user(self):
        self.assertEqual(messages.system("hi"), {"role": "system", "content": "hi"})
        self.assertEqual(messages.user("yo"), {"role": "user", "content": "yo"})

    def test_assistant_plain(self):
        self.assertEqual(messages.assistant("text"),
                         {"role": "assistant", "content": "text"})

    def test_assistant_with_tool_calls(self):
        msg = messages.assistant(None, [{"id": "1"}])
        self.assertEqual(msg["tool_calls"], [{"id": "1"}])

    def test_tool(self):
        self.assertEqual(messages.tool("c1", "out"),
                         {"role": "tool", "tool_call_id": "c1", "content": "out"})


class TestIsFinalAnswer(unittest.TestCase):
    def test_text_no_tool_calls_is_final(self):
        self.assertTrue(messages.is_final_answer({"role": "assistant",
                                                  "content": "done"}))
        self.assertTrue(messages.is_final_answer(
            {"role": "assistant", "content": "done", "tool_calls": []}))

    def test_empty_content_not_final(self):
        self.assertFalse(messages.is_final_answer(
            {"role": "assistant", "content": "", "tool_calls": None}))

    def test_tool_calls_not_final(self):
        self.assertFalse(messages.is_final_answer(
            {"role": "assistant", "content": "partial",
             "tool_calls": [{"id": "1"}]}))


class TestToolCallsFromMessage(unittest.TestCase):
    def test_wire_dict(self):
        wire = {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "bash",
                                             "arguments": '{"command": "ls"}'}}]}
        out = messages.tool_calls_from_message(wire)
        self.assertEqual(out, wire["tool_calls"])

    def test_sdk_object(self):
        obj = NS(tool_calls=[NS(id="c1", function=NS(name="bash",
                                                    arguments='{"command": "ls"}'))])
        out = messages.tool_calls_from_message(obj)
        self.assertEqual(out[0]["id"], "c1")
        self.assertEqual(out[0]["function"]["name"], "bash")
        self.assertEqual(out[0]["function"]["arguments"], '{"command": "ls"}')

    def test_none_returns_empty(self):
        self.assertEqual(messages.tool_calls_from_message(NS(tool_calls=None)), [])
        self.assertEqual(messages.tool_calls_from_message({"content": "x"}), [])


class TestParseArguments(unittest.TestCase):
    def test_valid(self):
        obj, reason = messages.parse_arguments('{"path": "a.txt"}')
        self.assertEqual(obj, {"path": "a.txt"})
        self.assertIsNone(reason)

    def test_empty(self):
        obj, reason = messages.parse_arguments("")
        self.assertIsNone(obj)
        self.assertIn("empty", reason)

    def test_invalid_json(self):
        obj, reason = messages.parse_arguments("{not json")
        self.assertIsNone(obj)
        self.assertIn("not valid JSON", reason)

    def test_non_object(self):
        obj, reason = messages.parse_arguments("[1, 2]")
        self.assertIsNone(obj)
        self.assertIn("must be a JSON object", reason)


class TestToolError(unittest.TestCase):
    def test_message(self):
        msg = messages.tool_error("c1", "bad args")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "c1")
        self.assertIn("could not execute", msg["content"])
        self.assertIn("bad args", msg["content"])


class TestAccumulateStream(unittest.TestCase):
    def _fn_part(self, idx, name=None, arguments=None, cid=None):
        return NS(index=idx, id=cid,
                  function=NS(name=name, arguments=arguments))

    def test_content_concatenation(self):
        acc = messages.new_accumulator()
        acc = messages.accumulate_stream(acc, NS(content="hel", tool_calls=None))
        acc = messages.accumulate_stream(acc, NS(content="lo", tool_calls=None))
        self.assertEqual(acc["content"], "hello")
        self.assertIsNone(acc["tool_calls"])

    def test_tool_call_fragments_by_index(self):
        acc = messages.new_accumulator()
        # index 1 arrives before index 0; slots must be created in order
        acc = messages.accumulate_stream(
            acc, NS(content=None,
                    tool_calls=[self._fn_part(1, cid="c1", name="bash",
                                              arguments='{"comm')]))
        acc = messages.accumulate_stream(
            acc, NS(content=None,
                    tool_calls=[self._fn_part(0, cid="c0", name="read_file",
                                              arguments='{"path":')]))
        acc = messages.accumulate_stream(
            acc, NS(content=None,
                    tool_calls=[self._fn_part(1, arguments='and": "ls"}')]))
        calls = acc["tool_calls"]
        self.assertEqual(calls[0]["id"], "c0")
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(calls[0]["function"]["arguments"], '{"path":')
        self.assertEqual(calls[1]["id"], "c1")
        self.assertEqual(calls[1]["function"]["name"], "bash")
        self.assertEqual(calls[1]["function"]["arguments"], '{"command": "ls"}')

    def test_new_accumulator(self):
        acc = messages.new_accumulator()
        self.assertEqual(acc["role"], "assistant")
        self.assertIsNone(acc["content"])
        self.assertIsNone(acc["tool_calls"])


if __name__ == "__main__":
    unittest.main()
