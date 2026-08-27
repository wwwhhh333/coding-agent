"""Unit tests for context management (agent.context)."""
import unittest

from agent import context


class TestEstimateTokens(unittest.TestCase):
    def test_minimum(self):
        self.assertEqual(context.estimate_tokens(""), 1)

    def test_char_heuristic(self):
        self.assertEqual(context.estimate_tokens("abcd"), 2)   # 4 / 2.0
        self.assertEqual(context.estimate_tokens("abc"), 1)    # int(1.5)


class TestTruncateObservations(unittest.TestCase):
    def test_long_tool_message_capped(self):
        big = "x" * (context.MAX_OBSERVATION_CHARS + 500)
        msgs = [{"role": "tool", "tool_call_id": "c", "content": big}]
        out = context.truncate_observations(msgs)
        self.assertEqual(len(out), 1)
        self.assertLessEqual(len(out[0]["content"]), context.MAX_OBSERVATION_CHARS + 60)
        self.assertIn("truncated", out[0]["content"])

    def test_short_and_non_tool_untouched(self):
        msgs = [
            {"role": "tool", "tool_call_id": "c", "content": "small"},
            {"role": "user", "content": "x" * (context.MAX_OBSERVATION_CHARS + 100)},
        ]
        out = context.truncate_observations(msgs)
        self.assertEqual(out[0]["content"], "small")
        self.assertEqual(out[1]["content"], msgs[1]["content"])

    def test_returns_new_list(self):
        msgs = [{"role": "tool", "tool_call_id": "c", "content": "ok"}]
        out = context.truncate_observations(msgs)
        self.assertIsNot(out, msgs)


class TestNeedsCompaction(unittest.TestCase):
    def test_over_budget_triggers(self):
        big = {"role": "assistant", "content": "y" * 1000}  # ~500+ tokens
        msgs = [big]
        self.assertTrue(context.needs_compaction(msgs, window_tokens=100))

    def test_small_history_ok(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertFalse(context.needs_compaction(msgs, window_tokens=100_000))


class TestCompactHistory(unittest.TestCase):
    def _history(self, n):
        return [{"role": "user", "content": f"msg {i}"} for i in range(n)]

    def test_small_history_returned_as_is(self):
        msgs = self._history(context.KEEP_HEAD + context.KEEP_TAIL)
        out = context.compact_history(msgs, lambda m: "S", window_tokens=1)
        self.assertEqual(out, msgs)

    def test_large_history_compacted(self):
        n = context.KEEP_HEAD + context.KEEP_TAIL + 6
        msgs = self._history(n)
        out = context.compact_history(msgs, lambda m: "summary-text", window_tokens=1)
        self.assertEqual(len(out), context.KEEP_HEAD + 1 + context.KEEP_TAIL)
        self.assertEqual(out[:context.KEEP_HEAD], msgs[:context.KEEP_HEAD])
        self.assertEqual(out[context.KEEP_HEAD]["role"], "user")
        self.assertIn("summary-text", out[context.KEEP_HEAD]["content"])
        self.assertEqual(out[context.KEEP_HEAD + 1:], msgs[-context.KEEP_TAIL:])

    def test_failed_summarizer_keeps_original(self):
        msgs = self._history(context.KEEP_HEAD + context.KEEP_TAIL + 4)
        out = context.compact_history(msgs, lambda m: "", window_tokens=1)
        self.assertEqual(out, msgs)


class TestMakeSummarizer(unittest.TestCase):
    def test_returns_callable(self):
        summarizer = context.make_summarizer(object())
        self.assertTrue(callable(summarizer))


if __name__ == "__main__":
    unittest.main()
