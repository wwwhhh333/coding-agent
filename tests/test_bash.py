"""Unit tests for the bash tool (agent.tools.bash)."""
import tempfile
import unittest

from agent.tools import bash


class TestBashRun(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="agent_test_bash_")

    def test_simple_command(self):
        out = bash.run("echo hello", self.ws)
        self.assertEqual(out, "hello")

    def test_dangerous_blocked(self):
        out = bash.run("rm -rf /", self.ws)
        self.assertTrue(out.startswith("[blocked]"), out)

    def test_no_output_reports_exit_code(self):
        out = bash.run("true", self.ws)
        self.assertIn("no output", out)

    def test_utf8_output_captured(self):
        # Regression: on Chinese-locale Windows, text=True decoded child
        # output as GBK and dropped non-ASCII. Must capture UTF-8 cleanly.
        out = bash.run('printf "\\u4f60\\u597d"', self.ws)
        self.assertEqual(out, "你好")

    def test_timeout(self):
        out = bash.run("sleep 5", self.ws, timeout=1)
        self.assertTrue(out.startswith("[timeout]"), out)

    def test_output_truncation(self):
        out = bash.run("python -c 'print(\"x\"*100)'", self.ws, max_output=20)
        self.assertIn("truncated", out)
        self.assertLess(len(out), 200)


if __name__ == "__main__":
    unittest.main()
