"""Unit tests for tool schema and dispatch (agent.tools.registry)."""
import tempfile
import unittest
from pathlib import Path

from agent.tools import registry


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="agent_test_reg_")

    def test_schemas_expose_five_tools(self):
        names = [s["function"]["name"] for s in registry.TOOL_SCHEMAS]
        self.assertEqual(sorted(names),
                         ["bash", "list_files", "read_file", "search",
                          "write_file"])

    def test_unknown_tool(self):
        out = registry.execute(self.ws, "frobnicate", {})
        self.assertIn("[error] unknown tool", out)

    def test_bash_dispatch(self):
        out = registry.execute(self.ws, "bash", {"command": "echo dispatch"})
        self.assertEqual(out, "dispatch")

    def test_bash_missing_arg(self):
        out = registry.execute(self.ws, "bash", {})
        self.assertIn("[error]", out)

    def test_read_file_dispatch(self):
        (Path(self.ws) / "f.txt").write_text("one\ntwo\n", encoding="utf-8")
        out = registry.execute(self.ws, "read_file", {"path": "f.txt"})
        self.assertIn("1: one", out)

    def test_write_file_dispatch(self):
        out = registry.execute(self.ws, "write_file",
                               {"path": "w.txt", "content": "data"})
        self.assertIn("wrote", out)
        self.assertEqual((Path(self.ws) / "w.txt").read_text(encoding="utf-8"),
                         "data")

    def test_list_files_dispatch(self):
        (Path(self.ws) / "a.txt").touch()
        out = registry.execute(self.ws, "list_files", {"pattern": "*.txt"})
        self.assertIn("a.txt", out)

    def test_search_dispatch(self):
        (Path(self.ws) / "s.txt").write_text("needle here\n", encoding="utf-8")
        out = registry.execute(self.ws, "search", {"pattern": "needle"})
        self.assertIn("s.txt:1", out)


if __name__ == "__main__":
    unittest.main()
