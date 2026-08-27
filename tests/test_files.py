"""Unit tests for the file tools (agent.tools.files)."""
import tempfile
import unittest
from pathlib import Path

from agent.tools import files


class TestFileTools(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="agent_test_files_")
        (Path(self.ws) / "sub").mkdir()
        (Path(self.ws) / "hello.txt").write_text("line1\nline2\nline3\n",
                                                 encoding="utf-8")

    def test_write_and_read_roundtrip(self):
        msg = files.write_file(self.ws, "out/data.txt", "alpha\nbeta\n")
        self.assertIn("wrote", msg)
        body = files.read_file(self.ws, "out/data.txt")
        self.assertIn("1: alpha", body)
        self.assertIn("2: beta", body)

    def test_read_missing_file(self):
        out = files.read_file(self.ws, "nope.txt")
        self.assertIn("[error] not a file", out)

    def test_read_line_range(self):
        out = files.read_file(self.ws, "hello.txt", start_line=2, end_line=2)
        self.assertIn("2: line2", out)
        self.assertNotIn("line1", out)

    def test_read_invalid_range(self):
        out = files.read_file(self.ws, "hello.txt", start_line=9, end_line=9)
        self.assertIn("[error] invalid line range", out)

    def test_write_creates_parent_dirs(self):
        files.write_file(self.ws, "a/b/c.txt", "x")
        self.assertTrue((Path(self.ws) / "a" / "b" / "c.txt").is_file())

    def test_list_files(self):
        out = files.list_files(self.ws, ".", "*.txt")
        self.assertIn("hello.txt", out)

    def test_list_files_empty(self):
        out = files.list_files(self.ws, "sub", "*")
        self.assertIn("no files matching", out)

    def test_search_hits(self):
        out = files.search(self.ws, r"line2")
        self.assertIn("hello.txt:2", out)

    def test_search_no_match(self):
        out = files.search(self.ws, r"zzz_nothing")
        self.assertIn("no matches", out)

    def test_search_invalid_regex(self):
        out = files.search(self.ws, "(")
        self.assertIn("[error] invalid regex", out)

    def test_escape_rejected(self):
        with self.assertRaises(ValueError):
            files.read_file(self.ws, "../secret.txt")


if __name__ == "__main__":
    unittest.main()
