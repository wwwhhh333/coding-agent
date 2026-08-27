"""Unit tests for the safety guards (agent.tools.security)."""
import unittest
from pathlib import Path

from agent.tools import security


class TestDangerousCommand(unittest.TestCase):
    BLOCKED = [
        "rm -rf /",
        "rm -fr /etc",
        "rm -r -f data",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "format c:",
        "diskpart",
        "regedit",
        "rmdir /s /q c:\\temp",
        "del /f /q *.tmp",
        "shutdown -s -t 0",
        "reboot",
        "poweroff",
        "sudo rm -rf /",
        "su root",
        ":(){ :|:& };:",
        "git push --force origin main",
        "git push -f",
        "git reset --hard HEAD",
        "git clean -f -d",
        "git filter-repo --force",
    ]

    def test_blocked(self):
        for cmd in self.BLOCKED:
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(security.dangerous_command(cmd), cmd)

    SAFE = [
        "ls -la",
        "echo hello",
        "rm old_backup.txt",
        "git status",
        "git log --oneline",
        "python -m unittest discover",
        "cat file.txt",
    ]

    def test_safe(self):
        for cmd in self.SAFE:
            with self.subTest(cmd=cmd):
                self.assertIsNone(security.dangerous_command(cmd), cmd)


class TestResolveInWorkspace(unittest.TestCase):
    def setUp(self):
        self.ws = Path(self.mktmpdir())
        (self.ws / "inside.txt").touch()

    @staticmethod
    def mktmpdir() -> str:
        import tempfile
        return tempfile.mkdtemp(prefix="agent_test_ws_")

    def test_relative_inside(self):
        p = security.resolve_in_workspace(self.ws, "inside.txt")
        self.assertEqual(p, (self.ws / "inside.txt").resolve())

    def test_absolute_inside(self):
        target = str(self.ws / "inside.txt")
        p = security.resolve_in_workspace(self.ws, target)
        self.assertEqual(p, (self.ws / "inside.txt").resolve())

    def test_parent_escape_raises(self):
        with self.assertRaises(ValueError):
            security.resolve_in_workspace(self.ws, "..")
        with self.assertRaises(ValueError):
            security.resolve_in_workspace(self.ws, "../../etc/passwd")

    def test_absolute_outside_raises(self):
        with self.assertRaises(ValueError):
            security.resolve_in_workspace(self.ws, str(Path("/tmp")))

    def test_missing_inside_still_resolves(self):
        p = security.resolve_in_workspace(self.ws, "not-there.txt")
        self.assertEqual(p, (self.ws / "not-there.txt").resolve())


if __name__ == "__main__":
    unittest.main()
