"""Offline unit tests for external resolvers in oh_player.py (run: python streamer/tests/test_resolvers.py)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import oh_streamer.player.oh_player as LS  # noqa: E402


class FindResolverTest(unittest.TestCase):
    def setUp(self):
        self.orig_file = LS.RESOLVERS_FILE
        self.orig_default = LS.DEFAULT_RESOLVERS_FILE
        self.tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.tmp.close()
        LS.RESOLVERS_FILE = Path(self.tmp.name)
        LS.DEFAULT_RESOLVERS_FILE = Path(self.tmp.name + ".nonexistent")

    def tearDown(self):
        LS.RESOLVERS_FILE = self.orig_file
        LS.DEFAULT_RESOLVERS_FILE = self.orig_default
        if os.path.exists(self.tmp.name):
            os.remove(self.tmp.name)

    def write_config(self, content):
        Path(self.tmp.name).write_text(content, encoding="utf-8")

    def test_non_url_returns_none(self):
        self.assertIsNone(LS.find_resolver("not-a-url"))
        self.assertIsNone(LS.find_resolver("C:\\Videos\\a.mp4"))
        self.assertIsNone(LS.find_resolver(""))

    def test_missing_or_invalid_file(self):
        os.remove(self.tmp.name)
        self.assertIsNone(LS.find_resolver("https://example.com/video"))
        self.write_config("invalid-json{")
        self.assertIsNone(LS.find_resolver("https://example.com/video"))
        self.write_config('{"not": "a list"}')
        self.assertIsNone(LS.find_resolver("https://example.com/video"))

    def test_malformed_entries(self):
        self.write_config(json.dumps([
            "string",
            {"match": 123, "command": ["cmd"]},
            {"match": "valid", "command": []},
            {"match": "valid", "command": "not-a-list"},
            {"match": "", "command": ["cmd"]},
        ]))
        self.assertIsNone(LS.find_resolver("https://valid.com/video"))

    def test_matching_case_insensitive(self):
        self.write_config(json.dumps([
            {"match": "SiteA.COM", "command": ["cmd_a"]},
            {"match": "siteb.com/clip", "command": ["python", "b.py"]},
        ]))
        r1 = LS.find_resolver("https://sub.sitea.com/play/123")
        self.assertIsNotNone(r1)
        self.assertEqual(r1["command"], ["cmd_a"])

        r2 = LS.find_resolver("HTTPS://SITEB.COM/CLIP/XYZ")
        self.assertIsNotNone(r2)
        self.assertEqual(r2["command"], ["python", "b.py"])

        self.assertIsNone(LS.find_resolver("https://sitec.com/video"))

    def test_fallback_to_default(self):
        os.remove(self.tmp.name)
        default_tmp = self.tmp.name + ".default"
        try:
            Path(default_tmp).write_text(json.dumps([{"match": "default.tv", "command": ["cmd"]}]), encoding="utf-8")
            LS.DEFAULT_RESOLVERS_FILE = Path(default_tmp)
            r = LS.find_resolver("https://default.tv/clip")
            self.assertIsNotNone(r)
            self.assertEqual(r["command"], ["cmd"])
        finally:
            if os.path.exists(default_tmp):
                os.remove(default_tmp)


class ResolveExternalTest(unittest.TestCase):
    @patch("subprocess.run")
    def test_resolve_success(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "url": "https://cdn.example.com/index.m3u8",
            "title": "Test Title",
            "headers": {"Referer": "https://example.com/"}
        })
        mock_run.return_value = mock_proc

        resolver = {"match": "example.com", "command": ["python", "my_script.py"]}
        url, headers, title = LS.resolve_external(
            "https://example.com/video/1",
            {"User-Agent": "CustomUA"},
            resolver
        )
        self.assertEqual(url, "https://cdn.example.com/index.m3u8")
        self.assertEqual(title, "Test Title")
        self.assertEqual(headers["Referer"], "https://example.com/")
        self.assertEqual(headers["User-Agent"], "CustomUA")

        # Verify command substitution: 'python' -> sys.executable
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], sys.executable)
        self.assertEqual(call_args[-1], "https://example.com/video/1")

    @patch("subprocess.run")
    def test_non_zero_exit_raises(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Browser closed unexpectedly"
        mock_run.return_value = mock_proc

        resolver = {"match": "example.com", "command": ["python", "script.py"]}
        with self.assertRaises(RuntimeError) as ctx:
            LS.resolve_external("https://example.com/video", {}, resolver)
        self.assertIn("External resolver failed", str(ctx.exception))

    @patch("subprocess.run")
    def test_invalid_json_raises(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Not JSON"
        mock_run.return_value = mock_proc

        resolver = {"match": "example.com", "command": ["python", "script.py"]}
        with self.assertRaises(RuntimeError) as ctx:
            LS.resolve_external("https://example.com/video", {}, resolver)
        self.assertIn("Invalid JSON", str(ctx.exception))

    @patch("subprocess.run")
    def test_missing_url_raises(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({"error": "Video removed by author"})
        mock_run.return_value = mock_proc

        resolver = {"match": "example.com", "command": ["python", "script.py"]}
        with self.assertRaises(RuntimeError) as ctx:
            LS.resolve_external("https://example.com/video", {}, resolver)
        self.assertIn("Video removed by author", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
