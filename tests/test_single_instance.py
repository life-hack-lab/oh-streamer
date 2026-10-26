"""Lock ownership and authenticated localhost forwarding; all state stays in temporary directories."""
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
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import oh_streamer.player.single_instance as si


class InstanceTests(unittest.TestCase):
    def test_posix_lock_busy_and_close_release_with_mock_backend(self):
        flock = Mock()
        fcntl = types.SimpleNamespace(flock=flock, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8)
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys, "platform", "darwin"), \
                patch.dict(sys.modules, {"fcntl": fcntl}), \
                patch.object(si, "_private_state_dir", return_value=Path(tmp)), \
                patch.object(si, "state_dir", return_value=Path(tmp)), \
                patch.object(os, "fchmod", create=True):
            first = si.acquire("test")
            self.assertIsNotNone(first)
            flock.assert_called_once()
            flock.side_effect = BlockingIOError()
            self.assertIsNone(si.acquire("test"))
            flock.side_effect = None
            first.close()
            self.assertEqual(flock.call_args.args[1], fcntl.LOCK_UN)
            first.close()

    def test_forwarding_and_token_rejection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(si, "state_dir", return_value=Path(tmp)):
            inst = si.Instance("test", None)
            messages = []
            try:
                inst.serve(lambda msg: messages.append(msg) or {"ok": True})
                self.assertEqual(si.forward("test", {"input": "影片.mp4"}, timeout=1), {"ok": True})
                self.assertEqual(messages, [{"input": "影片.mp4"}])
                info = json.loads(si.info_path("test").read_text())
                with socket.create_connection(("127.0.0.1", info["port"]), timeout=1) as conn:
                    conn.sendall(b'{"token":"bad","input":"untrusted"}\n')
                    self.assertEqual(conn.recv(100), b"")
                self.assertEqual(len(messages), 1)
            finally:
                inst.close()
            self.assertFalse(si.info_path("test").exists())

    @unittest.skipIf(sys.platform == "win32", "real POSIX flock requires POSIX")
    def test_posix_exclusion_release_and_private_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(si, "state_dir", return_value=Path(tmp) / "state"):
            first = si.acquire("test")
            try:
                self.assertIsNone(si.acquire("test"))
                first.serve(lambda msg: {"ok": True})
                self.assertEqual(si.state_dir().stat().st_mode & 0o777, 0o700)
                self.assertEqual(si.info_path("test").stat().st_mode & 0o777, 0o600)
            finally:
                first.close()
            second = si.acquire("test")
            self.assertIsNotNone(second)
            second.close()


if __name__ == "__main__":
    unittest.main()
