"""atomic_write_* ride out a transient Windows sharing violation on os.replace (virus scanner holding the temp file)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles import runtime


class AtomicWriteRetryTests(unittest.TestCase):
    def test_transient_permission_error_is_retried(self):
        real = runtime.os.replace
        calls = []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError(13, "in use")
            return real(src, dst)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runtime.os, "name", "nt"), \
                mock.patch.object(runtime.os, "replace", flaky), mock.patch.object(runtime.time, "sleep"):
            target = Path(tmp) / "a.json"
            runtime.atomic_write_json(target, {"k": 1})
            self.assertEqual(len(calls), 3)
            self.assertIn('"k": 1', target.read_text("utf-8"))

    def test_persistent_permission_error_still_raises_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runtime.os, "name", "nt"), \
                mock.patch.object(runtime.os, "replace", side_effect=PermissionError(13, "x")), \
                mock.patch.object(runtime.time, "sleep"):
            with self.assertRaises(PermissionError):
                runtime.atomic_write_text(Path(tmp) / "b.txt", "x")
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
