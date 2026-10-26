r"""run_self_check: reads the last stdout line of a torch check snippet; ok needs CUDA and deform_conv2d."""
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env.selfcheck import SNIPPET, run_self_check


def _printing(payload, noise=""):
    lines = [f"print({noise!r})"] if noise else []
    lines.append(f"print({json.dumps(payload)!r})")
    return "\n".join(lines)


class RunSelfCheckTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is None, "this interpreter has torch installed")
    def test_default_snippet_fails_in_base_env_without_torch(self):
        res = run_self_check(sys.executable, timeout_s=120)
        self.assertFalse(res.ok)
        self.assertIn("torch", res.message.lower())

    def test_ok_when_cuda_and_deform_conv2d_pass(self):
        payload = {"cuda_available": True, "device_name": "Fake GPU", "deform_conv2d_ok": True, "message": ""}
        res = run_self_check(sys.executable, snippet=_printing(payload, noise="UserWarning: noisy"))
        self.assertTrue(res.ok)
        self.assertTrue(res.cuda_available)
        self.assertTrue(res.deform_conv2d_ok)
        self.assertEqual(res.device_name, "Fake GPU")

    def test_not_ok_without_cuda(self):
        payload = {"cuda_available": False, "device_name": None, "deform_conv2d_ok": False, "message": "CUDA 不可用"}
        res = run_self_check(sys.executable, snippet=_printing(payload))
        self.assertFalse(res.ok)
        self.assertFalse(res.cuda_available)
        self.assertEqual(res.message, "CUDA 不可用")

    def test_not_ok_when_deform_conv2d_fails(self):
        payload = {"cuda_available": True, "device_name": "Fake GPU", "deform_conv2d_ok": False, "message": "boom op"}
        res = run_self_check(sys.executable, snippet=_printing(payload))
        self.assertFalse(res.ok)
        self.assertTrue(res.cuda_available)
        self.assertIn("boom op", res.message)

    def test_illegal_last_line_is_not_ok(self):
        res = run_self_check(sys.executable, snippet="print('{not json')")
        self.assertFalse(res.ok)
        self.assertIn("不是合法 JSON", res.message)

    def test_nonzero_exit_reports_stderr(self):
        snippet = "import sys\nsys.stderr.write('boom-marker\\n')\nsys.exit(3)"
        res = run_self_check(sys.executable, snippet=snippet)
        self.assertFalse(res.ok)
        self.assertIn("boom-marker", res.message)
        self.assertIn("3", res.message)

    @unittest.skipUnless(sys.platform == "win32", "CREATE_NO_WINDOW is Windows-only")
    def test_child_gets_no_console_window(self):
        from unittest import mock
        with mock.patch("oh_streamer.runtime_env.selfcheck.subprocess.run", wraps=__import__("subprocess").run) as run:
            run_self_check(sys.executable, snippet=_printing({"cuda_available": False}))
        self.assertEqual(run.call_args.kwargs["creationflags"] & 0x08000000, 0x08000000)

    def test_timeout_returns_promptly_and_is_not_ok(self):
        started = time.monotonic()
        res = run_self_check(sys.executable, timeout_s=2, snippet="import time\ntime.sleep(30)")
        elapsed = time.monotonic() - started
        self.assertFalse(res.ok)
        self.assertIn("逾時", res.message)
        self.assertLess(elapsed, 20)


if __name__ == "__main__":
    unittest.main()
