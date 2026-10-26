"""HTTP error diagnostics must preserve status while excluding credentials and URLs."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform.airplay_cast import failure_details, playback_status_rejected, short_reason


class DiagnosticsTest(unittest.TestCase):
    def error(self, uri="/pair-verify", code=403, module="pyatv.support.http", method="POST"):
        namespace = {"__name__": module, "SimpleNamespace": SimpleNamespace}
        exec("def send_and_receive(uri, code, method):\n"
             "    headers = {'Authorization': 'SECRET'}\n"
             "    body = b'PRIVATE CREDENTIALS'\n"
             "    response = SimpleNamespace(code=code)\n"
             "    raise RuntimeError('not authenticated')\n", namespace)
        try:
            namespace["send_and_receive"](uri, code, method)
        except RuntimeError as error:
            return error

    def test_status_and_endpoint_without_secrets(self):
        for code in (401, 403, 470):
            self.assertEqual(failure_details(self.error(code=code)),
                             f"request=POST /pair-verify status={code}")

    def test_private_urls_and_unknown_frames_are_redacted(self):
        self.assertEqual(failure_details(self.error("http://host/SECRET/movie?token=PRIVATE")),
                         "request=POST <redacted> status=403")
        self.assertEqual(failure_details(self.error(module="unrelated")), "HTTP metadata unavailable")

    def test_wrapped_exception(self):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = self.error()
        self.assertIn("/pair-verify status=403", failure_details(wrapper))
        wrapper.__cause__.__cause__ = wrapper
        self.assertEqual(failure_details(wrapper).count("status=403"), 1)

    def test_playback_status_rejection_is_not_a_pairing_error(self):
        error = self.error("/playback-info", method="GET")
        self.assertTrue(playback_status_rejected(error))
        self.assertIn("播放狀態查詢", short_reason(error))
        self.assertNotIn("重新配對", short_reason(error))
        self.assertFalse(playback_status_rejected(self.error()))
        self.assertFalse(playback_status_rejected(self.error("/playback-info", code=401, method="GET")))


if __name__ == "__main__":
    unittest.main()
