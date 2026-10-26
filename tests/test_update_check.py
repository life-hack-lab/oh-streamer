import json
import os
import threading
import tomllib
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from oh_streamer import update_check
from oh_streamer.queue.queue_app import Api
from oh_streamer.version import __version__

ROOT = Path(__file__).resolve().parents[1]


def serve(code, body):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class UpdateCheckTests(unittest.TestCase):
    def run_check(self, code, body):
        srv = serve(code, body)
        try:
            with mock.patch.dict(os.environ, {"OH_STREAMER_UPDATE_URL": f"http://127.0.0.1:{srv.server_port}/x"}):
                return update_check.check()
        finally:
            srv.shutdown()
            srv.server_close()

    def test_version_matches_pyproject(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["project"]["version"], __version__)

    def test_update(self):
        r = self.run_check(200, {"tag_name": "v9.0.0", "body": "x" * 5000,
                                 "html_url": "https://github.com/a/b/releases/tag/v9.0.0"})
        self.assertEqual(r["status"], "update")
        self.assertEqual(r["latest"], "9.0.0")
        self.assertEqual(r["current"], __version__)
        self.assertLessEqual(len(r["notes"]), update_check.NOTES_MAX + 1)
        self.assertTrue(r["url"].startswith("https://github.com/"))

    def test_update_drops_foreign_url(self):
        r = self.run_check(200, {"tag_name": "9.0.0", "html_url": "https://evil.example/x"})
        self.assertEqual((r["status"], r["url"]), ("update", ""))

    def test_latest(self):
        r = self.run_check(200, {"tag_name": "v" + __version__})
        self.assertEqual(r["status"], "latest")

    def test_no_release(self):
        self.assertEqual(self.run_check(404, {"message": "Not Found"})["status"], "no_release")
        self.assertEqual(self.run_check(200, {"tag_name": "v9.0.0", "draft": True})["status"], "no_release")
        self.assertEqual(self.run_check(200, {"tag_name": "v9.0.0", "prerelease": True})["status"], "no_release")

    def test_errors(self):
        self.assertEqual(self.run_check(500, {})["status"], "error")
        r = self.run_check(200, b"not json")
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["message"])
        with mock.patch.dict(os.environ, {"OH_STREAMER_UPDATE_URL": "http://127.0.0.1:1/x"}):
            self.assertEqual(update_check.check()["status"], "error")

    def test_compare(self):
        n = update_check.is_newer
        self.assertTrue(n("v0.2.0", "0.1.0"))
        self.assertTrue(n("0.10.0", "0.9.9"))
        self.assertFalse(n("0.1.0", "0.1.0"))
        self.assertFalse(n("0.1", "0.1.0"))
        self.assertFalse(n("0.1.0-rc1", "0.1.0"))
        self.assertTrue(n("0.1.0", "0.1.0-rc1"))
        self.assertTrue(n("0.1.0-rc2", "0.1.0-rc1"))
        self.assertFalse(n("garbage", "0.1.0"))

    def test_open_release_page_restricts_url(self):
        api = Api.__new__(Api)
        with mock.patch("webbrowser.open") as op:
            self.assertFalse(api.open_release_page("http://github.com/x")["ok"])
            self.assertFalse(api.open_release_page("https://github.com.evil/x")["ok"])
            self.assertFalse(api.open_release_page("file:///c:/x")["ok"])
            self.assertTrue(api.open_release_page("https://github.com/a/b/releases")["ok"])
            op.assert_called_once_with("https://github.com/a/b/releases")


if __name__ == "__main__":
    unittest.main()
