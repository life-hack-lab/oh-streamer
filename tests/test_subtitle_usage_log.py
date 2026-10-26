import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from oh_streamer.subtitles import model_adapters as ma
from oh_streamer.subtitles import usage_log as ul
from oh_streamer.subtitles.providers.http import ProviderError

SRC = str(Path(__file__).resolve().parents[1] / "src")
KEY = "sk-SECRET-KEY-123"
TEXT = "こんにちは字幕テキスト"
MODEL = {"id": "groq-whisper", "remote_model": "whisper-large-v3", "base_url": "https://api.groq.com/openai/v1",
         "api_family": "openai"}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = [1000.0]
        ul._dir_override = Path(self.tmp.name)
        ul._clock = lambda: self.now[0]
        ul.reset()
        self.addCleanup(self._restore)
        ul.run_start("item1", executor="cloud", duration_s=7200)
        self.path = ul.get("item1").path

    def _restore(self):
        import time
        ul._dir_override = None
        ul._clock = time.time
        ul.reset()

    def lines(self):
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()]


class RequestTests(Base):
    def test_request_line_fields_and_redaction(self):
        def transport(req, timeout):
            return 200, {}, json.dumps({"text": TEXT, "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                                                   "total_tokens": 15}}).encode()
        with ul.context("item1", slice_index=3, stage="asr", candidate="cand-a"):
            ul.set_audio_seconds(300)
            ma._request(MODEL, "/audio/transcriptions", b"AUDIO", KEY, "multipart/form-data", transport=transport)
        rec = [x for x in self.lines() if x["ev"] == "request"][0]
        self.assertEqual((rec["kind"], rec["host"], rec["model"], rec["status"], rec["audio_s"], rec["slice"]),
                         ("asr", "api.groq.com", "whisper-large-v3", 200, 300.0, 3))
        self.assertEqual((rec["prompt_tokens"], rec["completion_tokens"], rec["total_tokens"]), (10, 5, 15))
        self.assertEqual(rec["candidate"], "cand-a")
        raw = self.path.read_text(encoding="utf-8")
        for secret in (KEY, TEXT, "AUDIO", "Bearer"):
            self.assertNotIn(secret, raw)

    def test_error_line_has_status_and_retry_after(self):
        def boom():
            raise ProviderError("rate_limit", "x", retry_after=12.5, http_status=429)
        ticks = iter([0.0, 2.0])
        with ul.context("item1", slice_index=1, stage="mt"):
            with self.assertRaises(ProviderError):
                ul.timed_request(MODEL, "/chat/completions", boom, now=lambda: next(ticks))
        rec = [x for x in self.lines() if x["ev"] == "request"][0]
        self.assertEqual((rec["kind"], rec["status"], rec["retry_after_s"], rec["latency_s"], rec["error"]),
                         ("mt", 429, 12.5, 2.0, "rate_limit"))

    def test_no_context_no_line(self):
        ul.timed_request(MODEL, "/chat/completions", lambda: {})
        self.assertEqual([x["ev"] for x in self.lines()], ["run_start"])


class OtherEventTests(Base):
    def test_slice_and_run_lines(self):
        ul.slice_published("item1", track="primary", slice_index=2, start=60, end=120, cues=14, candidate="c1")
        ul.run_end("item1", executor="cloud", reason="complete")
        ls = self.lines()
        self.assertEqual(ls[0]["video_duration_s"], 7200.0)
        self.assertEqual((ls[1]["slice"], ls[1]["cues"], ls[1]["candidate"]), (2, 14, "c1"))
        self.assertEqual(ls[2]["ev"], "run_end")

    def test_view_sampling_throttle(self):
        v = {"state": "running", "covered_s": 100}
        ul.view_sample("item1", v, play_pos=None)
        ul.view_sample("item1", v, play_pos=5)                 # same state, <30 s: skipped
        self.now[0] += 31
        ul.view_sample("item1", v, play_pos=40, restore_end=50, download_end=900)
        ul.view_sample("item1", {"state": "stuck", "covered_s": 100}, play_pos=41)   # state change: logged
        views = [x for x in self.lines() if x["ev"] == "view"]
        self.assertEqual(len(views), 3)
        self.assertIsNone(views[0]["play_pos_s"])
        self.assertEqual((views[1]["play_pos_s"], views[1]["restore_end_s"], views[1]["download_end_s"]), (40.0, 50.0, 900.0))
        self.assertEqual(views[2]["state"], "stuck")

    def test_failure_tolerance(self):
        ul.get("item1").path = Path(self.tmp.name) / "no" / "\0bad"
        ul.slice_published("item1", track="p", slice_index=0, start=0, end=1, cues=1)
        ul._dir_override = Path(self.tmp.name) / "\0bad"
        ul.reset()
        ul.run_start("other")
        ul.view_sample("other", {"state": "x"}, play_pos=None)
        ul.run_end("other")
        # unreadable http result shape
        with ul.context("item1"):
            self.assertEqual(ul.timed_request({"base_url": 5}, "/x", lambda: [1]), [1])


class PruneTests(Base):
    def test_prune_count_and_size(self):
        d = Path(self.tmp.name) / "p"
        d.mkdir()
        for i in range(6):
            f = d / f"{i}.jsonl"
            f.write_text("x" * 10)
            os.utime(f, (i, i))
        ul.prune(d, max_files=4, max_bytes=1000)
        self.assertEqual(sorted(p.name for p in d.glob("*.jsonl")), ["2.jsonl", "3.jsonl", "4.jsonl", "5.jsonl"])
        ul.prune(d, max_files=50, max_bytes=25)
        self.assertEqual(sorted(p.name for p in d.glob("*.jsonl")), ["4.jsonl", "5.jsonl"])

    def test_new_run_prunes_oldest(self):
        for i in range(ul.MAX_FILES + 5):
            f = Path(self.tmp.name) / f"old{i:03d}.jsonl"
            f.write_text("{}")
            os.utime(f, (10 + i, 10 + i))
        ul.run_start("item2")
        files = list(Path(self.tmp.name).glob("*.jsonl"))
        self.assertLessEqual(len(files), ul.MAX_FILES)
        self.assertTrue(ul.get("item2").path.exists())
        self.assertFalse((Path(self.tmp.name) / "old000.jsonl").exists())


class ChildProcessTests(Base):
    def test_child_appends_to_same_file(self):
        code = ("import sys; from oh_streamer.subtitles.usage_log import UsageLog;"
                "UsageLog(sys.argv[1], 'item1').write({'ev':'child','n':1})")
        env = dict(os.environ, PYTHONPATH=SRC)
        subprocess.run([sys.executable, "-c", code, str(self.path)], check=True, env=env)
        ul.slice_published("item1", track="p", slice_index=0, start=0, end=1, cues=1)
        evs = [x["ev"] for x in self.lines()]
        self.assertEqual(evs, ["run_start", "child", "slice"])


if __name__ == "__main__":
    unittest.main()
