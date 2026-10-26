"""Tests for the video-span tail probe of queue_download._range_metadata (WP0): webm / fMP4+global sidx report an
absolute end time as container duration. Clips are generated with ffmpeg inside the tests.
Run from streamer/: python -m unittest tests.test_queue_download_span"""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_STREAMER_ROOT, _STREAMER_ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import oh_streamer.queue.queue_download as QD  # noqa: E402
from oh_streamer.sources import range_cache  # noqa: E402

FPS, SECONDS = 24, 6
FRAMES = FPS * SECONDS


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def _video_in(seconds=SECONDS):
    return ["-f", "lavfi", "-i", f"testsrc=size=160x120:rate={FPS}:duration={seconds}"]


def _audio_in(seconds):
    return ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]


X264 = ["-c:v", "libx264", "-preset", "ultrafast", "-bf", "2", "-pix_fmt", "yuv420p"]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not available")
class SpanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="span-test-")
        t = _Path(cls.tmp)
        cls.files = {
            "mp4_head": t / "head.mp4", "mp4_tail": t / "tail.mp4", "webm": t / "a.webm",
            "fmp4_sidx": t / "sidx.mp4", "long_audio": t / "longaudio.mp4"}
        _ffmpeg(*_video_in(), *X264, "-movflags", "+faststart", str(cls.files["mp4_head"]))
        _ffmpeg(*_video_in(), *X264, str(cls.files["mp4_tail"]))
        _ffmpeg(*_video_in(), "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "200k",
                str(cls.files["webm"]))
        _ffmpeg(*_video_in(), *X264, "-movflags", "+frag_keyframe+empty_moov+default_base_moof+global_sidx",
                str(cls.files["fmp4_sidx"]))
        _ffmpeg(*_video_in(4), *_audio_in(SECONDS), *X264, "-c:a", "aac", str(cls.files["long_audio"]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def meta(self, key):
        path = str(self.files[key])
        probe = QD._probe_range_url(path, threading.Event())
        return probe, QD._range_metadata(probe, path, threading.Event())

    def span_frames(self, md):
        return float(md["format"]["duration"]) * FPS

    def test_mp4_moov_head_unchanged(self):
        _, md = self.meta("mp4_head")
        self.assertAlmostEqual(self.span_frames(md), FRAMES, delta=0.05)

    def test_mp4_moov_tail_unchanged(self):
        _, md = self.meta("mp4_tail")
        self.assertAlmostEqual(self.span_frames(md), FRAMES, delta=0.05)

    def test_webm_absolute_end_corrected(self):
        probe, md = self.meta("webm")
        old = float(QD._range_metadata(probe)["format"]["duration"])
        start = float(md["streams"][0].get("start_time") or 0)
        self.assertAlmostEqual(self.span_frames(md), FRAMES, delta=0.05)
        if start > 0:        # the container duration included the start offset: the old value was too long
            self.assertGreater(old * FPS, FRAMES + 0.05)

    def test_fmp4_global_sidx_corrected(self):
        probe, md = self.meta("fmp4_sidx")
        self.assertAlmostEqual(self.span_frames(md), FRAMES, delta=0.05)
        self.assertEqual(round(self.span_frames(md)), FRAMES)

    def test_audio_longer_than_video(self):
        probe, md = self.meta("long_audio")
        self.assertGreater(float(probe["format"]["duration"]), 5.5)
        self.assertAlmostEqual(self.span_frames(md), 4 * FPS, delta=0.05)

    def test_probe_failure_falls_back_with_log(self):
        probe = QD._probe_range_url(str(self.files["webm"]), threading.Event())
        expected = QD._range_metadata(probe)["format"]["duration"]
        logs = []
        with mock.patch.object(QD, "log", logs.append):
            md = QD._range_metadata(probe, str(self.files["webm"]) + ".missing", threading.Event())
        self.assertEqual(md["format"]["duration"], expected)
        self.assertTrue(any("tail span" in line for line in logs))

    def test_cancel_propagates(self):
        probe = QD._probe_range_url(str(self.files["mp4_head"]), threading.Event())
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(QD, "_tail_span", side_effect=QD.DownloadCancelled()):
            with self.assertRaises(QD.DownloadCancelled):
                QD._range_metadata(probe, "x", cancel)


if __name__ == "__main__":
    unittest.main()
