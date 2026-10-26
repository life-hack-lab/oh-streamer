"""Focused tests for casting already-produced progressive queue segments.

The fixture creates small Matroska segments with ffmpeg and exercises the same RunInput contract
used by hls_cast.  No restoration models or GPU are loaded.
"""

from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from unittest import mock

try:
    import av
except ImportError:  # pragma: no cover - the runtime's bundled dependency environment has PyAV
    av = None

from oh_streamer.player import progressive_stream as P  # noqa: E402
from oh_streamer.player.progressive_cast import ProgressiveCastSource  # noqa: E402
from oh_streamer.platform import hls_cast as H  # noqa: E402


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
HAVE_MEDIA_TOOLS = bool(FFMPEG and FFPROBE and av is not None)
_TMP_ROOT = _STREAMER_ROOT.parent / "testdata" / "progressive-cast-tmp"


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def write_manifest(work, segments, *, last_k, total_frames, count, fps="10/1", n=20, video_start=0.5,
                   rate_mult=2):
    doc = {
        "version": 2,
        "segments": segments,
        "last_k": last_k,
        "complete": last_k is not None,
        "total_frames": total_frames,
        "play": {
            "fps": str(fps),
            "n": n,
            "video_start": video_start,
            "rate_mult": rate_mult,
            "layers": ["processed", "original"],
            "count": count,
        },
    }
    tmp = os.path.join(work, "segments.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, os.path.join(work, "segments.json"))


@unittest.skipUnless(HAVE_MEDIA_TOOLS, "ffmpeg, ffprobe, and PyAV are required")
class ProgressiveCastSourceTest(unittest.TestCase):
    def setUp(self):
        _TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="progressive_cast_test_", dir=str(_TMP_ROOT))
        self.source_path = os.path.join(self.tmp, "source.mkv")
        self.work = os.path.join(self.tmp, "work")
        self.seg_dir = os.path.join(self.work, "segments")
        os.makedirs(self.seg_dir)
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=c=black:s=32x24:r=10:d=4", "-f", "lavfi", "-i",
             "sine=frequency=440:sample_rate=48000:duration=4", "-map", "0:v:0", "-map", "1:a:0",
             "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", self.source_path],
            check=True,
        )
        self._make_segment(0)
        self.info = {"duration": 4.0, "fps": Fraction(10), "video_start": 0.5,
                     "frames": 40, "has_audio": True}
        self.playback = _FakePlayback(self.source_path, self.work, self.info)
        self.source = None

    def tearDown(self):
        if self.source is not None:
            self.source.stop()
        self.playback.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_segment(self, k, duration=2):
        path = os.path.join(self.seg_dir, f"seg_{k:06d}.mkv")
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color=c=red:s=32x24:r=20:d={duration}",
             "-f", "lavfi", "-i", f"color=c=blue:s=32x24:r=20:d={duration}",
             "-map", "0:v:0", "-map", "1:v:0", "-an", "-t", str(duration),
             "-c:v", "libx264", "-preset", "ultrafast", "-bf", "0", "-pix_fmt", "yuv420p",
             "-metadata:s:v:0", "title=oh:original", "-metadata:s:v:1", "title=oh:processed", path],
            check=True,
        )
        return path

    def _start_source(self, *, ks=(0,), final=0, fps=Fraction(10), n=20, video_start=0.5, rate_mult=2,
                      frames_per_segment=20):
        segments = [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": frames_per_segment} for k in ks]
        write_manifest(self.work, segments, last_k=final,
                       total_frames=(final + 1) * frames_per_segment if final is not None else None,
                       count=(final + 1 if final is not None else 3), fps=fps, n=n,
                       video_start=video_start, rate_mult=rate_mult)
        self.source = ProgressiveCastSource(self.playback)
        return self.source

    def test_selects_processed_track_uses_output_fps_and_trims_seek_exactly(self):
        source = self._start_source()
        self.assertTrue(source.is_progressive_cast)
        self.assertEqual(source.meta.video_fps_exact, Fraction(20))  # source 10 fps, interpolated 2x
        self.assertEqual(source.meta.frames_count, 40)  # the known manifest tail clamps the source probe estimate

        inp = source.run_input(0.5 + 0.451)
        try:
            self.assertEqual(inp.audio, {"src": self.source_path, "args": [], "fmt_start": source.fmt_start})
            frames = list(inp.frames)
        finally:
            inp.frames.close()

        self.assertEqual(len(frames), 31)
        frame, timestamp = frames[0]
        self.assertAlmostEqual(timestamp, 0.951, places=6)  # nearest frame is normalized to the requested seek
        self.assertGreater(float(frame[:, :, 0].mean()), 180)  # blue BGR channel: oh:processed
        self.assertLess(float(frame[:, :, 2].mean()), 80)      # original track is red
        self.assertAlmostEqual(frames[1][1], 1.0, places=6)

    def test_gap_waits_requests_repeatedly_and_close_unblocks_iterator(self):
        self._make_segment(2)
        source = self._start_source(ks=(0, 2), final=2)
        inp = source.run_input(0.5)
        outcome = []

        def consume():
            try:
                outcome.extend(list(inp.frames))
            except Exception as exc:  # exposed to the test thread
                outcome.append(exc)

        with mock.patch.object(P, "WANT_INTERVAL", 0.1):
            thread = threading.Thread(target=consume, daemon=True)
            thread.start()
            self.assertTrue(wait_for(lambda: source.waiting_k == 1))
            self.assertTrue(wait_for(lambda: self.playback.wants.count(1) >= 2, 2.0))
            self.assertEqual(source.waiting_k, 1)
            inp.frames.close()
            thread.join(2.0)

        self.assertFalse(thread.is_alive())
        self.assertIsNone(source.waiting_k)
        self.assertTrue(any(k == 1 for k in self.playback.wants))
        self.assertFalse(any(isinstance(item, Exception) for item in outcome))

    def test_gap_arrival_continues_through_final_tail(self):
        self.info = {"duration": 6.0, "fps": Fraction(10), "video_start": 0.5,
                     "frames": 60, "has_audio": True}
        self.playback = _FakePlayback(self.source_path, self.work, self.info)
        self._make_segment(2)
        write_manifest(self.work,
                       [{"k": 0, "file": "seg_000000.mkv", "frames": 20},
                        {"k": 2, "file": "seg_000002.mkv", "frames": 20}],
                       last_k=None, total_frames=None, count=3)
        source = ProgressiveCastSource(self.playback)
        self.source = source
        inp = source.run_input(0.5)
        result = []
        failures = []

        def consume():
            try:
                result.extend(list(inp.frames))
            except Exception as exc:  # make thread errors visible in the test
                failures.append(exc)

        with mock.patch.object(P, "WANT_INTERVAL", 0.1):
            thread = threading.Thread(target=consume, daemon=True)
            thread.start()
            self.assertTrue(wait_for(lambda: source.waiting_k == 1))
            self.assertTrue(wait_for(lambda: self.playback.wants.count(1) >= 2, 2.0))
            self._make_segment(1)
            write_manifest(self.work,
                           [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": 20} for k in range(3)],
                           last_k=2, total_frames=60, count=3)
            thread.join(5.0)
        inp.frames.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(result), 120)
        self.assertAlmostEqual(result[0][1], 0.5, places=6)
        self.assertAlmostEqual(result[-1][1], 6.45, places=6)
        self.assertTrue(all(result[i][1] < result[i + 1][1] for i in range(len(result) - 1)))
        self.assertTrue(any(k == 1 for k in self.playback.wants))
        self.assertEqual(source.meta.frames_count, 120)

    def test_legacy_single_untitled_video_track_is_accepted(self):
        path = os.path.join(self.seg_dir, "seg_000000.mkv")
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=32x24:r=20:d=2", "-an", "-c:v", "libx264", "-preset", "ultrafast",
             "-bf", "0", "-pix_fmt", "yuv420p", path],
            check=True,
        )
        source = self._start_source()
        frames = source.run_input(0.5)
        try:
            decoded = list(frames.frames)
        finally:
            frames.frames.close()
        self.assertEqual(len(decoded), 40)
        self.assertGreater(float(decoded[0][0][:, :, 0].mean()), 180)

    def test_close_after_first_frame_closes_iterator_owned_segment(self):
        source = self._start_source()
        inp = source.run_input(0.5)
        iterator = iter(inp.frames)
        next(iterator)
        self.assertIs(iter(iterator), iterator)
        inp.frames.close()
        self.assertFalse(source._active)
        original = os.path.join(self.seg_dir, "seg_000000.mkv")
        renamed = original + ".closed"
        os.replace(original, renamed)
        os.replace(renamed, original)

    def test_hls_audio_survives_gap_then_restart_seek_aligns_to_video(self):
        long_source = os.path.join(self.tmp, "source_10s.mkv")
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=c=black:s=32x24:r=10:d=10", "-f", "lavfi", "-i",
             "sine=frequency=440:sample_rate=48000:duration=10", "-map", "0:v:0", "-map", "1:a:0",
             "-t", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", long_source],
            check=True,
        )
        self.info = {"duration": 9.0, "fps": Fraction(10), "video_start": 0.5,
                     "frames": 90, "has_audio": True}
        self.playback = _FakePlayback(long_source, self.work, self.info)
        self._make_segment(0, duration=3)
        self._make_segment(2, duration=3)
        source = self._start_source(ks=(0, 2), final=None, n=30, video_start=0.5,
                                    frames_per_segment=30)
        plan = H.SegmentPlan(source.meta.video_fps_exact, source.meta.frames_count,
                             source.meta.video_width, source.meta.video_height,
                             video_start=source.target, has_audio=True)
        self.assertEqual(plan.count, 3)
        logs = []
        vod = H.HlsVod(plan, source.run_input, "libx264", start_k=0,
                       cache_root=os.path.join(self.tmp, "hls_gap_cache"), log=logs.append,
                       request_timeout=20)
        try:
            first_result = []
            thread = threading.Thread(target=lambda: first_result.append(vod.get_segment(0)), daemon=True)
            thread.start()
            self.assertTrue(wait_for(lambda: source.waiting_k == 1))
            self._make_segment(1, duration=3)
            write_manifest(self.work,
                           [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": 30} for k in range(3)],
                           last_k=2, total_frames=90, count=3, n=30)
            thread.join(25.0)
            self.assertFalse(thread.is_alive())
            self.assertTrue(first_result)
            first_data, first_reason = first_result[0]
            self.assertIsNotNone(first_data, first_reason)
            self._assert_audio_video_alignment(first_data, 0)

            # Quality/model changes use invalidate() and restart the provider at an HLS
            # boundary. At 4s this boundary falls inside queue segment 1 (which spans 3-6s).
            vod.invalidate(at_k=1)
            vod.prefetch(1)
            data, reason = vod.get_segment(1)
            self.assertIsNotNone(data, f"{reason}; last_error={vod.last_error!r}; "
                                        f"source_error={source.error!r}; logs={logs[-8:]!r}")
            self.assertGreaterEqual(vod.status()["restarts"], 1)
            self._assert_audio_video_alignment(data, 4)
        finally:
            vod.close()

    def _assert_audio_video_alignment(self, data, segment_start):
        with av.open(io.BytesIO(data), format="mpegts") as container:
            video = container.streams.video[0]
            video_packets = [p for p in container.demux(video) if p.pts is not None and p.size]
        # Open a second demuxer so each stream can be collected independently.
        with av.open(io.BytesIO(data), format="mpegts") as container:
            audio = container.streams.audio[0]
            audio_packets = [p for p in container.demux(audio) if p.pts is not None and p.size]
        self.assertTrue(video_packets)
        self.assertTrue(audio_packets)
        expected = H.TS_BASE + H.MPEGTS_DELAY + segment_start
        video_start = float(Fraction(video_packets[0].pts) * video_packets[0].time_base)
        audio_start = float(Fraction(audio_packets[0].pts) * audio_packets[0].time_base)
        self.assertAlmostEqual(video_start, expected, delta=0.08)
        self.assertAlmostEqual(audio_start, expected, delta=0.15)

    def test_paused_wait_continues_then_failed_queue_state_surfaces(self):
        source = self._start_source(ks=(0, 2), final=2)
        self.playback.hold = "paused"
        inp = source.run_input(0.5)
        errors = []

        def consume():
            try:
                list(inp.frames)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        self.assertTrue(wait_for(lambda: source.waiting_k == 1))
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        self.assertIsNone(source.error)

        self.playback.hold = "failed"
        source.tick()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertIsInstance(source.error, Exception)
        self.assertIn("failed", str(source.error))

    def test_stop_stops_only_the_local_player_feeder(self):
        self._start_source()
        self.source.stop()
        self.source.stop()
        self.assertEqual(self.playback.stop_calls, 2)

    def test_fractional_output_rate_hls_duration_and_seek_timestamps(self):
        source_fps = Fraction(30000, 1001)
        output_fps = source_fps * 2
        fractional_segment = os.path.join(self.seg_dir, "seg_000000.mkv")
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "color=c=blue:s=32x24:r=60000/1001", "-frames:v", "240", "-an", "-c:v", "libx264",
             "-preset", "ultrafast", "-bf", "0", "-pix_fmt", "yuv420p", "-metadata:s:v:0",
             "title=oh:processed", fractional_segment],
            check=True,
        )
        self.info = {"duration": float(Fraction(120, 1) / source_fps), "fps": source_fps,
                     "video_start": 0.5, "frames": 120, "has_audio": False}
        source = self._start_source(fps=source_fps, n=120, frames_per_segment=120)
        self.assertEqual(source.meta.video_fps_exact, output_fps)
        source.has_audio = False

        seek = source.run_input(0.5 + 0.451)
        try:
            _, first_s = next(iter(seek.frames))
        finally:
            seek.frames.close()
        expected_frame = round(Fraction(451, 1000) * output_fps)
        self.assertAlmostEqual(first_s, 0.5 + 0.451, places=6)
        self.assertLessEqual(abs(Fraction(expected_frame, 1) / output_fps - Fraction(451, 1000)),
                             Fraction(1, 2) / output_fps)

        plan = H.SegmentPlan(source.meta.video_fps_exact, source.meta.frames_count,
                             source.meta.video_width, source.meta.video_height, video_start=source.target,
                             has_audio=False)
        vod = H.HlsVod(plan, source.run_input, "libx264", start_k=0,
                       cache_root=os.path.join(self.tmp, "hls_cache"), log=lambda _message: None)
        try:
            data, reason = vod.get_segment(0)
            self.assertIsNotNone(data, reason)
            with av.open(io.BytesIO(data), format="mpegts") as container:
                stream = container.streams.video[0]
                decoded = list(container.decode(stream))
                self.assertEqual(len(decoded), 240)
                self.assertAlmostEqual(float(stream.average_rate), float(output_fps), places=2)
                first = Fraction(decoded[0].pts) * decoded[0].time_base
                last = Fraction(decoded[-1].pts) * decoded[-1].time_base
                self.assertAlmostEqual(float(last - first), float(Fraction(239, 1) / output_fps), delta=0.02)
            self.assertAlmostEqual(float(plan.total_duration), float(Fraction(240, 1) / output_fps), places=9)
        finally:
            vod.close()


class _FakePlayback:
    def __init__(self, source_path, work, info):
        self.po = P.ProgressiveOpen(source_path, work, "queue-1", "test", info)
        self.watcher = P.ManifestWatcher(work, P.play_from_probe(info), interval=0.05).start()
        self.hold = None
        self.wants = []
        self.stop_calls = 0

    def want(self, k):
        self.wants.append(k)

    def stop_feeder(self):
        self.stop_calls += 1

    def close(self):
        self.watcher.close()


if __name__ == "__main__":
    unittest.main()
