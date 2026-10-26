"""I-frame preview correctness and bounded, producer-independent extraction."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform.cast_thumbnails import HlsThumbnails
from oh_streamer.platform.hls_cast import MPEGTS_DELAY, TS_BASE, SegmentPlan, parse_media_playlist


class SnapshotVod:
    def __init__(self, plan=None):
        self.plan = plan or SegmentPlan(30, 265, 1920, 1080)
        self.cond = threading.Condition()
        self.gen, self.closed = 0, False
        self.segments = {}
        self.anchor, self.restarts, self.requests = 2, 0, 0

    def peek_segment(self, k):
        with self.cond:
            return self.gen, self.segments.get(k)

    def get_segment(self, k):
        raise AssertionError("preview must never trigger restoration")


class CastThumbnailTests(unittest.TestCase):
    def provider(self, **kwargs):
        vod = SnapshotVod()
        provider = HlsThumbnails(vod, "private-source.mp4", **kwargs)
        self.addCleanup(provider.close)
        return vod, provider

    def test_playlist_covers_every_segment_including_short_tail(self):
        vod, provider = self.provider()
        text = provider.playlist()
        entries = parse_media_playlist(text)
        self.assertIn("#EXT-X-I-FRAMES-ONLY\n", text)
        self.assertIn("#EXT-X-VERSION:4\n", text)
        self.assertEqual(len(entries), vod.plan.count)
        self.assertEqual([uri for _, uri in entries], ["thumb_0.ts", "thumb_1.ts", "thumb_2.ts"])
        self.assertAlmostEqual(sum(d for d, _ in entries), float(vod.plan.total_duration), places=5)
        self.assertEqual((provider.width, provider.height), (320, 180))
        self.assertTrue(text.endswith("#EXT-X-ENDLIST\n"))

    def test_portrait_and_small_sizes_preserve_aspect_without_upscale(self):
        for dims, expected in [((1080, 1920), (100, 180)), ((64, 48), (64, 48))]:
            vod = SnapshotVod(SegmentPlan(30, 300, *dims))
            provider = HlsThumbnails(vod, None)
            self.addCleanup(provider.close)
            self.assertEqual((provider.width, provider.height), expected)

    def test_original_cache_upgrades_to_ready_file_then_hls_restored(self):
        ready = Mock(return_value=None)
        vod, provider = self.provider(ready_source=ready)
        provider._encode = Mock(side_effect=[b"original", b"queue", b"restored"])
        self.assertEqual(provider.get_segment(1), b"original")
        self.assertEqual(provider.get_segment(1), b"original")
        ready.return_value = ("ready.mp4", [], 1.5, 0)
        self.assertEqual(provider.get_segment(1), b"queue")
        self.assertEqual(provider._encode.call_args.args[3], ("ready.mp4", (), 1.5, 0))
        vod.segments[1] = b"finished TS"
        self.assertEqual(provider.get_segment(1), b"restored")
        self.assertEqual(provider._encode.call_args.args[1], b"finished TS")
        self.assertEqual(provider.get_segment(1), b"restored")
        vod.segments.clear()  # Main playback cache eviction must not downgrade.
        ready.return_value = None
        self.assertEqual(provider.get_segment(1), b"restored")
        self.assertEqual(provider._encode.call_count, 3)
        self.assertEqual((vod.anchor, vod.restarts, vod.requests), (2, 0, 0))

    def test_generation_change_discards_completed_encode_and_cache(self):
        vod, provider = self.provider()
        provider._encode = Mock(return_value=b"old")
        self.assertEqual(provider.get_segment(0), b"old")
        with vod.cond:
            vod.gen += 1
        provider._encode.return_value = b"new"
        self.assertEqual(provider.get_segment(0), b"new")
        def invalidate(*args):
            with vod.cond:
                vod.gen += 1
            return b"obsolete"
        provider._encode.side_effect = invalidate
        self.assertIsNone(provider.get_segment(1))
        self.assertNotIn(1, provider._cache)

    def test_late_original_encode_cannot_overwrite_restored_preview(self):
        vod, provider = self.provider()
        started, release = threading.Event(), threading.Event()
        def encode(k, restored, *args):
            if restored is not None:
                return b"restored"
            started.set()
            release.wait(3)
            return b"original"
        provider._encode = Mock(side_effect=encode)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(provider.get_segment, 0)
            self.assertTrue(started.wait(2))
            vod.segments[0] = b"TS"
            self.assertEqual(provider.get_segment(0), b"restored")
            release.set()
            self.assertEqual(first.result(3), b"restored")
        self.assertEqual(provider.get_segment(0), b"restored")

    def test_cache_is_byte_bounded_and_invalid_requests_do_not_encode(self):
        _, provider = self.provider(cache_bytes=5)
        provider._encode = Mock(return_value=b"1234")
        self.assertIsNone(provider.get_segment(-1))
        self.assertIsNone(provider.get_segment(provider.plan.count))
        provider._encode.assert_not_called()
        self.assertEqual(provider.get_segment(0), b"1234")
        self.assertEqual(provider.get_segment(1), b"1234")
        self.assertEqual(list(provider._cache), [1])
        self.assertEqual(provider._bytes, 4)
        provider.close()
        self.assertIsNone(provider.get_segment(1))

    def test_concurrent_identical_requests_share_encode(self):
        vod, provider = self.provider()
        started, release, second_snapshot = threading.Event(), threading.Event(), threading.Event()
        original_peek = vod.peek_segment
        reads = []
        def peek(k):
            reads.append(k)
            if len(reads) == 2:
                second_snapshot.set()
            return original_peek(k)
        vod.peek_segment = peek
        def encode(*args):
            started.set()
            self.assertTrue(release.wait(3))
            return b"one encode"
        provider._encode = Mock(side_effect=encode)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(provider.get_segment, 0)
            self.assertTrue(started.wait(2))
            second = pool.submit(provider.get_segment, 0)
            self.assertTrue(second_snapshot.wait(2))
            release.set()
            self.assertEqual(first.result(3), b"one encode")
            self.assertEqual(second.result(3), b"one encode")
        self.assertEqual(provider._encode.call_count, 1)

    def test_overload_is_bounded_and_close_prevents_stale_result(self):
        _, provider = self.provider(max_workers=1, timeout=0.2)
        started, release = threading.Event(), threading.Event()
        def encode(*args):
            started.set()
            release.wait(2)
            return b"late"
        provider._encode = Mock(side_effect=encode)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(provider.get_segment, 0)
            self.assertTrue(started.wait(2))
            self.assertIsNone(provider.get_segment(1))
            provider.close()
            release.set()
            self.assertIsNone(first.result(3))
        self.assertEqual(provider._encode.call_count, 1)

    def test_source_clock_headers_and_absolute_stream_mapping(self):
        vod, provider = self.provider(fmt_start=2.0, stream_index=3)
        vod.plan.video_start = 3.5
        provider.input_args = ("-headers", "Referer: https://example.test/\r\n")
        cmd = provider._command(1, False)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "5.500000000")
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:3")
        self.assertLess(cmd.index("-headers"), cmd.index("-i"))
        ready = provider._command(1, False, ("piece.mp4", (), 1.5, None))
        self.assertEqual(ready[ready.index("-ss") + 1], "1.500000000")
        self.assertEqual(ready[ready.index("-i") + 1], "piece.mp4")

    @patch("oh_streamer.platform.cast_thumbnails.subprocess.Popen")
    def test_timeout_kills_drains_and_reaps_process(self, popen):
        _, provider = self.provider()
        proc = popen.return_value
        proc.communicate.side_effect = [subprocess.TimeoutExpired("ffmpeg", 1), (b"", None)]
        proc.poll.return_value = None
        self.assertIsNone(provider._encode(0, None, 0.01))
        proc.kill.assert_called_once()
        self.assertEqual(proc.communicate.call_count, 2)
        self.assertFalse(provider._processes)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg tools required")
class RealThumbnailTests(unittest.TestCase):
    def test_real_one_frame_h264_clocks_and_restored_upgrade(self):
        scratch = Path(__file__).resolve().parents[1] / ".test-tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="thumbnails_", dir=scratch) as directory:
            original = str(Path(directory) / "original.mp4")
            restored = str(Path(directory) / "restored.ts")
            for color, target, duration in [("blue", original, 9), ("red", restored, 1)]:
                subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi",
                    "-i", f"color=c={color}:size=160x90:rate=10:duration={duration}",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", target],
                    check=True, capture_output=True, timeout=20)
            vod = SnapshotVod(SegmentPlan(Fraction(10), 83, 160, 90, video_start=0.2))
            provider = HlsThumbnails(vod, original, fmt_start=0.2)
            self.addCleanup(provider.close)
            def inspect(data, k, red):
                self.assertTrue(data)
                self.assertEqual(len(data) % 188, 0)
                result = subprocess.run(["ffprobe", "-v", "error", "-f", "mpegts", "-show_streams",
                    "-show_packets", "-of", "json", "pipe:0"], input=data, capture_output=True,
                    check=True, timeout=10)
                info = json.loads(result.stdout)
                self.assertEqual(len(info["streams"]), 1)
                stream = info["streams"][0]
                self.assertEqual(stream["codec_name"], "h264")
                self.assertEqual(stream["profile"], "Constrained Baseline")
                self.assertEqual(stream["level"], 30)
                self.assertEqual((stream["width"], stream["height"]), (provider.width, provider.height))
                self.assertEqual(len(info["packets"]), 1)
                packet = info["packets"][0]
                self.assertIn("K", packet["flags"])
                self.assertAlmostEqual(float(packet["pts_time"]),
                    TS_BASE + MPEGTS_DELAY + float(vod.plan.start(k)), places=5)
                frame = subprocess.run(["ffmpeg", "-v", "error", "-f", "mpegts", "-i", "pipe:0",
                    "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                    input=data, capture_output=True, check=True, timeout=10).stdout
                self.assertEqual(len(frame), provider.width * provider.height * 3)
                r, g, b = frame[:3]
                self.assertGreater(r if red else b, 200)
                self.assertLess(b if red else r, 30)
            for k in (0, 2):
                inspect(provider.get_segment(k), k, False)
            vod.segments[2] = Path(restored).read_bytes()
            inspect(provider.get_segment(2), 2, True)
            self.assertEqual((vod.anchor, vod.requests, vod.restarts), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
