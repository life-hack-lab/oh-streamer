"""Focused tests for seekbar thumbnail extraction."""

from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

from fractions import Fraction
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from oh_streamer.player.thumbnails import ThumbnailWorker


class FakeFrame:
    pts = 3500

    def reformat(self, *, width, height, format):
        self.format_args = (width, height, format)
        return self

    def to_ndarray(self):
        return SimpleNamespace(tobytes=lambda: b"bgra pixels")


class FakeContainer:
    def __init__(self):
        codec = SimpleNamespace(width=1920, height=1080, thread_count=0, skip_frame=None)
        self.stream = SimpleNamespace(codec_context=codec, width=1920, height=1080,
                                      time_base=Fraction(1, 1000))
        self.streams = SimpleNamespace(video=[self.stream])
        self.frame = FakeFrame()
        self.seek_args = None
        self.closed = False

    def seek(self, offset, **kwargs):
        self.seek_args = (offset, kwargs)

    def demux(self, _stream):
        return [SimpleNamespace(decode=lambda: [self.frame])]

    def close(self):
        self.closed = True


class ThumbnailTests(unittest.TestCase):
    def test_decode_writes_frame_and_query_queues_a_missing_bucket(self):
        container = FakeContainer()
        opened = []
        av_module = SimpleNamespace(open=lambda *args, **kwargs: (opened.append((args, kwargs)) or container))
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory:
            worker = ThumbnailWorker(
                "https://example.test/video", {"User-Agent": "test"}, 20, 640, directory,
                av_module=av_module,
            )
            worker._decode(2)

            path = Path(directory) / "0000003500.bgra"
            self.assertEqual(path.read_bytes(), b"bgra pixels")
            self.assertEqual(container.seek_args[0], 4000)
            self.assertEqual(container.stream.codec_context.thread_count, 1)
            self.assertEqual(container.stream.codec_context.skip_frame, "NONKEY")
            self.assertEqual(opened[0][1]["options"]["user_agent"], "test")
            self.assertEqual(worker.query(4), (str(path), 640, 360, True))

            nearest = worker.query(7)
            self.assertEqual(nearest, (str(path), 640, 360, False))
            self.assertEqual(worker.want, 3)
            self.assertTrue(worker.wake.is_set())

    def test_prefetch_interval_bounds_total_prefetch_work(self):
        worker = ThumbnailWorker("video.mp4", {}, 6000, 320, ".")
        worker.start_prefetch(1)

        self.assertEqual(worker.interval, 10)
        self.assertTrue(worker.wake.is_set())

    def test_failed_bucket_is_marked_after_three_attempts_and_warns_once(self):
        messages = []
        worker = ThumbnailWorker(
            "video.mp4", {}, 20, 320, ".",
            av_module=SimpleNamespace(open=lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("bad input"))),
            logger=messages.append,
        )

        for _ in range(3):
            worker._decode(1)

        self.assertIn(1, worker.buckets)
        self.assertIsNone(worker.buckets[1])
        self.assertEqual(worker.errors, 0)
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
