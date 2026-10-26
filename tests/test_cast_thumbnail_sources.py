"""Preview input selection never schedules restoration or changes queue state."""
from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_streamer.player import cast_thumbnail_sources as S
from oh_streamer.player.progressive_stream import PlayInfo, Segment, ManifestState
from oh_streamer.platform.hls_cast import SegmentPlan


class CastThumbnailSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(__file__).resolve().parents[1] / ".test-tmp"
        self.tmp_root.mkdir(exist_ok=True)
        self.plan = SegmentPlan("30000/1001", 600, 64, 48, video_start=7.25)
        self.vod = NS(plan=self.plan, get_segment=Mock(side_effect=AssertionError("production requested")))

    def test_live_source_keeps_headers_and_format_origin(self):
        source = NS(src="https://example.test/video", fmt_start=3.25,
                    http=NS(cli_args=Mock(return_value=["-headers", "Referer: example\r\n"])))
        session = NS(source=source, run_input=Mock(side_effect=AssertionError("restoration requested")))
        with patch.object(S, "HlsThumbnails") as cls:
            self.assertIs(S.create_thumbnails(self.vod, session), cls.return_value)
        cls.assert_called_once_with(self.vod, source.src, ["-headers", "Referer: example\r\n"], fmt_start=3.25)
        session.run_input.assert_not_called()
        self.vod.get_segment.assert_not_called()

    def test_finished_file_keeps_selected_processed_stream(self):
        session = NS(path="output.mkv", fmt_start=1.5, stream_index=3)
        with patch.object(S, "HlsThumbnails") as cls:
            S.create_thumbnails(self.vod, session)
        cls.assert_called_once_with(self.vod, "output.mkv", fmt_start=1.5, stream_index=3)

    def test_unknown_source_is_optional(self):
        with patch.object(S, "HlsThumbnails") as cls:
            self.assertIsNone(S.create_thumbnails(self.vod, NS()))
        cls.assert_not_called()

    def _progressive(self, directory):
        play = PlayInfo("30000/1001", 60, video_start=7.25, rate_mult=2, count=10)
        state = ManifestState(play, {2: Segment(2, "seg_2.mkv", 60)})
        watcher = NS(state=state, segment_path=lambda seg: os.path.join(directory, seg.file),
                     poll=Mock(side_effect=AssertionError("poll requested")),
                     wait_newer=Mock(side_effect=AssertionError("wait requested")),
                     wake=Mock(side_effect=AssertionError("wake requested")))
        return NS(is_progressive_cast=True, watcher=watcher, fmt_start=6.5,
                  po=NS(source_url="http://127.0.0.1/download", path="original.mkv"),
                  playback=NS(want=Mock(side_effect=AssertionError("queue requested"))),
                  run_input=Mock(side_effect=AssertionError("production requested")))

    def test_progressive_original_uses_download_proxy_and_snapshot_callback(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as directory:
            session = self._progressive(directory)
            with patch.object(S, "HlsThumbnails") as cls:
                S.create_thumbnails(self.vod, session)
            args, kw = cls.call_args
            self.assertEqual(args, (self.vod, session.po.source_url))
            self.assertEqual(kw["fmt_start"], 6.5)
            self.assertIsNone(kw["ready_source"](0))
            self.assertIsNone(kw["ready_source"](1))  # manifest entry exists, file does not
            for method in (session.watcher.poll, session.watcher.wait_newer, session.watcher.wake,
                           session.playback.want, session.run_input, self.vod.get_segment):
                method.assert_not_called()
            session.po.source_url = None
            with patch.object(S, "HlsThumbnails") as cls:
                S.create_thumbnails(self.vod, session)
            self.assertEqual(cls.call_args.args[1], "original.mkv")

    def test_ready_source_fractional_boundary_and_probe_cache(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as directory:
            session = self._progressive(directory)
            path = Path(directory) / "seg_2.mkv"
            path.write_bytes(b"finished")
            ready = S._progressive_ready_source(self.vod, session)
            with patch.object(S, "_processed_track", return_value=(4, 0.125)) as probe:
                # HLS segment 1 starts at queue segment 2 despite a fractional frame rate.
                self.assertEqual(ready(1), (str(path), (), 0.125, 4))
                self.assertEqual(ready(1), (str(path), (), 0.125, 4))
                self.assertEqual(probe.call_count, 1)
                path.write_bytes(b"replaced with other content")
                self.assertIsNotNone(ready(1))
                self.assertEqual(probe.call_count, 2)

    def test_ready_source_local_offset_with_different_queue_grid(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as directory:
            session = self._progressive(directory)
            play = PlayInfo(30, 180, video_start=7.25, count=4)
            session.watcher.state = ManifestState(play, {0: Segment(0, "seg_0.mkv", 180)})
            path = Path(directory) / "seg_0.mkv"
            path.write_bytes(b"finished")
            ready = S._progressive_ready_source(self.vod, session)
            with patch.object(S, "_processed_track", return_value=(1, 0)):
                self.assertAlmostEqual(ready(1)[2], float(self.plan.start(1)))
                self.assertIsNone(ready(2))

    def test_missing_invalid_and_failed_segments_fall_back(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as directory:
            session = self._progressive(directory)
            ready = S._progressive_ready_source(self.vod, session)
            path = Path(directory) / "seg_2.mkv"
            path.write_bytes(b"finished")
            with patch.object(S, "_processed_track", side_effect=ValueError("bad media")) as probe:
                self.assertIsNone(ready(1))
                self.assertIsNone(ready(1))
                self.assertEqual(probe.call_count, 1)
                self.assertIsNone(ready(-1))
                self.assertIsNone(ready(self.plan.count))
            session.watcher.state.segments[2] = Segment(2, "../outside.mkv", 60)
            self.assertIsNone(ready(1))
            session.watcher.state.segments[2] = Segment(2, "seg_2.mkv", 0)
            self.assertIsNone(ready(1))
            session.watcher.state.segments[2] = Segment(2, "seg_2.mkv", 1)
            path.unlink()
            self.assertIsNone(ready(1))

    def test_processed_track_selection_rejects_unknown_video_tracks(self):
        def inspect(streams):
            doc = {"streams": streams, "format": {"start_time": "1.25"}}
            with patch.object(S.subprocess, "run", return_value=NS(stdout=json.dumps(doc))) as run:
                result = S._processed_track("segment.mkv")
            self.assertEqual(run.call_args.kwargs["timeout"], 3)
            return result
        original = {"index": 0, "codec_type": "video", "tags": {"title": "oh:original"}}
        processed = {"index": 3, "codec_type": "video", "tags": {"title": "oh:processed"},
                     "start_time": "1.5"}
        self.assertEqual(inspect([original, processed]), (3, 0.25))
        self.assertEqual(inspect([{"index": 2, "codec_type": "video"}]), (2, 0))
        self.assertIsNone(inspect([original]))
        self.assertIsNone(inspect([{"index": 0, "codec_type": "video"},
                                   {"index": 1, "codec_type": "video"}]))
        self.assertIsNone(inspect([dict(processed, disposition={"attached_pic": 1})]))


if __name__ == "__main__":
    unittest.main()
