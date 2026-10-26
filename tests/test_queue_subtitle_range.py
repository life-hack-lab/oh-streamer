"""Subtitles over a range-cache (.rcache) source: ready-time limit, media id, live-session handling."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from oh_streamer.queue import subtitle_host as sh  # noqa: F401
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.sources import media_input, range_cache as RC
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr
from tests.test_queue_subtitle_host import Base
from tests.test_range_cache import Upstream, make_data

BS = 1000
DURATION = 100.0


class RangeHostTests(Base):
    def setUp(self):
        super().setUp()
        self.data = make_data(10 * BS)
        self.up = Upstream(self.data)
        self.addCleanup(self.up.close)

    def session(self, name="a"):
        folder = self.root / name / "range-cache"
        s, _ = RC.prepare_session(folder, url=self.up.url, total_bytes=len(self.data), request_key=f"key-{name}",
                                  block_size=BS, auto_start=False,
                                  metadata={"format": {"duration": str(DURATION)}})
        self.addCleanup(RC.close_session, s.playlist_path)
        return s

    def item(self, s, status="downloading", item_id="it1"):
        item = self.add(item_id, source=s.playlist_path)
        item.source_status = status
        self.svc._session_for = lambda it, reopen_complete=True: RC.get_session(it.source_path) or (
            RC.open_session(it.source_path) if reopen_complete and it.source_status == "complete" else None)
        return item

    def fill(self, s, indices):
        for i in indices:
            RC.commit_block(s, i, self.data[i * BS:(i + 1) * BS])

    def test_limit_is_contiguous_ready_prefix_minus_one_block(self):
        s = self.session()
        self.item(s)
        self.fill(s, [0, 1, 2, 3])                     # 4 blocks of 10 -> 40 s, minus one block (10 s)
        self.assertEqual(self.host().source_state(), (True, 30.0, DURATION))

    def test_out_of_order_blocks_do_not_raise_the_limit(self):
        s = self.session()
        self.item(s)
        self.fill(s, [0, 1, 8, 9])                     # tail present (moov) but a hole in the middle
        self.assertEqual(self.host().source_state(), (True, 10.0, DURATION))

    def test_nothing_ready_gives_zero_limit_not_unusable(self):
        s = self.session()
        self.item(s)
        self.assertEqual(self.host().source_state(), (True, 0.0, DURATION))

    def test_complete_has_no_limit_and_reports_total(self):
        s = self.session()
        self.item(s, "complete")
        self.fill(s, range(10))
        RC.complete_session(s)
        self.assertEqual(self.host().source_state(), (True, None, DURATION))

    def test_no_live_session_is_unavailable_not_an_error(self):
        s = self.session()
        item = self.item(s)
        RC.close_session(s.playlist_path)
        self.assertEqual(self.host().source_state(), (False, 0.0, None))
        item.source_status = "paused"
        self.assertEqual(self.host().source_state(), (False, 0.0, None))

    def test_complete_item_with_closed_session_is_reopened(self):
        s = self.session()
        self.item(s, "complete")
        self.fill(s, range(10))
        RC.complete_session(s)
        RC.close_session(s.playlist_path)
        self.assertIsNone(RC.get_session(s.playlist_path))
        self.assertEqual(self.host().source_state(), (True, None, DURATION))
        self.assertIsNotNone(RC.get_session(s.playlist_path))      # reopened, runtime_url available again

    def test_media_id_uses_sidecar_identity_and_differs_between_items(self):
        a, b = self.session("a"), self.session("b")
        self.item(a, item_id="ia")
        self.item(b, item_id="ib")
        ida, idb = self.host("ia").media_id(), self.host("ib").media_id()
        self.assertEqual(ida, "rcache:" + a.identity)
        self.assertNotEqual(ida, idb)

    def test_fingerprint_of_markers_does_not_collide(self):
        a, b = self.session("a"), self.session("b")
        self.assertEqual(Path(a.playlist_path).read_bytes(), Path(b.playlist_path).read_bytes())   # identical markers
        fa, fb = sr.media_fingerprint(a.playlist_path), sr.media_fingerprint(b.playlist_path)
        self.assertNotEqual(fa, fb)
        self.assertEqual(fa, "rcache:" + a.identity)
        self.assertEqual(fa, sr.media_fingerprint(a.playlist_path))      # stable

    def test_fingerprint_without_sidecar_mixes_in_the_path(self):
        paths = []
        for name in ("x", "y"):
            folder = self.root / name
            folder.mkdir()
            (folder / "source.rcache").write_text("{}", encoding="utf-8")
            paths.append(folder / "source.rcache")
        self.assertNotEqual(sr.media_fingerprint(paths[0]), sr.media_fingerprint(paths[1]))

    def test_audio_offset_without_session_is_not_remembered(self):
        s = self.session()
        item = self.item(s)
        RC.close_session(s.playlist_path)
        calls = []
        from oh_streamer.queue.subtitle_jobs import SubtitleQueueMixin

        def probe(path):
            calls.append(path)
            return SubtitleQueueMixin._subtitle_audio_offset(path)
        self.svc._subtitle_audio_offset = probe
        host = self.host()
        self.assertEqual(host.audio_offset(item), 0.0)
        self.assertEqual(host.audio_offset(item), 0.0)
        self.assertEqual(len(calls), 2)                  # probed again each time: 0.0 was never cached
        self.assertIsNone(host._audio_offset)


class SchedulerSourceGoneTests(unittest.TestCase):
    def executor(self, source, usable):
        host = SimpleNamespace(source_state=lambda: (usable, None, None))
        released = []
        state = SimpleNamespace(status=sp.RUNNING)
        plan = SimpleNamespace(slices=[SimpleNamespace(get=lambda stage, track: state)],
                               release=lambda *a: released.append(a))
        progress = []
        ex = SimpleNamespace(source=source, host=host, plan=plan, _lock=threading.RLock(), _save=lambda: None,
                             _progress=lambda **kw: progress.append(kw))
        ex._source_gone = lambda exc: ss.CloudSliceExecutor._source_gone(ex, exc)
        return ex, released, progress

    def test_source_unavailable_error_is_always_gone(self):
        ex, *_ = self.executor("a/video.mp4", True)
        self.assertTrue(ss.CloudSliceExecutor._source_gone(ex, media_input.SourceUnavailable("x")))
        self.assertFalse(ss.CloudSliceExecutor._source_gone(ex, RuntimeError("model blew up")))

    def test_any_error_on_an_rcache_source_that_went_away_is_gone(self):
        ex, *_ = self.executor("a/source.rcache", False)
        self.assertTrue(ss.CloudSliceExecutor._source_gone(ex, RuntimeError("ffmpeg connection refused")))
        live, *_ = self.executor("a/source.rcache", True)
        self.assertFalse(ss.CloudSliceExecutor._source_gone(live, RuntimeError("real asr error")))

    def test_failed_releases_the_task_instead_of_failing_it(self):
        ex, released, progress = self.executor("a/source.rcache", False)
        task = SimpleNamespace(slice_index=0, stage=sp.STAGE_ASR, track=sp.PRIMARY_TRACK)
        ss.CloudSliceExecutor._failed(ex, task, media_input.SourceUnavailable("closed"))
        self.assertEqual(len(released), 1)
        self.assertEqual(progress[0]["state"], "deferred")


if __name__ == "__main__":
    unittest.main()
