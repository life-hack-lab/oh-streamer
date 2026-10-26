"""Local-mode subtitles overlap an HLS download: restoration starts at once, blocks follow the downloaded range.

Downloader session, audio extraction and the ASR/translation processes are fakes; no models are loaded.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oh_streamer.queue import queue_service
from tests import test_queue_service as queue_test_support
from tests.test_queue_cloud_subtitle_overlap import GrowingSession, SEGMENTS, SEGMENT_SECONDS

HEAD = 100          # 700 s source -> 7 blocks of 100 s


class LocalSubtitleOverlapTests(queue_test_support.ServiceTestBase):
    def setUp(self):
        class FakeInstance:
            def serve(self, _handler, log=None):
                pass

            def close(self):
                pass

        with patch.object(queue_test_support.QS.single_instance, "acquire", return_value=FakeInstance()):
            super().setUp()
        self.svc.pause_all()
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.svc._paused_all = False
        self.windows = []           # VAD windows analysed for lazy cuts
        self.speech = []            # synthetic whole-source speech intervals; each window sees them clipped

    def downloading_item(self, head=HEAD):
        playlist = self.tmp / "hls" / "playlist.m3u8"
        playlist.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:10"]
        for i in range(SEGMENTS):
            lines += [f"#EXTINF:{SEGMENT_SECONDS:.6f},", f"segment-{i:06d}.ts"]
        playlist.write_text("\n".join(lines + ["#EXT-X-ENDLIST"]) + "\n", encoding="utf-8")
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")     # local pipeline, no frozen cloud plan
        item = self.svc.store.add("http://h/local-overlap", "url", None, None,
                                  {"subtitles": True, "subtitle_head_seconds": head, "fake_mode": "ok"}, title="local")
        self.svc.store.update(item.id, state="queued", source_path=str(playlist), source_status="downloading",
                              subtitle={"status": "queued", "percent": 0})
        session = GrowingSession(playlist)
        self.svc._sources[item.id] = session
        from oh_streamer.sources import hls_cache
        hls_cache.sidecar_path(playlist).write_text('{"version": 1, "identity": "cache-1"}', encoding="utf-8")
        self.install_fakes()
        return item.id, session

    def install_fakes(self):
        svc = self.svc
        svc._subtitle_check = Mock()
        svc._subtitle_item_environment = Mock(return_value={"available": True, "models": str(self.tmp / "models"),
                                                           "python": "python", "server": "server"})
        svc._subtitle_audio_offset = Mock(return_value=0)
        svc._subtitle_publish_update = Mock()
        svc._maybe_play_after_subtitles = Mock()
        svc._subtitle_mark_ready = Mock()
    def test_downloading_local_item_is_not_held_and_never_shows_waiting_gpu(self):
        item_id, _session = self.downloading_item()
        launched = []
        with patch.object(self.svc, "_run_worker", side_effect=lambda item, _run: launched.append(item.id)):
            self.svc._maybe_start_worker()
            self.svc._wk.thread.join(5)
        self.assertEqual(launched, [item_id])
        self.assertNotEqual(self.svc.store.get(item_id).subtitle["status"], "waiting_gpu")
        self.svc._wk = None

    def test_play_after_subtitles_picks_downloading_local_item(self):
        item_id, _session = self.downloading_item()
        self.svc.store.update(item_id, play_after_subtitles=True)
        picked = []
        with patch.object(self.svc, "_maybe_start_worker", side_effect=lambda it=None: picked.append(it)), \
                patch.object(self.svc, "is_player_busy", return_value=False), \
                patch.object(self.svc, "_maybe_play_after_subtitles"):
            self.svc._step()
        self.assertTrue(any(getattr(p, "id", None) == item_id for p in picked))


if __name__ == "__main__":
    unittest.main()
