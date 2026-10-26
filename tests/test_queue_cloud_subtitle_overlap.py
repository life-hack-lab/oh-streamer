"""Cloud subtitles overlap an HLS download: restoration starts at once, slices follow the downloaded range.

Downloader session, audio extraction and the cloud backends are fakes; no network or real credentials.
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
from tests.test_subtitle_cloud_first_runtime import settings as cloud_settings

SEGMENTS = 70           # 70 x 10 s = 700 s -> slices 0-300, 300-600, 600-700
SEGMENT_SECONDS = 10.0
SILENCES = [(294.0, 296.0), (594.0, 596.0)]     # synthetic silences: the lazy cuts snap to 295 and 595


class GrowingSession:
    """HLS session whose ready segments the test advances (the downloader's role)."""
    def __init__(self, playlist):
        self.playlist_path = str(playlist)
        self.runtime_url = "http://127.0.0.1:1/playlist.m3u8"
        self.identity, self.generation = "id", "gen"
        self.metadata = {}
        self.status_path = ""
        self.status = "downloading"
        self.ready = 0

    def snapshot(self):
        return {"status": self.status, "segment_count": SEGMENTS,
                "ready_segments": [{"index": i} for i in range(self.ready)]}

    def cancel(self):
        pass

    def close(self):
        pass


class CloudSubtitleOverlapTests(queue_test_support.ServiceTestBase):
    def setUp(self):
        class FakeInstance:
            def serve(self, _handler, log=None):
                pass

            def close(self):
                pass

        with patch.object(queue_test_support.QS.single_instance, "acquire", return_value=FakeInstance()):
            super().setUp()
        self.svc.pause_all()
        self.stop_scheduler()
        self.windows = []

    def stop_scheduler(self):
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.svc._paused_all = False

    def downloading_item(self, cloud=True):
        playlist = self.tmp / "hls" / "playlist.m3u8"
        playlist.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:10"]
        for i in range(SEGMENTS):
            lines += [f"#EXTINF:{SEGMENT_SECONDS:.6f},", f"segment-{i:06d}.ts"]
        playlist.write_text("\n".join(lines + ["#EXT-X-ENDLIST"]) + "\n", encoding="utf-8")
        if not cloud:       # the legacy contract keeps the serialized local pipeline
            self.svc.store.set_settings(subtitle_runtime_contract="legacy")
        cfg = cloud_settings(processing="smart_horizon") if cloud else {"subtitles": True}
        item = self.svc.store.add("http://h/overlap", "url", None, None, dict(cfg, fake_mode="ok"), title="overlap")
        self.svc.store.update(item.id, state="queued", source_path=str(playlist), source_status="downloading",
                              subtitle={"status": "queued", "percent": 0})
        session = GrowingSession(playlist)
        self.svc._sources[item.id] = session
        return item.id, session

    def test_cloud_item_downloading_starts_restore_worker_without_waiting_gpu(self):
        item_id, _session = self.downloading_item()
        launched = []
        with patch.object(self.svc, "_run_worker", side_effect=lambda item, _run: launched.append(item.id)):
            self.svc._maybe_start_worker()
            self.svc._wk.thread.join(5)
        self.assertEqual(launched, [item_id])
        self.assertNotEqual(self.svc.store.get(item_id).subtitle["status"], "waiting_gpu")
        self.svc._wk = None






    def test_play_after_subtitles_picks_downloading_item_in_cloud_and_local_mode(self):
        for cloud in (True, False):
            with self.subTest(cloud=cloud):
                item_id, _session = self.downloading_item(cloud=cloud)
                self.svc.store.update(item_id, play_after_subtitles=True)
                picked = []
                with patch.object(self.svc, "_maybe_start_worker", side_effect=lambda it=None: picked.append(it)),                         patch.object(self.svc, "is_player_busy", return_value=False),                         patch.object(self.svc, "_maybe_play_after_subtitles"):
                    self.svc._step()
                self.assertTrue(any(getattr(p, "id", None) == item_id for p in picked))
                self.svc.store.remove(item_id) if hasattr(self.svc.store, "remove") else None



if __name__ == "__main__":
    unittest.main()
