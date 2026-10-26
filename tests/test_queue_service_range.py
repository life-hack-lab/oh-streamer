"""queue_service with a byte-range (progressive http file, `.rcache`) source handle: it must behave like an HLS handle.
Run from streamer/: python -m unittest tests.test_queue_service_range"""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_STREAMER_ROOT, _STREAMER_ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import threading
import types
import unittest
from pathlib import Path

from tests.test_queue_service import (  # noqa: E402
    FakeHLSRegistry, FakeHLSSession, ServiceTestBase, wait_for)


class FakeRangeSession(FakeHLSSession):
    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        self.runtime_url = "http://127.0.0.1:45331/source.mp4"
        self.status_path = str(Path(path).with_name("range-source.json"))

    def snapshot(self):
        return {"status": self.status, "ready_segments": [{"index": 0}], "segment_count": 3,
                "total_bytes": 12_000_000, "downloaded_bytes": 4_194_304, "block_size": 4_194_304}


class FakeRangeRegistry(FakeHLSRegistry):
    def open_session(self, path):
        old = self.sessions.get(self.key(path))
        if not old or old.status != "complete":
            return None
        if not old.closed:
            return old
        reopened = FakeRangeSession(path, status="complete", identity=old.identity, generation=old.generation)
        self.add(reopened)
        return reopened


class RangeDownloader:
    """Announces a range session once its first block is ready, then waits for `release` like a real download."""

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.ready = threading.Event()
        self.release = threading.Event()
        self.cancel_seen = threading.Event()
        self.callback = None
        self.handle = None

    def __call__(self, target, headers, quality, dest_dir, on_progress, cancel, *, on_source_ready):
        self.calls.append(target)
        cache = Path(dest_dir) / "range-cache"
        cache.mkdir(parents=True, exist_ok=True)
        marker = cache / "source.rcache"
        marker.write_text('{"format": "range-cache"}\n', encoding="utf-8")
        (cache / "source.data").write_bytes(b"d" * 4096)
        (cache / "range-source.json").write_text(json.dumps(
            {"status": "complete", "total_bytes": 4096, "ext": "mp4"}), encoding="utf-8")
        self.handle = FakeRangeSession(marker)
        self.registry.add(self.handle)
        self.callback = on_source_ready
        on_source_ready(self.handle)
        self.ready.set()
        while not self.release.wait(0.02):
            if cancel.is_set():
                self.cancel_seen.set()
                raise RuntimeError("cancelled")
        self.handle.status = "complete"
        return types.SimpleNamespace(path=str(marker), title="Range title", duration=8.0)


class RangeQueueTests(ServiceTestBase):
    def range_item(self):
        self.svc.pause_all()
        registry = FakeRangeRegistry()
        downloader = RangeDownloader(registry)
        self.svc._range_cache_module = lambda: registry
        self.svc._hls_cache_module = lambda: None       # an HLS lookup for an .rcache path would find nothing
        self.svc._downloader = downloader
        self.svc._downloader_accepts_source_ready = True
        item_id = self.add("http://range.test/video.mp4", mode="listen")
        self.svc.resume_all()
        return item_id, downloader, registry

    def test_registry_dispatch_by_marker_suffix(self):
        hls, rng = object(), object()
        self.svc._hls_cache_module = lambda: hls
        self.svc._range_cache_module = lambda: rng
        self.assertIs(self.svc._registry_for("/x/source.rcache"), rng)
        self.assertIs(self.svc._registry_for("/x/SOURCE.RCACHE"), rng)
        self.assertIs(self.svc._registry_for("/x/playlist.m3u8"), hls)
        self.assertIs(self.svc._registry_for(""), hls)

    def test_ready_range_source_starts_worker_with_runtime_url_and_exact_progress(self):
        item_id, downloader, _registry = self.range_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id))
                                 and self.svc.store.get(item_id).progress.get("stage") == "processing", 10))
        item = self.svc.store.get(item_id)
        self.assertEqual((item.state, item.source_status), ("processing", "downloading"))
        self.assertTrue(item.source_path.endswith("source.rcache"))
        self.assertEqual(item.download_progress["total_bytes"], 12_000_000)
        self.assertEqual(item.download_progress["downloaded_bytes"], 4_194_304)
        self.assertEqual((item.download_progress["ready_segments"], item.download_progress["segment_count"]), (1, 3))

        job = json.loads((self.svc.store.item_dir(item_id) / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["source"], item.source_path)
        self.assertEqual(job["source_url"], downloader.handle.runtime_url)
        self.assertEqual(job["source_identity"], downloader.handle.identity)
        self.assertEqual(job["source_generation"], downloader.handle.generation)
        self.assertEqual(job["source_probe"], downloader.handle.metadata)
        self.assertTrue(job["source_status_path"].endswith("range-source.json"))

        downloader.release.set()
        self.assertTrue(wait_for(lambda: self.svc.store.get(item_id).source_status == "complete", 10))
        self.assertEqual(self.state(item_id), "processing")
        self.finish_worker()
        self.assertTrue(wait_for(lambda: self.state(item_id) == "done" and self.svc._wk is None, 10))

    def test_player_payload_uses_the_local_range_server(self):
        item_id, downloader, _registry = self.range_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set(), 10))
        payload = self.svc._source_payload(self.svc.store.get(item_id))
        self.assertEqual(payload["input"], self.svc.store.get(item_id).source_path)
        self.assertEqual(payload["source_url"], downloader.handle.runtime_url)
        self.assertEqual(payload["source_probe"], downloader.handle.metadata)
        self.assertTrue(payload["progressive"])

    def test_pause_cancels_the_session_and_keeps_the_cache_files(self):
        item_id, downloader, _registry = self.range_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        data = Path(downloader.handle.playlist_path).with_name("source.data")
        self.assertTrue(self.svc.pause(item_id)["ok"])
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.svc._dl is None
                                 and downloader.cancel_seen.is_set(), 10))
        item = self.svc.store.get(item_id)
        self.assertEqual((item.state, item.source_status), ("paused", "paused"))
        self.assertEqual(data.read_bytes(), b"d" * 4096)

    def test_completed_range_cache_is_reclaimed_after_the_player_leaves(self):
        item_id, downloader, _registry = self.range_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        cache_dir = Path(downloader.handle.playlist_path).parent
        with self.svc._lock:
            self.svc._watch[item_id] = self.svc._clock()
            self.svc._player_item = item_id
        downloader.release.set()
        self.assertTrue(wait_for(lambda: self.svc.store.get(item_id).source_status == "complete", 10))
        self.finish_worker()
        self.assertTrue(wait_for(lambda: self.state(item_id) == "done" and self.svc._wk is None, 10))
        self.assertTrue(cache_dir.exists())
        self.assertEqual(self.svc.store.get(item_id).source_path, str(cache_dir / "source.rcache"))
        self.svc._cleanup_source_sessions(set())
        self.assertTrue(cache_dir.exists(), "a watched item keeps its cache")

        with self.svc._lock:
            self.svc._watch.pop(item_id, None)
            self.svc._player_item = None
        self.svc._cleanup_source_sessions(set())
        item = self.svc.store.get(item_id)
        self.assertEqual((item.source_path, item.source_status, item.download_progress), ("", "", {}))
        self.assertFalse(cache_dir.exists())
        self.assertTrue(downloader.handle.closed)

    def test_remove_closes_the_session_and_ignores_a_late_callback(self):
        item_id, downloader, _registry = self.range_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        callback = downloader.callback
        self.assertEqual(self.svc.remove(item_id), {"ok": True})
        self.assertTrue(downloader.handle.closed)
        self.assertIsNone(self.svc.store.get(item_id))
        callback(downloader.handle)
        self.assertNotIn(item_id, self.svc._sources)

    def test_complete_range_cache_reopens_after_restart(self):
        self.svc.pause_all()
        registry = FakeRangeRegistry()
        self.svc._range_cache_module = lambda: registry
        item_id = self.add("http://range.test/done.mp4")
        marker = self.svc.store.item_dir(item_id) / "range-cache" / "source.rcache"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")
        handle = FakeRangeSession(marker, status="complete")
        registry.add(handle)
        self.svc.store.update(item_id, state="queued", source_path=str(marker), source_status="complete")
        handle.close()                       # the proxy died with the old process
        item = self.svc.store.get(item_id)
        self.assertTrue(self.svc._source_ready(item))
        reopened = self.svc._session_for(item)
        self.assertIsNot(reopened, handle)
        self.assertEqual(reopened.identity, handle.identity)

    def test_reprocess_reuses_a_complete_range_cache_data_file(self):
        self.svc.pause_all()
        item_id = self.add("http://range.test/again.mp4")
        folder = self.svc.store.item_dir(item_id) / "range-cache"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source.rcache").write_text("{}", encoding="utf-8")
        (folder / "source.data").write_bytes(b"v" * 100)
        side = {"status": "complete", "total_bytes": 100, "ext": "webm"}
        (folder / "range-source.json").write_text(json.dumps(side), encoding="utf-8")
        with self.svc._lock:                 # a watched item: the scheduler must not reclaim the cache meanwhile
            self.svc._watch[item_id] = self.svc._clock()
            self.svc._player_item = item_id
        self.svc.store.update(item_id, state="done", source_path=str(folder / "source.rcache"),
                              source_status="complete")
        found = self.svc._reprocess_source(self.svc.store.get(item_id))
        self.assertEqual((found["mode"], found["path"], found["suffix"]),
                         ("cache", str(folder / "source.data"), ".webm"))
        # a cache that is not complete cannot be reused
        side["status"] = "paused"
        (folder / "range-source.json").write_text(json.dumps(side), encoding="utf-8")
        self.assertEqual(self.svc._reprocess_source(self.svc.store.get(item_id))["mode"], "download")


if __name__ == "__main__":
    unittest.main()
