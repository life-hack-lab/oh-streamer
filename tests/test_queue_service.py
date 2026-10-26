"""Tests for queue_service: fake downloader + fake worker script, no GPU, no network, no real shutdown."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from unittest.mock import Mock, patch
from pathlib import Path

import psutil

import oh_streamer.queue.queue_ipc as queue_ipc
import oh_streamer.queue.queue_service as QS
import oh_streamer.queue.queue_store as queue_store

FAKE_WORKER = r'''
import json, os, subprocess, sys, time
job = json.load(open(sys.argv[sys.argv.index("--job") + 1], encoding="utf-8"))
mode = job["settings"].get("fake_mode", "ok")
out = job["output"]
d = os.path.dirname(out)
with open(os.path.join(d, "start_k.txt"), "a") as f:
    f.write(repr(job.get("start_k")) + "\n")
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(os.path.join(d, "pids.txt"), "a") as f:
    f.write("%d %d\n" % (os.getpid(), child.pid))
def emit(o):
    print(json.dumps(o), flush=True)
emit({"type": "progress", "frame": 5, "frames": 10, "fps": 30.0, "eta": 1.0, "segments": 1, "phase": "restore"})
if mode == "listen":                  # keeps running, logs the stdin lines, finishes on {"cmd": "finish"}
    for line in sys.stdin:
        with open(os.path.join(d, "stdin.log"), "a") as f:
            f.write(line)
        if "finish" in line:
            break
marker = os.path.join(d, "marker")
if mode == "slow" or (mode == "slow_once" and not os.path.exists(marker)):
    open(marker, "w").close()
    time.sleep(120)
if mode == "fail":
    child.kill()
    sys.stderr.write("boom from stderr\n")
    sys.exit(3)
if mode == "errline":
    child.kill()
    emit({"type": "error", "message": "model exploded"})
    sys.exit(1)
with open(out, "wb") as f:
    f.write(b"x" * 1234)
child.kill()
emit({"type": "done", "output": out})
'''


def wait_for(cond, timeout=10.0, step=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return cond()


class FakeDownloader:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, target, headers, quality, dest_dir, on_progress, cancel):
        with self.lock:
            self.calls.append(target)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if target.startswith("http://fail"):
                raise RuntimeError("HTTP Error 500")
            if target.startswith("http://slow"):
                on_progress({"stage": "downloading", "downloaded_bytes": 10, "total_bytes": 100})
                while not cancel.wait(0.02):
                    pass
                raise RuntimeError("cancelled")
            if not target.startswith(("http://", "https://")):      # local file: returned as is (like the real one)
                return types.SimpleNamespace(path=target, title=Path(target).stem, duration=None)
            time.sleep(0.05)
            on_progress({"stage": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
            p = Path(dest_dir)
            p.mkdir(parents=True, exist_ok=True)
            src = p / "source.mp4"
            src.write_bytes(b"s" * 1000)
            return types.SimpleNamespace(path=str(src), title="T " + target, duration=10.0)
        finally:
            with self.lock:
                self.active -= 1


class FakeHLSSession:
    def __init__(self, path, *, status="downloading", identity="identity-1", generation="generation-1"):
        self.playlist_path = str(path)
        self.runtime_url = "http://127.0.0.1:45231/playlist.m3u8"
        self.identity = identity
        self.generation = generation
        self.metadata = {"streams": [{"codec_type": "video", "codec_name": "h264"}],
                         "format": {"duration": "8.0"}}
        self.status_path = str(Path(path).with_name("hls-source.json"))
        self.status = status
        self.closed = False
        self.cancelled = False

    def snapshot(self):
        return {"status": self.status, "ready_segments": [0], "segment_count": 2}

    def cancel(self):
        self.cancelled = True
        if self.status == "downloading":
            self.status = "paused"

    def close(self):
        self.closed = True


class FakeHLSRegistry:
    def __init__(self):
        self.sessions = {}

    @staticmethod
    def key(path):
        return str(Path(path).resolve())

    def add(self, handle):
        self.sessions[self.key(handle.playlist_path)] = handle

    def get_session(self, path):
        session = self.sessions.get(self.key(path))
        return session if session and not session.closed else None

    def open_session(self, path):
        old = self.sessions.get(self.key(path))
        if not old or old.status != "complete":
            return None
        if not old.closed:
            return old
        reopened = FakeHLSSession(path, status="complete", identity=old.identity, generation=old.generation)
        self.add(reopened)
        return reopened

    def close_session(self, path):
        session = self.sessions.get(self.key(path))
        if session:
            session.close()


class SlowProgressiveDownloader:
    def __init__(self, registry, ready_gate=None):
        self.registry = registry
        self.calls = []
        self.ready = threading.Event()
        self.release = threading.Event()
        self.ready_gate = ready_gate or threading.Event()
        if ready_gate is None:
            self.ready_gate.set()
        self.cancel_seen = threading.Event()
        self.callback = None
        self.handle = None

    def __call__(self, target, headers, quality, dest_dir, on_progress, cancel, *, on_source_ready):
        self.calls.append(target)
        cache = Path(dest_dir) / "hls-cache"
        cache.mkdir(parents=True, exist_ok=True)
        playlist = cache / "playlist.m3u8"
        playlist.write_text("#EXTM3U\n", encoding="utf-8")
        (cache / "segment-000000.ts").write_bytes(b"first-complete-segment")
        self.handle = FakeHLSSession(playlist)
        self.registry.add(self.handle)
        self.callback = on_source_ready
        while not self.ready_gate.wait(0.02):
            if cancel.is_set():
                self.cancel_seen.set()
                raise RuntimeError("cancelled before first segment became ready")
        on_source_ready(self.handle)
        self.ready.set()
        while not self.release.wait(0.02):
            if cancel.is_set():
                self.cancel_seen.set()
                raise RuntimeError("cancelled")
        self.handle.status = "complete"
        return types.SimpleNamespace(path=str(playlist), title="HLS title", duration=8.0)


class ServiceTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="qsvc-"))
        self.name = "test-" + uuid.uuid4().hex[:8]
        self.dl = FakeDownloader()
        self.worker_py = self.tmp / "fake_worker.py"
        self.worker_py.write_text(FAKE_WORKER, encoding="utf-8")
        self.exec_states = []
        self.shutdowns = []
        self.now = [1000.0]
        self.launched = []
        self.players = []          # replies for send_to_player
        self.svc = QS.QueueService(
            self.tmp / "root", player_name=self.name,
            launch_player=lambda path, payload: self.launched.append((path, payload)),
            downloader=self.dl,
            worker_command=lambda job: [sys.executable, str(self.worker_py), "--job", job],
            set_execution_state=self.exec_states.append,
            run_shutdown=self.shutdowns.append,
            send_to_player=lambda payload: (self.players.append(payload) or self.player_reply),
            clock=lambda: self.now[0], tick=0.02)
        self.player_reply = None
        self._orig_retry = QS.SPACE_RETRY_INTERVAL
        self._orig_work_retry = QS.WORK_RETRY_INTERVAL
        QS.SPACE_RETRY_INTERVAL = 0.05
        QS.WORK_RETRY_INTERVAL = 0.05
        self.assertTrue(self.svc.start())

    def tearDown(self):
        QS.SPACE_RETRY_INTERVAL = self._orig_retry
        QS.WORK_RETRY_INTERVAL = self._orig_work_retry
        self.svc.stop()
        for pidfile in self.tmp.rglob("pids.txt"):       # only processes our fake workers started
            for tok in pidfile.read_text().split():
                try:
                    psutil.Process(int(tok)).kill()
                except psutil.Error:
                    pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add(self, target, mode="ok", headers=None, kind="url", **settings):
        settings = dict(settings, fake_mode=mode)
        return self.svc.store.add(target, kind, headers, None, settings, title=target).id

    def state(self, item_id):
        return self.svc.store.get(item_id).state

    def ready(self, state="queued", mode="ok", headers=None):
        """An item whose source is downloaded, in the given state. Leaves pause_all on so the scheduler ignores it."""
        self.svc.pause_all()
        i = self.add("http://h/" + uuid.uuid4().hex[:6], mode, headers)
        src = self.svc.store.item_dir(i) / "source.mp4"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"s" * 100)
        self.svc.store.update(i, state=state, source_path=str(src))
        return i

    def log_lines(self, item_id, name):
        f = self.svc.store.item_dir(item_id) / name
        return f.read_text().splitlines() if f.exists() else []

    def finish_worker(self):
        """Tell the running 'listen' fake worker to finish."""
        proc = self.svc._wk.proc
        proc.stdin.write(b'{"cmd": "finish"}\n')
        proc.stdin.flush()

    def pids(self, item_id):
        f = self.svc.store.item_dir(item_id) / "pids.txt"
        if not f.exists():
            return []
        return [tuple(int(t) for t in ln.split()) for ln in f.read_text().splitlines() if ln.strip()]

    def alive(self, pid):
        try:
            return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False


class SchedulingTests(ServiceTestBase):
    def test_items_processed_in_order_one_at_a_time(self):
        ids = [self.add(f"http://h/{n}") for n in "abc"]
        self.assertTrue(wait_for(lambda: all(self.state(i) == "done" for i in ids), 30))
        self.assertEqual(self.dl.calls, [f"http://h/{n}" for n in "abc"])
        self.assertEqual(self.dl.max_active, 1)
        for i in ids:
            it = self.svc.store.get(i)
            self.assertEqual(it.size_bytes, 1234)
            self.assertTrue(os.path.isfile(it.output_path))
            self.assertEqual(it.source_path, "")
            self.assertFalse((self.svc.store.item_dir(i) / "source.mp4").exists())
            self.assertEqual(it.title, "T " + it.target)      # title was the target -> taken from the download
        # one worker at a time: the pid files are written in order of completion
        starts = [os.path.getmtime(self.svc.store.item_dir(i) / "job.json") for i in ids]
        self.assertEqual(starts, sorted(starts))


class ProgressiveQueueTests(ServiceTestBase):
    def test_finished_source_audio_is_kept_until_progressive_viewer_leaves(self):
        i = self.ready(mode="listen")
        src = Path(self.svc.store.get(i).source_path)
        self.svc._on_message({"cmd": "watching", "id": i})
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.svc._wk is not None and self.svc._wk.proc is not None))
        self.finish_worker()
        self.assertTrue(wait_for(lambda: self.state(i) == "done"))
        self.assertTrue(src.exists())
        self.assertEqual(self.svc.store.get(i).source_path, str(src))
        self.now[0] += 1000
        self.assertTrue(wait_for(lambda: not src.exists()))
        self.assertEqual(self.svc.store.get(i).source_path, "")

    def progressive_item(self, mode="listen", ready_gate=None):
        self.svc.pause_all()
        registry = FakeHLSRegistry()
        downloader = SlowProgressiveDownloader(registry, ready_gate=ready_gate)
        self.svc._hls_cache_module = lambda: registry
        self.svc._downloader = downloader
        self.svc._downloader_accepts_source_ready = True
        item_id = self.add("http://hls.test/media.m3u8", mode=mode)
        self.svc.resume_all()
        return item_id, downloader, registry

    def test_ready_source_starts_worker_while_download_continues(self):
        item_id, downloader, _registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id))
                                 and self.svc.store.get(item_id).progress.get("stage") == "processing", 10))
        item = self.svc.store.get(item_id)
        self.assertEqual(item.state, "processing")
        self.assertEqual(item.source_status, "downloading")
        self.assertTrue(self.svc._dl is not None and not downloader.release.is_set())
        self.assertEqual(item.download_progress["stage"], "downloading")
        self.assertEqual(item.progress["stage"], "processing")

        job = json.loads((self.svc.store.item_dir(item_id) / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["source"], item.source_path)
        self.assertEqual(job["source_url"], downloader.handle.runtime_url)
        self.assertEqual(job["source_identity"], downloader.handle.identity)
        self.assertEqual(job["source_generation"], downloader.handle.generation)
        self.assertEqual(job["source_probe"], downloader.handle.metadata)
        self.assertEqual(job["source_status_path"], downloader.handle.status_path)

        downloader.release.set()
        self.assertTrue(wait_for(lambda: self.svc.store.get(item_id).source_status == "complete", 10))
        self.assertEqual(self.state(item_id), "processing")  # download completion must not reset a live worker
        self.assertEqual(len(self.pids(item_id)), 1)
        self.finish_worker()
        self.assertTrue(wait_for(lambda: self.state(item_id) == "done" and self.svc._wk is None, 10))
        self.assertEqual(downloader.calls, ["http://hls.test/media.m3u8"])

    def test_worker_auto_pause_keeps_ready_same_item_queued_while_download_runs(self):
        item_id, downloader, _registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        self.svc._busy = True
        self.svc._busy_ts = self.svc._clock()
        self.svc._changed()
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.state(item_id) == "queued", 10))
        self.assertIsNotNone(self.svc._dl)
        self.assertFalse(downloader.release.is_set())

    def test_cancelled_worker_reader_cannot_overwrite_replacement_progress(self):
        item_id = self.ready(state="processing")
        sentinel = {"stage": "processing", "frame": 7, "frames": 20, "phase": "restore"}
        self.svc.store.update(item_id, progress=sentinel)
        generation = 12
        old_run = QS._Run(item_id, generation)
        old_run.cancel.set()
        replacement = QS._Run(item_id, generation)
        with self.svc._lock:
            self.svc._generations[item_id] = generation
            self.svc._wk = replacement

        proc = types.SimpleNamespace(stdout=[json.dumps({
            "type": "progress", "frame": 99, "frames": 100, "phase": "restore",
        }).encode("utf-8")])
        self.svc._read_worker_output(proc, item_id, self.svc.store.item_dir(item_id), {}, old_run)

        self.assertEqual(self.svc.store.get(item_id).progress, sentinel)
        with self.svc._lock:
            self.svc._wk = None

    def test_retry_keeps_a_valid_download_generation_and_stops_old_worker(self):
        item_id, downloader, _registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        self.svc._busy = True
        self.svc._busy_ts = self.svc._clock()
        download_run = self.svc._dl
        worker_run = self.svc._wk
        generation = download_run.generation
        self.svc.store.update(item_id, state="failed", error="transient worker failure")
        self.assertEqual(self.svc.retry(item_id), {"ok": True})
        self.assertIs(self.svc._dl, download_run)
        self.assertFalse(download_run.cancel.is_set())
        self.assertEqual(download_run.generation, generation)
        self.assertTrue(worker_run.cancel.is_set())
        self.assertEqual(self.svc._generations[item_id], generation)
        self.assertTrue(wait_for(lambda: self.svc._wk is None, 10))
        downloader.release.set()
        self.assertTrue(wait_for(lambda: self.svc.store.get(item_id).source_status == "complete", 10))
        self.assertEqual(self.state(item_id), "queued")

    def test_resumed_source_refreshes_only_the_current_queue_player(self):
        item_id, downloader, registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        self.svc._watch[item_id] = self.svc._clock()
        self.svc._player_item = item_id
        self.players.clear()

        replacement = FakeHLSSession(downloader.handle.playlist_path, generation="generation-2")
        replacement.runtime_url = "http://127.0.0.1:45232/playlist.m3u8"
        registry.add(replacement)
        downloader.callback(replacement)
        self.assertEqual([p["source_url"] for p in self.players], [replacement.runtime_url])

        self.players.clear()
        self.svc._player_item = "another-queue-item"
        replacement_again = FakeHLSSession(downloader.handle.playlist_path, generation="generation-3")
        replacement_again.runtime_url = "http://127.0.0.1:45233/playlist.m3u8"
        registry.add(replacement_again)
        downloader.callback(replacement_again)
        self.assertEqual(self.players, [])

    def test_completed_hls_cache_waits_for_player_lease_then_is_reclaimed(self):
        item_id, downloader, _registry = self.progressive_item()
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
        self.assertEqual(self.svc.store.get(item_id).source_path, str(cache_dir / "playlist.m3u8"))

        with self.svc._lock:
            self.svc._watch.pop(item_id, None)
            self.svc._player_item = None
        self.svc._cleanup_source_sessions(set())
        item = self.svc.store.get(item_id)
        self.assertEqual((item.source_path, item.source_status, item.download_progress), ("", "", {}))
        self.assertFalse(cache_dir.exists())

    def test_play_before_source_ready_waits_then_opens_progressive_queue_item(self):
        gate = threading.Event()
        item_id, downloader, _registry = self.progressive_item(ready_gate=gate)
        self.player_reply = {"ok": True}
        self.assertTrue(wait_for(lambda: self.state(item_id) == "downloading", 5))
        self.assertEqual(self.svc.play(item_id), {"ok": True, "pending": True})
        self.assertEqual(self.players, [])
        self.assertEqual(self.launched, [])
        gate.set()
        self.assertTrue(wait_for(lambda: bool(self.players), 10))
        self.assertEqual(self.players[0]["input"], self.svc.store.get(item_id).source_path)
        self.assertEqual(self.players[0]["queue_id"], item_id)
        self.assertTrue(self.players[0]["progressive"])
        self.assertEqual(self.players[0]["source_url"], downloader.handle.runtime_url)
        self.assertEqual(self.players[0]["source_probe"], downloader.handle.metadata)
        self.assertNotIn("cmd", self.players[0])

    def test_pause_stops_both_runs_and_keeps_committed_segments(self):
        item_id, downloader, _registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        worker_pid, _child_pid = self.pids(item_id)[0]
        segment = Path(downloader.handle.playlist_path).with_name("segment-000000.ts")
        self.assertTrue(self.svc.pause(item_id)["ok"])
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.svc._dl is None
                                 and not self.alive(worker_pid) and downloader.cancel_seen.is_set(), 10))
        item = self.svc.store.get(item_id)
        self.assertEqual((item.state, item.source_status), ("paused", "paused"))
        self.assertEqual(segment.read_bytes(), b"first-complete-segment")

    def test_remove_stops_both_runs_and_ignores_late_source_callback(self):
        item_id, downloader, _registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        worker_pid, _child_pid = self.pids(item_id)[0]
        callback = downloader.callback
        item = self.svc.store.get(item_id)
        self.assertEqual(self.svc.remove(item_id), {"ok": True})
        self.assertTrue(downloader.cancel_seen.is_set())
        self.assertFalse(self.alive(worker_pid))
        self.assertIsNone(self.svc.store.get(item_id))
        self.assertFalse(self.svc.store.item_dir(item_id).exists())
        callback(downloader.handle)  # an old callback cannot recreate state after the removal tombstone
        self.assertNotIn(item_id, self.svc._sources)
        self.assertIsNone(self.svc.store.get(item_id))

    def test_stop_stops_downloader_and_worker_together(self):
        item_id, downloader, _registry = self.progressive_item()
        self.assertTrue(wait_for(lambda: downloader.ready.is_set() and self.svc._wk is not None
                                 and bool(self.pids(item_id)), 10))
        worker_pid, _child_pid = self.pids(item_id)[0]
        self.svc.stop()
        self.assertTrue(downloader.cancel_seen.is_set())
        self.assertFalse(self.alive(worker_pid))
        self.assertIsNone(self.svc._dl)
        self.assertIsNone(self.svc._wk)

    def test_restart_reopens_complete_cache_and_restarts_incomplete_download(self):
        self.svc.pause_all()
        registry = FakeHLSRegistry()
        self.svc._hls_cache_module = lambda: registry
        complete = self.add("http://hls.test/complete.m3u8")
        complete_path = self.svc.store.item_dir(complete) / "hls-cache" / "playlist.m3u8"
        complete_path.parent.mkdir(parents=True, exist_ok=True)
        complete_path.write_text("#EXTM3U\n", encoding="utf-8")
        complete_handle = FakeHLSSession(complete_path, status="complete")
        registry.add(complete_handle)
        self.svc.store.update(complete, state="queued", source_path=str(complete_path), source_status="complete")
        self.svc.stop()
        complete_handle.close()

        # Model a process crash during an incomplete progressive download.
        incomplete = self.svc.store.add("http://hls.test/incomplete.m3u8", "url", {}, None, {},
                                        title="incomplete").id
        incomplete_path = self.svc.store.item_dir(incomplete) / "hls-cache" / "playlist.m3u8"
        incomplete_path.parent.mkdir(parents=True, exist_ok=True)
        incomplete_path.write_text("#EXTM3U\n", encoding="utf-8")
        self.svc.store.update(incomplete, state="processing", source_path=str(incomplete_path),
                              source_status="downloading")

        downloader = SlowProgressiveDownloader(registry)
        replacement = QS.QueueService(
            self.tmp / "root", player_name=self.name,
            launch_player=lambda path, payload: None, downloader=downloader,
            worker_command=lambda job: [sys.executable, str(self.worker_py), "--job", job],
            set_execution_state=lambda _flags: None, run_shutdown=lambda _args: None,
            send_to_player=lambda _payload: {"ok": True}, tick=0.02)
        replacement._hls_cache_module = lambda: registry
        self.assertTrue(replacement.start())
        job_path = replacement.store.item_dir(complete) / "job.json"
        self.assertTrue(wait_for(lambda: job_path.is_file(), 10))
        job = json.loads(job_path.read_text(encoding="utf-8"))
        self.assertEqual(job["source_identity"], complete_handle.identity)
        self.assertEqual(job["source_generation"], complete_handle.generation)
        self.assertTrue(wait_for(lambda: downloader.ready.is_set(), 10))
        self.assertEqual(replacement.store.get(incomplete).source_status, "downloading")
        replacement.stop()
        self.svc = replacement  # tearDown owns the replacement service now

    def test_job_settings_merge_and_no_headers(self):
        self.svc.set_settings({"encoder": "h264_nvenc", "interpolation": "off", "save_compare": True})
        i = self.add("http://h/a", headers={"Cookie": "SECRETCOOKIE"}, detection_model="v4-fast")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        text = (self.svc.store.item_dir(i) / "job.json").read_text(encoding="utf-8")
        job = json.loads(text)
        self.assertEqual(job["settings"]["encoder"], "h264_nvenc")
        self.assertEqual(job["settings"]["interpolation"], "off")
        self.assertEqual(job["settings"]["compare"], "on")
        self.assertEqual(job["settings"]["detection_model"], "v4-fast")
        self.assertEqual(job["output"], str(self.svc.store.item_dir(i) / "output.mkv"))
        self.assertNotIn("SECRETCOOKIE", text)
        self.assertNotIn("headers", job)

    def test_local_file_source_is_never_deleted(self):
        f = self.tmp / "mine.mp4"
        f.write_bytes(b"q" * 500)
        i = self.add(str(f), kind="file")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertTrue(f.exists())

    def test_failures(self):
        a = self.add("http://fail/x")
        b = self.add("http://h/ok", mode="fail")
        c = self.add("http://h/ok2", mode="errline")
        self.assertTrue(wait_for(lambda: all(self.state(i) == "failed" for i in (a, b, c)), 30))
        self.assertIn("HTTP Error 500", self.svc.store.get(a).error)
        self.assertIn("boom from stderr", self.svc.store.get(b).error)
        self.assertEqual(self.svc.store.get(c).error, "model exploded")
        # retry of a failed worker item keeps its downloaded source
        self.assertTrue(self.svc.retry(b)["ok"])
        self.assertIn(self.state(b), ("queued", "processing", "failed"))

    def test_state_notifications_are_coalesced(self):
        calls = []
        self.svc.on_state.append(lambda st: calls.append(time.monotonic()))
        time.sleep(0.7)
        calls.clear()
        i = self.add("http://slow/a")
        t0 = time.monotonic()
        for n in range(60):
            self.svc.store.update(i, title=f"t{n}")
            time.sleep(0.01)
        time.sleep(0.6)
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(len(calls), 1)
        self.assertLessEqual(len(calls), int(elapsed * 2) + 2)


class PauseTests(ServiceTestBase):
    def test_pause_kills_worker_tree_and_resume_restarts(self):
        i = self.add("http://h/a", mode="slow_once")
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        worker_pid, child_pid = self.pids(i)[0]
        self.assertTrue(self.alive(worker_pid) and self.alive(child_pid))
        self.assertTrue(self.svc.pause(i)["ok"])
        self.assertTrue(wait_for(lambda: not self.alive(worker_pid) and not self.alive(child_pid), 10))
        time.sleep(0.3)
        self.assertEqual(self.state(i), "paused")            # not restarted by the scheduler
        self.assertEqual(len(self.pids(i)), 1)
        self.assertTrue(self.svc.resume(i)["ok"])
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertEqual(len(self.pids(i)), 2)                # a new worker process did the rest

    def test_pause_waiting_item_is_skipped_by_scheduler(self):
        a = self.add("http://slow/a")        # occupies the downloader
        b = self.add("http://h/b")
        c = self.add("http://h/c")
        self.assertTrue(wait_for(lambda: self.state(a) == "downloading"))
        self.svc.pause(b)
        self.svc.pause(a)
        self.assertTrue(wait_for(lambda: self.state(c) == "done", 20))
        self.assertEqual(self.state(b), "paused")
        self.assertEqual(self.state(a), "paused")
        self.assertTrue(self.svc.resume(b)["ok"])
        self.assertTrue(wait_for(lambda: self.state(b) == "done", 20))

    def test_pause_all_and_resume_all(self):
        i = self.add("http://h/a", mode="slow_once")
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        worker_pid, _ = self.pids(i)[0]
        self.svc.pause_all()
        self.assertTrue(wait_for(lambda: not self.alive(worker_pid) and self.state(i) == "queued", 10))
        j = self.add("http://h/b")
        time.sleep(0.4)
        self.assertEqual(self.state(j), "waiting")           # no downloads while paused_all
        self.assertTrue(self.svc.get_state()["paused_all"])
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "done" and self.state(j) == "done", 30))

    def test_remove_running_item_kills_worker_and_deletes_files(self):
        i = self.add("http://h/a", mode="slow")
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        worker_pid, child_pid = self.pids(i)[0]
        d = self.svc.store.item_dir(i)
        self.assertTrue(self.svc.remove(i)["ok"])
        self.assertTrue(wait_for(lambda: not self.alive(worker_pid) and not self.alive(child_pid), 10))
        self.assertIsNone(self.svc.store.get(i))
        self.assertFalse(d.exists())


class PlayerBusyTests(ServiceTestBase):
    def test_auto_pause_resume_and_expiry(self):
        i = self.add("http://h/a", mode="slow_once")
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        worker_pid, _ = self.pids(i)[0]
        reply = self.svc._on_message({"cmd": "player_busy", "busy": True, "pid": 1})
        self.assertTrue(reply["ok"])
        self.assertTrue(wait_for(lambda: not self.alive(worker_pid) and self.state(i) == "queued", 10))
        self.assertTrue(self.svc.get_state()["player_busy"])
        time.sleep(0.4)
        self.assertEqual(self.state(i), "queued")                 # held back while the player is busy
        self.now[0] += QS.queue_ipc.BUSY_EXPIRE + 1                 # heartbeats stopped: busy flag expires
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertFalse(self.svc.get_state()["player_busy"])

    def test_explicit_not_busy_resumes(self):
        i = self.add("http://h/a", mode="slow_once")
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.assertTrue(wait_for(lambda: self.state(i) == "queued"))
        self.svc._on_message({"cmd": "player_busy", "busy": False})
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))

    def test_busy_does_not_stop_downloads(self):
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        i = self.add("http://h/a")
        self.assertTrue(wait_for(lambda: self.state(i) == "queued", 10))
        time.sleep(0.3)
        self.assertEqual(self.state(i), "queued")


class WantWatchTests(ServiceTestBase):
    def test_want_prioritizes_hls_at_requested_segment_not_old_clock(self):
        i = self.ready("paused")
        item = self.svc.store.get(i)
        source = Mock()
        with patch.object(self.svc, "_source_ready", return_value=True), \
                patch.object(self.svc, "_session_for", return_value=source):
            original = item.source_path
            self.svc.store.update(i, source_path=str(Path(original).with_suffix(".m3u8")))
            try:
                r = self.svc._on_message({"cmd": "want", "id": i, "k": 450, "pos": 10})
                self.assertEqual(r["error"], "paused")
                source.request_position.assert_called_once_with(450 * QS.SEGMENT_SECONDS)
            finally:
                self.svc.store.update(i, source_path=original)

    def test_want_preempts_running_item(self):
        a = self.add("http://h/a", mode="slow_once")
        self.assertTrue(wait_for(lambda: self.state(a) == "processing" and self.pids(a), 20))
        b = self.add("http://h/b")
        self.assertTrue(wait_for(lambda: self.state(b) == "queued", 20))      # downloaded while a is processing
        worker_pid, child_pid = self.pids(a)[0]
        self.assertTrue(self.svc._on_message({"cmd": "want", "id": b, "k": 3})["ok"])
        self.assertTrue(wait_for(lambda: not self.alive(worker_pid) and not self.alive(child_pid), 10))
        self.assertTrue(wait_for(lambda: self.state(b) == "done", 20))
        self.assertEqual(self.log_lines(b, "start_k.txt"), ["3"])
        self.assertNotEqual(self.state(a), "paused")                          # not a user pause: a resumes by itself
        self.assertTrue(wait_for(lambda: self.state(a) == "done", 20))
        self.assertEqual(len(self.pids(a)), 2)
        self.assertLessEqual(os.path.getmtime(self.svc.store.get(b).output_path),
                             os.path.getmtime(self.svc.store.get(a).output_path))
        self.assertEqual(self.log_lines(a, "start_k.txt"), ["None", "None"])

    def test_want_on_user_paused_item_answers_paused_and_stays_paused(self):
        i = self.ready("paused")
        self.svc.resume_all()
        r = self.svc._on_message({"cmd": "want", "id": i, "k": 3})
        self.assertEqual(r, {"ok": False, "error": "paused"})
        time.sleep(0.4)
        self.assertEqual(self.state(i), "paused")                              # the player does not override the user
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.pids(i), [])
        self.assertEqual(self.svc._on_message({"cmd": "want", "id": i, "k": 3})["error"], "paused")   # re-sent: same

    def test_want_under_pause_all_answers_paused_then_watched_item_runs_first(self):
        other = self.ready("queued")                                           # first in the queue
        i = self.ready("queued", "listen")
        r = self.svc._on_message({"cmd": "want", "id": i, "k": 2})
        self.assertEqual(r, {"ok": False, "error": "paused"})
        time.sleep(0.4)
        self.assertEqual(self.state(i), "queued")                              # pause_all wins, also for the watched item
        self.assertEqual(self.state(other), "queued")
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.pids(i) + self.pids(other), [])
        self.assertEqual(self.svc._watched_ids(), {i})                         # but the want / goto are remembered
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        self.assertEqual(self.state(other), "queued")                          # watched item first, not queue order
        self.assertEqual(self.log_lines(i, "start_k.txt"), ["2"])
        self.assertTrue(self.svc._on_message({"cmd": "want", "id": i, "k": 5})["ok"])   # now it is just a goto
        self.finish_worker()
        self.assertTrue(wait_for(lambda: self.state(i) == "done" and self.state(other) == "done", 20))
        self.assertLessEqual(os.path.getmtime(self.svc.store.get(i).output_path),
                             os.path.getmtime(self.svc.store.get(other).output_path))

    def test_pause_all_stops_running_watched_item(self):
        i = self.ready("queued", "listen")
        self.svc._on_message({"cmd": "watching", "id": i})
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        worker_pid, _ = self.pids(i)[0]
        self.svc.pause_all()
        self.assertTrue(wait_for(lambda: not self.alive(worker_pid) and self.state(i) == "queued", 10))
        time.sleep(0.3)
        self.assertEqual(self.state(i), "queued")                              # not restarted while paused_all
        self.assertEqual(len(self.pids(i)), 1)
        self.assertEqual(self.svc._on_message({"cmd": "want", "id": i, "k": 0})["error"], "paused")

    def test_want_on_failed_item_answers_failed_and_does_not_retry(self):
        i = self.ready("failed")
        self.svc.store.update(i, error="boom")
        self.svc.resume_all()
        r = self.svc._on_message({"cmd": "want", "id": i, "k": 1})
        self.assertEqual(r, {"ok": False, "error": "failed", "message": "boom"})
        time.sleep(0.4)
        self.assertEqual(self.state(i), "failed")
        self.assertEqual(self.pids(i), [])
        self.assertIsNone(self.svc._wk)

    def test_play_on_paused_downloaded_item_requeues_it(self):
        self.player_reply = {"ok": True}
        i = self.ready("paused")                                               # pause_all stays on: nothing starts
        self.assertEqual(self.svc.play(i), {"ok": True, "launched": False})
        self.assertEqual(self.state(i), "queued")                              # explicit user action: no longer paused
        it = self.svc.store.get(i)
        sent = [dict(p) for p in self.players]
        self.assertIsInstance(sent[0].pop("pressed_at"), float)     # T1.3: the press stamp travels with the payload
        self.assertEqual(sent, [{"input": it.source_path, "progressive": True, "queue_id": i,
                                 "work_dir": str(self.svc.store.item_dir(i) / "work"), "title": it.title}])
        self.assertTrue(self.svc.get_state()["paused_all"])
        self.assertEqual(self.svc._on_message({"cmd": "want", "id": i, "k": 0})["error"], "paused")   # only pause_all now

    def test_want_while_player_busy(self):
        i = self.ready("queued")
        other = self.ready("queued")
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.svc.resume_all()
        time.sleep(0.4)
        self.assertEqual(self.state(i), "queued")
        self.assertTrue(self.svc._on_message({"cmd": "want", "id": i, "k": 2})["ok"])
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        time.sleep(0.3)
        self.assertEqual(self.state(other), "queued")                          # busy still holds the others back
        self.assertTrue(self.svc.get_state()["player_busy"])

    def test_want_ignored_without_downloaded_source(self):
        self.svc.pause_all()
        i = self.add("http://h/a")                                             # waiting: nothing downloaded
        self.assertFalse(self.svc._on_message({"cmd": "want", "id": i, "k": 1})["ok"])
        self.assertEqual(self.state(i), "waiting")
        self.assertFalse(self.svc._on_message({"cmd": "want", "id": "nope", "k": 1})["ok"])
        d = self.ready("queued")
        self.svc.store.update(d, state="done")
        self.assertFalse(self.svc._on_message({"cmd": "want", "id": d, "k": 1})["ok"])
        self.assertEqual(self.svc._watched_ids(), set())

    def test_want_runs_worker_and_goto_reaches_running_worker(self):
        i = self.ready("queued", "listen")
        self.assertEqual(self.svc._on_message({"cmd": "want", "id": i, "k": 4})["error"], "paused")   # pause_all: k kept
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        self.assertEqual(self.log_lines(i, "start_k.txt"), ["4"])
        self.assertTrue(self.svc._on_message({"cmd": "want", "id": i, "k": 9})["ok"])    # already running: goto
        self.assertTrue(self.svc.goto(i, 12)["running"])
        self.assertTrue(wait_for(lambda: len(self.log_lines(i, "stdin.log")) == 2, 10))
        self.assertEqual([json.loads(x) for x in self.log_lines(i, "stdin.log")],
                         [{"cmd": "goto", "k": 9}, {"cmd": "goto", "k": 12}])
        self.assertEqual(len(self.pids(i)), 1)                                 # no restart

    def test_clock_reaches_running_worker_only(self):
        i = self.ready("queued", "listen")
        clock = {"cmd": "clock", "id": i, "pos": 12.5, "playing": True}
        self.assertEqual(self.svc._on_message(dict(clock)), {"ok": False})            # no worker running yet
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        self.assertEqual(self.svc._on_message(dict(clock)), {"ok": True})
        self.assertEqual(self.svc._on_message({"cmd": "clock", "id": i, "pos": 13, "playing": False}), {"ok": True})
        self.assertFalse(self.svc._on_message({"cmd": "clock", "id": i, "pos": "x", "playing": True})["ok"])
        self.assertFalse(self.svc._on_message({"cmd": "clock", "id": "nope", "pos": 1.0, "playing": True})["ok"])
        other = self.add("http://h/other")
        self.assertEqual(self.svc._on_message({"cmd": "clock", "id": other, "pos": 1.0, "playing": True}),
                         {"ok": False})                                               # not the running item
        self.assertTrue(wait_for(lambda: len(self.log_lines(i, "stdin.log")) == 2, 10))
        self.assertEqual([json.loads(x) for x in self.log_lines(i, "stdin.log")],
                         [{"cmd": "clock", "pos": 12.5, "playing": True}, {"cmd": "clock", "pos": 13.0, "playing": False}])
        self.assertEqual(len(self.pids(i)), 1)
        self.svc.pause(i)
        self.assertTrue(wait_for(lambda: self.state(i) == "paused" and not self.svc._wk, 10))
        self.assertEqual(self.svc._on_message(dict(clock)), {"ok": False})            # worker gone again

    def test_goto_when_not_running_sets_start_k_once(self):
        i = self.ready("queued", "listen")
        self.assertEqual(self.svc.goto(i, 6), {"ok": True, "running": False})
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        self.assertEqual(self.log_lines(i, "start_k.txt"), ["6"])
        self.assertEqual(json.loads((self.svc.store.item_dir(i) / "job.json").read_text())["start_k"], 6)
        self.assertNotIn(i, self.svc._start_k)                                 # cleared after use
        self.svc.pause(i)
        self.assertTrue(wait_for(lambda: self.state(i) == "paused" and not self.svc._wk, 10))
        self.svc.resume(i)
        self.assertTrue(wait_for(lambda: len(self.pids(i)) == 2, 20))
        self.assertEqual(self.log_lines(i, "start_k.txt"), ["6", "None"])
        self.assertNotIn("start_k", json.loads((self.svc.store.item_dir(i) / "job.json").read_text()))

    def test_watching_expiry_ends_priority(self):
        i = self.ready("queued", "listen")
        self.svc.resume_all()
        self.svc._on_message({"cmd": "player_busy", "busy": True})             # busy holds back unwatched items
        self.assertFalse(self.svc._on_message({"cmd": "watching", "id": "nope"})["ok"])
        time.sleep(0.3)
        self.assertEqual(self.state(i), "queued")
        self.assertTrue(self.svc._on_message({"cmd": "watching", "id": i})["ok"])
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))   # despite player_busy
        self.assertEqual(self.svc.store.protected_ids, {i})
        self.now[0] += QS.queue_ipc.BUSY_EXPIRE - 1
        self.svc._on_message({"cmd": "watching", "id": i})                     # heartbeat renews it
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.now[0] += QS.queue_ipc.BUSY_EXPIRE - 1
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        time.sleep(0.3)
        self.assertEqual(self.state(i), "processing")
        self.now[0] += 2                                                       # no watching heartbeat for BUSY_EXPIRE s
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.assertTrue(wait_for(lambda: self.state(i) == "queued", 10))       # player_busy holds it again
        self.assertEqual(self.svc.store.protected_ids, set())

    def test_work_dir_kept_while_watched_then_deleted(self):
        i = self.ready("queued", "listen")
        work = self.svc.store.item_dir(i) / "work"
        work.mkdir(parents=True)
        (work / "seg_0.mkv").write_bytes(b"x")
        self.svc._on_message({"cmd": "watching", "id": i})
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        self.finish_worker()
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        time.sleep(0.4)
        self.assertTrue((work / "seg_0.mkv").exists())                         # still watched
        self.svc._on_message({"cmd": "watching", "id": i})
        self.now[0] += QS.queue_ipc.BUSY_EXPIRE - 1
        time.sleep(0.3)
        self.assertTrue(work.exists())
        self.now[0] += 2
        self.assertTrue(wait_for(lambda: not work.exists(), 10))
        self.assertTrue(os.path.isfile(self.svc.store.get(i).output_path))

    def test_work_dir_deleted_at_once_when_not_watched(self):
        i = self.ready("queued")
        work = self.svc.store.item_dir(i) / "work"
        work.mkdir(parents=True)
        (work / "seg_0.mkv").write_bytes(b"x")
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertTrue(wait_for(lambda: not work.exists(), 10))

    def test_work_dir_delete_is_retried_when_locked(self):
        real = shutil.rmtree
        calls = []

        def flaky(path, *a, **kw):
            if Path(path).name == "work" and len(calls) < 2:
                calls.append(path)                  # the player still has a file open: nothing could be deleted
                return
            return real(path, *a, **kw)
        QS.shutil.rmtree = flaky
        try:
            i = self.ready("queued")
            work = self.svc.store.item_dir(i) / "work"
            work.mkdir(parents=True)
            self.svc.resume_all()
            self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
            self.assertTrue(wait_for(lambda: not work.exists(), 10))
            self.assertEqual(len(calls), 2)
        finally:
            QS.shutil.rmtree = real

    def test_work_dir_of_done_items_deleted_on_start(self):
        root = self.tmp / "root3"
        store = queue_store.QueueStore(root)
        it = store.add("http://h/old", "url", {}, None, {})
        out = store.item_dir(it.id) / "output.mkv"
        (store.item_dir(it.id) / "work").mkdir(parents=True)
        (store.item_dir(it.id) / "work" / "seg_0.mkv").write_bytes(b"x")
        out.write_bytes(b"o")
        store.update(it.id, state="done", output_path=str(out))
        svc = QS.QueueService(root, player_name=self.name + "b", downloader=self.dl, tick=0.02)
        try:
            self.assertTrue(svc.start())
            self.assertTrue(wait_for(lambda: not (store.item_dir(it.id) / "work").exists(), 10))
            self.assertTrue(out.exists())
        finally:
            svc.stop()

    def test_eviction_skips_watched_items(self):
        i = self.ready("queued")
        self.svc.store.update(i, state="done", watched=True)
        (self.svc.store.item_dir(i) / "big.bin").write_bytes(b"b" * 5000)
        self.svc.store.set_settings(cap_gb=0.000002)                           # ~2 KB
        self.svc._on_message({"cmd": "watching", "id": i})
        self.assertFalse(self.svc.store.ensure_space(10))
        self.assertIsNotNone(self.svc.store.get(i))
        self.now[0] += QS.queue_ipc.BUSY_EXPIRE + 1
        self.svc._watched_ids()
        self.assertTrue(self.svc.store.ensure_space(10))
        self.assertIsNone(self.svc.store.get(i))

    def test_ipc_roundtrip(self):
        i = self.ready("queued")
        r = queue_ipc.send_to_queue({"cmd": "want", "id": i, "k": 0}, self.name, timeout=5)
        self.assertEqual(r, {"ok": False, "error": "paused"})                  # pause_all (ready): replies reach the player
        self.svc.resume_all()
        self.assertTrue(queue_ipc.send_to_queue({"cmd": "want", "id": i, "k": 0}, self.name, timeout=5)["ok"])
        self.assertTrue(queue_ipc.send_to_queue({"cmd": "watching", "id": i}, self.name, timeout=5)["ok"])
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))


class SpaceTests(ServiceTestBase):
    def test_space_full_notice_then_proceeds(self):
        asked = []
        room = [False]

        def ensure(n):
            asked.append(n)
            return room[0]
        self.svc.store.ensure_space = ensure
        i = self.add("http://h/a")
        self.assertTrue(wait_for(lambda: self.svc.get_state()["notice"] == "空間不足，佇列已暫停"))
        self.assertEqual(self.state(i), "waiting")
        self.assertEqual(self.dl.calls, [])
        self.assertEqual(asked[0], QS.DEFAULT_DOWNLOAD_ESTIMATE)
        room[0] = True
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertEqual(self.svc.get_state()["notice"], "")
        self.assertIn(1500, asked)                                  # process estimate = source size (1000) x 1.5

    def test_process_stage_blocked(self):
        room = [False]
        self.svc.store.ensure_space = lambda n: room[0] or n == QS.DEFAULT_DOWNLOAD_ESTIMATE
        i = self.add("http://h/a")
        self.assertTrue(wait_for(lambda: self.svc.get_state()["notice"] == QS.SPACE_NOTICE))
        self.assertEqual(self.state(i), "queued")
        room[0] = True
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))


class SleepShutdownTests(ServiceTestBase):
    def test_prevent_sleep_flags(self):
        i = self.add("http://h/a")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertTrue(wait_for(lambda: self.exec_states and self.exec_states[-1] == QS.ES_CONTINUOUS))
        self.assertIn(QS.ES_CONTINUOUS | QS.ES_SYSTEM_REQUIRED, self.exec_states)

    def test_no_sleep_flags_when_disabled(self):
        self.svc.set_settings({"prevent_sleep": False})
        i = self.add("http://h/a")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertNotIn(QS.ES_CONTINUOUS | QS.ES_SYSTEM_REQUIRED, self.exec_states)

    def test_shutdown_once_when_everything_done(self):
        self.assertEqual(self.shutdowns, [])
        time.sleep(0.2)
        self.assertEqual(self.shutdowns, [])                      # idle at startup: nothing ran, nothing to shut down
        self.svc.set_settings({"shutdown_when_done": True})
        a = self.add("http://h/a")
        b = self.add("http://h/b")
        self.assertTrue(wait_for(lambda: self.shutdowns, 30))
        self.assertEqual(self.state(a), "done")
        self.assertEqual(self.state(b), "done")
        time.sleep(0.4)
        self.assertEqual(self.shutdowns, [["shutdown", "/s", "/t", "60"]])
        self.assertTrue(self.svc.get_state()["shutdown_pending"])
        self.svc.cancel_shutdown()
        self.assertEqual(self.shutdowns[-1], ["shutdown", "/a"])
        self.assertFalse(self.svc.store.get_settings()["shutdown_when_done"])
        self.assertFalse(self.svc.get_state()["shutdown_pending"])

    def test_no_shutdown_while_something_is_paused(self):
        self.svc.set_settings({"shutdown_when_done": True})
        a = self.add("http://h/a")
        b = self.add("http://h/b")
        self.svc.pause(b)
        self.assertTrue(wait_for(lambda: self.state(a) == "done", 20))
        time.sleep(0.4)
        self.assertEqual(self.shutdowns, [])


class IpcTests(ServiceTestBase):
    def send(self, payload):
        return queue_ipc.send_to_queue(payload, self.name, timeout=5)

    def test_add_lookup_watched_show_busy(self):
        shown = []
        self.svc.on_show.append(lambda: shown.append(1))
        self.svc.pause_all()                                       # keep items in their initial state
        r = self.send({"cmd": "add", "input": "http://example.com/v.mp4", "headers": {"Referer": "http://r/"},
                       "quality": 720, "title": "Vid", "settings": {"detection_model": "v4-fast"}})
        self.assertTrue(r["ok"])
        item = self.svc.store.get(r["id"])
        self.assertEqual(item.target, "http://example.com/v.mp4")
        self.assertEqual(item.headers, {"Referer": "http://r/"})
        self.assertEqual(item.title, "Vid")
        self.assertEqual(item.quality, 720)

        look = self.send({"cmd": "lookup", "input": "http://example.com/v.mp4", "quality": "720"})
        self.assertEqual(look["item"]["id"], r["id"])
        self.assertEqual(look["item"]["state"], "waiting")
        self.assertEqual(self.send({"cmd": "lookup", "input": "http://example.com/v.mp4", "quality": 480})["item"], None)
        self.assertEqual(self.send({"cmd": "lookup", "input": "http://example.com/other", "quality": 720})["item"], None)

        # same target + quality with different settings: the done one is preferred
        out = self.tmp / "done.mkv"
        out.write_bytes(b"d")
        other = self.svc.store.add("http://example.com/v.mp4", "url", {}, 720, {"detection_model": "other"})
        self.svc.store.update(other.id, state="done", output_path=str(out))
        look = self.send({"cmd": "lookup", "input": "http://example.com/v.mp4", "quality": 720})
        self.assertEqual(look["item"]["id"], other.id)
        self.assertEqual(look["item"]["output"], str(out))

        self.assertFalse(self.svc.store.get(other.id).watched)
        before = self.svc.store.get(other.id).last_used
        time.sleep(0.01)
        self.assertTrue(self.send({"cmd": "watched", "id": other.id})["ok"])
        got = self.svc.store.get(other.id)
        self.assertTrue(got.watched)
        self.assertGreater(got.last_used, before)

        self.assertTrue(self.send({"cmd": "show"})["ok"])
        self.assertEqual(shown, [1])

        self.assertTrue(self.send({"cmd": "player_busy", "busy": True, "pid": 123})["ok"])
        self.assertTrue(self.svc.get_state()["player_busy"])
        self.assertTrue(self.send({"cmd": "player_busy", "busy": False, "pid": 123})["ok"])
        self.assertFalse(self.svc.get_state()["player_busy"])

        self.assertFalse(self.send({"cmd": "nope"})["ok"])

    def test_second_service_with_same_name_does_not_start(self):
        other = QS.QueueService(self.tmp / "root2", player_name=self.name, downloader=self.dl)
        self.assertFalse(other.start())


class StateTests(ServiceTestBase):
    def test_headers_never_in_state(self):
        r = self.svc.add("http://example.com/x.mp4\nCookie: SECRETVALUE123\nReferer: http://ref/")
        self.assertTrue(r["ok"])
        self.svc.pause_all()
        st = self.svc.get_state()
        blob = json.dumps(st, ensure_ascii=False)       # must be JSON-able
        self.assertNotIn("SECRETVALUE123", blob)
        self.assertNotIn("http://ref/", blob)
        it = [x for x in st["items"] if x["id"] == r["id"]][0]
        self.assertNotIn("headers", it)
        self.assertTrue(it["has_headers"])
        self.assertTrue(self.svc.clear_headers(r["id"])["ok"])
        it = [x for x in self.svc.get_state()["items"] if x["id"] == r["id"]][0]
        self.assertFalse(it["has_headers"])
        for key in ("items", "settings", "usage_bytes", "cap_bytes", "paused_all", "player_busy", "notice"):
            self.assertIn(key, st)

    def test_add_errors_move_settings_keep(self):
        self.assertFalse(self.svc.add("   ")["ok"])
        self.svc.pause_all()
        a = self.svc.add("http://example.com/a.mp4")["id"]
        b = self.svc.add("http://example.com/b.mp4")["id"]
        self.svc.move(b, 0)
        self.assertEqual([x["id"] for x in self.svc.get_state()["items"]], [b, a])
        self.assertTrue(self.svc.set_keep(a, True)["ok"])
        self.assertTrue(self.svc.store.get(a).keep)
        s = self.svc.set_settings({"cap_gb": "20", "bogus": 1, "encoder": "h264_nvenc"})["settings"]
        self.assertEqual(s["cap_gb"], 20.0)
        self.assertNotIn("bogus", s)
        self.assertEqual(self.svc.get_state()["cap_bytes"], 20 * 1024 ** 3)
        self.assertFalse(self.svc.pause("nope")["ok"])

    def test_retired_igpu_setting_is_rejected_as_unknown(self):
        self.svc.set_settings({"subtitle_translation_igpu": True})
        self.assertNotIn("subtitle_translation_igpu", self.svc.store.get_settings())

    def test_open_folder(self):
        opened = []
        self.svc._startfile = opened.append
        i = self.add("http://h/a")
        self.assertTrue(self.svc.open_folder(i)["ok"])
        self.assertTrue(self.svc.open_folder(None)["ok"])
        self.assertEqual(opened, [str(self.svc.store.item_dir(i)), str(self.svc.store.root)])


class PlayExportTests(ServiceTestBase):
    def make_done(self, name="v"):
        i = self.add("http://h/" + name)
        self.svc.pause_all()
        out = self.svc.store.item_dir(i) / "output.mkv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"m" * 10)
        self.svc.store.update(i, state="done", output_path=str(out), title="My Title")
        return i, out

    def test_play_prefers_running_player(self):
        i, out = self.make_done()
        self.player_reply = {"ok": True}
        before = self.svc.store.get(i).last_used
        time.sleep(0.01)
        r = self.svc.play(i)
        self.assertTrue(r["ok"])
        self.assertFalse(r["launched"])
        self.assertEqual(self.players, [{"input": str(out), "headers": {}, "processed": True, "queue_id": i,
                                         "title": "My Title"}])
        self.assertEqual(self.launched, [])
        self.assertGreater(self.svc.store.get(i).last_used, before)

    def test_play_marks_seen_in_any_state(self):
        self.player_reply = {"ok": True}
        i, _out = self.make_done()
        self.assertFalse(self.svc.store.get(i).seen)
        self.assertTrue(self.svc.play(i)["ok"])
        self.assertTrue(self.svc.store.get(i).seen)
        j = self.ready("processing")                              # progressive playback of an unfinished item
        self.assertFalse(self.svc.store.get(j).seen)
        self.assertTrue(self.svc.play(j)["ok"])
        self.assertTrue(self.svc.store.get(j).seen)
        self.assertTrue(self.svc.get_state()["items"][0]["seen"])

    def test_failed_play_does_not_mark_seen(self):
        i = self.ready("failed")
        self.assertFalse(self.svc.play(i)["ok"])
        self.assertFalse(self.svc.store.get(i).seen)

    def test_ipc_player_start_marks_seen(self):
        i, _out = self.make_done("lk")
        self.assertFalse(self.svc.store.get(i).seen)
        r = self.svc._on_message({"cmd": "lookup", "input": "http://h/lk"})
        self.assertEqual(r["item"]["id"], i)
        self.assertTrue(self.svc.store.get(i).seen)
        j = self.ready("processing")
        self.assertFalse(self.svc.store.get(j).seen)
        self.svc._on_message({"cmd": "watching", "id": j})
        self.assertTrue(self.svc.store.get(j).seen)
        k = self.ready("queued")
        self.svc._on_message({"cmd": "want", "id": k, "k": 0})
        self.assertTrue(self.svc.store.get(k).seen)

    def test_mark_all_seen(self):
        a, _ = self.make_done("a")
        b, _ = self.make_done("b")
        self.assertEqual(self.svc.mark_all_seen(), {"ok": True, "count": 2})
        self.assertTrue(all(it.seen for it in self.svc.store.items()))
        self.assertTrue(all(d["seen"] for d in self.svc.get_state()["items"]))

    def test_play_launches_player_when_none_answers(self):
        i, out = self.make_done()
        self.player_reply = None
        r = self.svc.play(i)
        self.assertTrue(r["launched"])
        self.assertEqual(len(self.launched), 1)
        path, payload = self.launched[0]
        self.assertEqual(path, str(out))
        self.assertTrue(payload["processed"])

    def test_play_requires_something_to_show(self):
        i = self.ready("failed")                                  # failed, no segments.json
        self.assertFalse(self.svc.play(i)["ok"])
        self.svc.store.update(i, state="done", output_path=str(self.tmp / "missing.mkv"))
        self.assertFalse(self.svc.play(i)["ok"])
        self.assertEqual(self.players, [])

    def test_play_progressive_when_source_downloaded(self):
        self.player_reply = {"ok": True}
        for state in ("queued", "processing", "paused"):
            i = self.ready(state)
            self.players.clear()
            self.assertEqual(self.svc.play(i), {"ok": True, "launched": False})
            it = self.svc.store.get(i)
            sent = [dict(p) for p in self.players]
            self.assertIsInstance(sent[0].pop("pressed_at"), float)     # T1.3: the press stamp travels with the payload
            self.assertEqual(sent, [{"input": it.source_path, "progressive": True, "queue_id": i,
                                     "work_dir": str(self.svc.store.item_dir(i) / "work"),
                                     "title": it.title}])
        f = self.ready("failed")
        work = self.svc.store.item_dir(f) / "work"
        work.mkdir(parents=True)
        (work / "segments.json").write_text("{}")
        self.players.clear()
        self.assertTrue(self.svc.play(f)["ok"])
        self.assertTrue(self.players[0]["progressive"])
        self.assertEqual(self.players[0]["work_dir"], str(work))

    def test_play_waits_for_queue_source_without_opening_a_second_live_stream(self):
        self.player_reply = {"ok": True}
        self.svc.pause_all()
        w = self.add("http://h/w", headers={"Cookie": "c1"})
        d = self.add("http://h/d", headers={"Referer": "http://r/"})
        self.svc.store.update(d, state="downloading")
        for i in (w, d):
            self.players.clear()
            self.assertEqual(self.svc.play(i), {"ok": True, "pending": True})
            self.assertEqual(self.players, [])
        p = self.add("http://h/p")
        self.svc.store.update(p, state="paused")                  # paused before its download finished
        self.players.clear()
        self.assertEqual(self.svc.play(p), {"ok": True, "pending": True})
        self.assertEqual(self.players, [])

    def test_downloading_play_reaches_player_open_inbox(self):
        from oh_streamer.player.controller import PlayerController
        from types import SimpleNamespace
        import queue

        self.svc.pause_all()
        item = self.add("http://h/downloading", headers={"Referer": "http://h/"})
        self.svc.store.update(item, state="downloading")
        player = PlayerController(None, None, deps=SimpleNamespace())
        player.inbox = queue.Queue()
        self.svc._send_to_player = player.on_message

        self.assertEqual(self.svc.play(item), {"ok": True, "pending": True})
        self.assertTrue(player.inbox.empty())
        self.assertEqual(self.svc.store.get(item).state, "downloading")

    def test_play_progressive_launches_player_when_none_answers(self):
        i = self.ready("processing")
        self.player_reply = None
        self.assertTrue(self.svc.play(i)["launched"])
        path, payload = self.launched[0]
        self.assertEqual(path, self.svc.store.get(i).source_path)
        self.assertTrue(payload["progressive"])

    def test_play_start_reaches_processed_and_progressive_payloads(self):
        self.player_reply = {"ok": True}
        i, out = self.make_done()
        self.assertTrue(self.svc.play(i, "12.5")["ok"])
        self.assertEqual(self.players[-1]["start"], 12.5)
        self.assertEqual(self.players[-1]["input"], str(out))
        p = self.ready("processing")
        self.assertTrue(self.svc.play(p, 90)["ok"])
        self.assertEqual(self.players[-1]["start"], 90.0)
        self.assertTrue(self.players[-1]["progressive"])

    def test_play_seeds_the_seek_clock_before_contacting_the_player(self):
        item_id = self.ready("processing")
        observed = []

        def send(payload):
            with self.svc._lock:
                self.assertIn(item_id, self.svc._watch)
                self.assertEqual(self.svc._subtitle_clocks[item_id], (2400.0, False))
                self.assertEqual(self.svc._subtitle_wants[item_id], (2400.0, self.now[0]))
                self.assertEqual(self.svc._subtitle_seek_guard[item_id], (2400.0, self.now[0]))
            self.assertEqual(payload["start"], 2400.0)
            observed.append(payload)
            return {"ok": True}

        with patch.object(self.svc, "_source_facts", return_value={"duration": 3600.0}), \
                patch.object(self.svc, "_send_to_player", side_effect=send):
            self.assertTrue(self.svc.play(item_id, start=2400)["ok"])
        self.assertEqual(len(observed), 1)

    def test_pending_play_seeds_the_seek_clock_before_the_source_is_ready(self):
        self.svc.pause_all()
        item_id = self.add("http://h/pending-seek")
        with patch.object(self.svc, "_source_facts", return_value={"duration": 3600.0}):
            self.assertEqual(self.svc.play(item_id, start=2400), {"ok": True, "pending": True})
        with self.svc._lock:
            self.assertIn(item_id, self.svc._watch)
            self.assertEqual(self.svc._subtitle_clocks[item_id], (2400.0, False))
            self.assertEqual(self.svc._subtitle_wants[item_id], (2400.0, self.now[0]))
            self.assertEqual(self.svc._subtitle_seek_guard[item_id], (2400.0, self.now[0]))
        self.assertEqual(self.players, [])

    def test_play_invalid_start_is_ignored_and_none_is_unchanged(self):
        self.player_reply = {"ok": True}
        i, _out = self.make_done()
        for bad in (None, "x", float("nan"), float("inf"), -1, "-3", True, [], ""):
            self.players.clear()
            self.assertTrue(self.svc.play(i, bad)["ok"])
            self.assertNotIn("start", self.players[0], bad)
        self.players.clear()
        self.assertTrue(self.svc.play(i, 0)["ok"])
        self.assertEqual(self.players[0]["start"], 0.0)

    def test_play_start_is_clamped_to_known_duration(self):
        self.player_reply = {"ok": True}
        p = self.ready("processing")
        with patch.object(self.svc, "_source_facts", return_value={"duration": 60.0}):
            self.assertTrue(self.svc.play(p, 500)["ok"])
        self.assertEqual(self.players[-1]["start"], 60.0)

    def test_play_start_is_passed_to_a_launched_player(self):
        i, _out = self.make_done()
        self.player_reply = None
        self.assertTrue(self.svc.play(i, 33)["launched"])
        self.assertEqual(self.launched[0][1]["start"], 33.0)

    def test_pending_play_keeps_start_until_source_is_ready(self):
        self.svc.pause_all()
        w = self.add("http://h/w")
        self.assertEqual(self.svc.play(w, 20), {"ok": True, "pending": True})
        self.assertEqual(self.svc._pending_start, 20.0)
        self.assertEqual(self.svc.play(w), {"ok": True, "pending": True})     # a later play without start resets it
        self.assertIsNone(self.svc._pending_start)
        self.assertEqual(self.svc.play(w, 20), {"ok": True, "pending": True})
        self.player_reply = {"ok": True}
        with patch.object(self.svc, "_source_ready", return_value=True), \
                patch.object(self.svc, "_source_payload", return_value={"input": "s.mp4", "progressive": True}):
            self.assertTrue(self.svc._pending_play_ready(w))
        self.assertEqual(self.players[-1]["start"], 20.0)
        self.assertIsNone(self.svc._pending_start)

    def test_build_export_cmd(self):
        cmd = QS.build_export_cmd("in.mkv", "out.mp4", [("video", "hevc"), ("audio", "aac")])
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertEqual(cmd[cmd.index("-i") + 1], "in.mkv")
        self.assertEqual(cmd[-1], "out.mp4")
        self.assertEqual(cmd[cmd.index("-c") + 1], "copy")
        self.assertEqual(cmd[cmd.index("-tag:v") + 1], "hvc1")
        self.assertEqual(cmd[cmd.index("-movflags") + 1], "+faststart")
        self.assertNotIn("-c:a", cmd)
        h264 = QS.build_export_cmd("in.mkv", "out.mp4", [("video", "h264"), ("audio", "aac")])
        self.assertNotIn("-tag:v", h264)
        dts = QS.build_export_cmd("in.mkv", "out.mp4", [("video", "hevc"), ("audio", "aac"), ("audio", "dts")])
        self.assertEqual(dts[dts.index("-c:a") + 1], "aac")
        silent = QS.build_export_cmd("in.mkv", "out.mp4", [("video", "hevc")])
        self.assertNotIn("-c:a", silent)
        forced = QS.build_export_cmd("in.mkv", "out.mp4", [("video", "hevc"), ("audio", "aac")], force_aac=True)
        self.assertEqual(forced[forced.index("-c:a") + 1], "aac")

    def test_export_runs_ffmpeg_and_moves_result(self):
        i, out = self.make_done()
        cmds = []

        def run(cmd):
            cmds.append(cmd)
            Path(cmd[-1]).write_bytes(b"mp4")
            return types.SimpleNamespace(returncode=0, stderr="")
        self.svc._run_cmd = run
        self.svc._probe = lambda p: [("video", "hevc"), ("audio", "aac")]
        dest = self.tmp / "exported.mp4"
        r = self.svc.export(i, str(dest))
        self.assertTrue(r["ok"], r)
        self.assertEqual(dest.read_bytes(), b"mp4")
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0][cmds[0].index("-i") + 1], str(out))
        self.assertIn("hvc1", cmds[0])
        self.assertFalse((self.tmp / "exported.part.mp4").exists())

    def test_export_falls_back_to_aac_when_probe_unknown(self):
        i, _ = self.make_done()
        cmds = []

        def run(cmd):
            cmds.append(cmd)
            if "-c:a" not in cmd:
                return types.SimpleNamespace(returncode=1, stderr="Could not find tag for codec")
            Path(cmd[-1]).write_bytes(b"ok")
            return types.SimpleNamespace(returncode=0, stderr="")
        self.svc._run_cmd = run
        self.svc._probe = lambda p: None
        r = self.svc.export(i, str(self.tmp / "e"))              # missing .mp4 suffix gets added
        self.assertTrue(r["ok"], r)
        self.assertEqual(len(cmds), 2)
        self.assertTrue((self.tmp / "e.mp4").exists())

    def test_export_failure_reports_error(self):
        i, _ = self.make_done()
        self.svc._run_cmd = lambda cmd: types.SimpleNamespace(returncode=1, stderr="bad things")
        self.svc._probe = lambda p: [("video", "hevc")]
        r = self.svc.export(i, str(self.tmp / "x.mp4"))
        self.assertFalse(r["ok"])
        self.assertIn("bad things", r["error"])


FFMETA = """;FFMETADATA1
[CHAPTER]
TIMEBASE=1/1000
START=0
END=1500
title=one
[CHAPTER]
TIMEBASE=1/1000
START=1500
END=3000
title=two
"""


def make_compare_mkv(path: Path, titles=("oh:processed", "oh:original"), seconds=3, audio=True) -> None:
    """A tiny real mkv: one titled h264 video track per title, optional aac audio, two chapters, global tags."""
    meta = path.with_suffix(".ffmeta")
    meta.write_text(FFMETA, encoding="utf-8")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for _t in titles:
        cmd += ["-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=10:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={seconds}"]
    cmd += ["-i", str(meta)]
    for n in range(len(titles)):
        cmd += ["-map", str(n)]
    if audio:
        cmd += ["-map", str(len(titles))]
    cmd += ["-map_metadata", str(len(titles) + (1 if audio else 0)), "-c:v", "libx264", "-preset", "ultrafast"]
    if audio:
        cmd += ["-c:a", "aac"]
    for n, t in enumerate(titles):
        if t:
            cmd += [f"-metadata:s:v:{n}", f"title={t}"]
    cmd += ["-metadata", "OH_COMPARE=1", "-metadata", "OH_ORIGINAL=copy", "-metadata", "OH_RATE_MULT=2",
            "-metadata", "OH_VIDEO_START=0.5", "-metadata", "OH_SOURCE_FPS=10/1", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def ffprobe_all(path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-show_chapters", "-of", "json",
                        str(path)], capture_output=True, text=True)
    return json.loads(p.stdout)


class CompareStorageTests(ServiceTestBase):
    def test_state_with_unicode_track_titles_does_not_depend_on_windows_codepage(self):
        item_id, output = self.done_item(titles=("繁體中文（AI）",))
        real_run = QS.subprocess.run

        def cp950_default(*args, **kwargs):
            if args and args[0][0] == "ffprobe" and kwargs.get("text"):
                kwargs.setdefault("encoding", "cp950")
            return real_run(*args, **kwargs)

        with patch.object(QS.subprocess, "run", side_effect=cp950_default):
            info = QS.probe_tracks(str(output))
            self.assertIsNotNone(info)
            self.assertEqual(info["streams"][0]["title"], "繁體中文（AI）")
            self.assertEqual(self.view(item_id)["id"], item_id)

    def test_empty_probe_stdout_does_not_break_completed_item_state(self):
        item_id, _ = self.done_item()
        empty = subprocess.CompletedProcess([], 0, stdout=None, stderr="")
        with patch.object(QS.subprocess, "run", return_value=empty):
            self.assertIsNone(QS.probe_tracks("unused"))
            self.assertIsNone(QS.probe_streams("unused"))
            self.assertIsNone(QS.probe_video_facts("unused"))
            self.assertEqual(self.view(item_id)["id"], item_id)

    def done_item(self, titles=("oh:processed", "oh:original"), **kw):
        i = self.add("http://h/" + uuid.uuid4().hex[:6])
        self.svc.pause_all()
        out = self.svc.store.item_dir(i) / "output.mkv"
        out.parent.mkdir(parents=True, exist_ok=True)
        make_compare_mkv(out, titles, **kw)
        self.svc.store.update(i, state="done", output_path=str(out), size_bytes=out.stat().st_size)
        return i, out

    def view(self, i):
        return [x for x in self.svc.get_state()["items"] if x["id"] == i][0]

    def test_job_settings_compare_on_and_off(self):
        self.svc.set_settings({"save_compare": True, "compare_original": "encode", "compare_quality": "tiny"})
        i = self.add("http://h/a")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        job = json.loads((self.svc.store.item_dir(i) / "job.json").read_text(encoding="utf-8"))
        self.assertEqual((job["settings"]["compare"], job["settings"]["compare_original"],
                          job["settings"]["compare_quality"]), ("on", "encode", "tiny"))
        self.assertEqual(self.svc.store.get(i).settings["compare_original"], "encode")

    def test_job_settings_compare_off_omits_options(self):
        self.svc.set_settings({"save_compare": False, "compare_original": "encode"})
        i = self.add("http://h/b")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        job = json.loads((self.svc.store.item_dir(i) / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["settings"]["compare"], "off")
        self.assertNotIn("compare_original", job["settings"])
        self.assertNotIn("compare_quality", job["settings"])

    def test_settings_validation(self):
        s = self.svc.set_settings({"compare_original": "encode", "compare_quality": "tiny"})["settings"]
        self.assertEqual((s["compare_original"], s["compare_quality"]), ("encode", "tiny"))
        s = self.svc.set_settings({"compare_original": "bogus", "compare_quality": 7})["settings"]
        self.assertEqual((s["compare_original"], s["compare_quality"]), ("encode", "tiny"))
        st = self.svc.get_state()["settings"]
        self.assertEqual((st["compare_original"], st["compare_quality"]), ("encode", "tiny"))

    def test_has_compare_flag_and_cache(self):
        a, _ = self.done_item()
        b, _ = self.done_item(titles=("oh:processed",))
        c, _ = self.done_item(titles=(None,))
        waiting = self.add("http://h/w")
        calls = []
        real = self.svc._probe_tracks
        self.svc._probe_tracks = lambda p: (calls.append(p) or real(p))
        self.assertTrue(self.view(a)["has_compare"])
        self.assertEqual(self.view(a)["compare_original"], "copy")          # from the OH_ORIGINAL tag
        self.assertFalse(self.view(b)["has_compare"])
        self.assertIsNone(self.view(b)["compare_original"])
        self.assertFalse(self.view(c)["has_compare"])
        self.assertFalse(self.view(waiting)["has_compare"])
        self.assertIsNone(self.view(waiting)["compare_original"])
        n = len(calls)
        self.assertEqual(n, 3)                                              # one probe per done output
        self.view(a)
        self.svc.get_state()
        self.assertEqual(len(calls), n)                                     # cached by path + mtime

    def test_has_compare_for_stored_restored_track(self):
        i, _ = self.done_item(titles=("oh:processed", "oh:original", "oh:restored"))
        self.assertTrue(self.view(i)["has_compare"])
        j, _ = self.done_item(titles=("oh:processed", "oh:restored"))
        self.assertTrue(self.view(j)["has_compare"])

    def test_compare_original_from_item_settings(self):
        i, _ = self.done_item()
        self.svc.store.update(i, settings={"compare_original": "encode"})
        self.assertEqual(self.view(i)["compare_original"], "encode")

    def test_remove_compare_end_to_end(self):
        i, out = self.done_item()
        before = out.stat().st_size
        v0 = [s for s in ffprobe_all(out)["streams"] if s["codec_type"] == "video"]
        self.assertEqual(len(v0), 2)
        self.assertTrue(self.view(i)["has_compare"])
        r = self.svc.remove_compare(i)
        self.assertTrue(r["ok"], r)
        after = out.stat().st_size
        self.assertLess(after, before)
        self.assertEqual(r["freed_bytes"], before - after)
        self.assertEqual(self.svc.store.get(i).size_bytes, after)
        info = ffprobe_all(out)
        vids = [s for s in info["streams"] if s["codec_type"] == "video"]
        self.assertEqual([s["tags"]["title"] for s in vids], ["oh:processed"])
        self.assertEqual(len([s for s in info["streams"] if s["codec_type"] == "audio"]), 1)
        self.assertEqual([c["tags"]["title"] for c in info["chapters"]], ["one", "two"])
        tags = {k.upper(): v for k, v in info["format"]["tags"].items()}
        self.assertEqual(tags["OH_COMPARE"], "0")
        self.assertEqual(tags["OH_ORIGINAL"], "none")
        for k, v in (("OH_RATE_MULT", "2"), ("OH_VIDEO_START", "0.5"), ("OH_SOURCE_FPS", "10/1")):
            self.assertEqual(tags[k], v)
        self.assertLess(abs(float(info["format"]["duration"]) - 3.0), 0.5)
        self.assertFalse((out.parent / "output.nocompare.tmp.mkv").exists())
        v = self.view(i)
        self.assertFalse(v["has_compare"])
        self.assertIsNone(v["compare_original"])
        self.assertEqual(v["size_bytes"], after)
        self.assertFalse(self.svc.remove_compare(i)["ok"])                    # nothing left to remove

    def test_remove_compare_keeps_processed_when_it_is_not_first(self):
        i, out = self.done_item(titles=("oh:original", "oh:processed"))
        self.assertTrue(self.svc.remove_compare(i)["ok"])
        vids = [s for s in ffprobe_all(out)["streams"] if s["codec_type"] == "video"]
        self.assertEqual([s["tags"]["title"] for s in vids], ["oh:processed"])

    def test_remove_compare_untitled_first_track_is_kept(self):
        i, out = self.done_item(titles=(None, "oh:original"))
        self.assertTrue(self.svc.remove_compare(i)["ok"])
        vids = [s for s in ffprobe_all(out)["streams"] if s["codec_type"] == "video"]
        self.assertEqual(len(vids), 1)
        self.assertNotEqual(vids[0].get("tags", {}).get("title"), "oh:original")

    def test_remove_compare_refusals(self):
        self.assertFalse(self.svc.remove_compare("nope")["ok"])
        w = self.add("http://h/w")
        self.assertFalse(self.svc.remove_compare(w)["ok"])
        i, out = self.done_item(titles=("oh:processed",))
        self.assertFalse(self.svc.remove_compare(i)["ok"])
        self.assertTrue(out.exists())

    def test_remove_compare_refused_while_watched(self):
        i, out = self.done_item()
        with self.svc._lock:
            self.svc._watch[i] = self.now[0]
        before = out.stat().st_size
        r = self.svc.remove_compare(i)
        self.assertEqual(r, {"ok": False, "error": QS.PLAYING_ERROR})
        self.assertEqual(out.stat().st_size, before)

    def test_remove_compare_locked_file_reports_playing(self):
        i, out = self.done_item()
        real = os.replace

        def locked(src, dst):
            raise PermissionError(13, "in use")
        os.replace = locked
        try:
            r = self.svc.remove_compare(i)
        finally:
            os.replace = real
        self.assertEqual(r, {"ok": False, "error": QS.PLAYING_ERROR})
        self.assertTrue(out.exists())
        self.assertFalse((out.parent / "output.nocompare.tmp.mkv").exists())
        self.assertTrue(self.svc.remove_compare(i)["ok"])                     # works once the file is free

    def test_remove_compare_duration_mismatch_keeps_original(self):
        i, out = self.done_item()
        before = out.stat().st_size
        real = self.svc._probe_tracks
        seen = []

        def fake(path):
            info = real(path)
            if len(seen) >= 1:      # the second probe is the remuxed temp file
                info = dict(info, duration=info["duration"] + 5)
            seen.append(path)
            return info
        self.svc._probe_tracks = fake
        r = self.svc.remove_compare(i)
        self.assertFalse(r["ok"])
        self.assertEqual(out.stat().st_size, before)
        self.assertFalse((out.parent / "output.nocompare.tmp.mkv").exists())

    def test_remove_compare_ffmpeg_failure(self):
        i, out = self.done_item()
        self.svc._run_cmd = lambda cmd: types.SimpleNamespace(returncode=1, stderr="muxer exploded")
        r = self.svc.remove_compare(i)
        self.assertFalse(r["ok"])
        self.assertIn("muxer exploded", r["error"])

    def test_export_default_is_processed_only_and_compare_keeps_all(self):
        i, out = self.done_item()
        plain = self.tmp / "plain.mp4"
        r = self.svc.export(i, str(plain))
        self.assertTrue(r["ok"], r)
        self.assertEqual(len([s for s in ffprobe_all(plain)["streams"] if s["codec_type"] == "video"]), 1)
        both = self.tmp / "both.mp4"
        r = self.svc.export(i, str(both), include_compare=True)
        self.assertTrue(r["ok"], r)
        vids = [s for s in ffprobe_all(both)["streams"] if s["codec_type"] == "video"]
        self.assertEqual(len(vids), 2)
        self.assertEqual([v.get("tags", {}).get("handler_name") for v in vids], ["oh:processed", "oh:original"])

    def test_export_cmd_include_compare(self):
        streams = [("video", "hevc"), ("video", "h264"), ("audio", "aac")]
        plain = QS.build_export_cmd("in.mkv", "o.mp4", streams)
        self.assertEqual(plain[plain.index("-map") + 1], "0:v:0")
        cmd = QS.build_export_cmd("in.mkv", "o.mp4", streams, include_compare=True,
                                  titles=["oh:processed", "oh:original"])
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:v")
        self.assertEqual(cmd[cmd.index("-tag:v:0") + 1], "hvc1")
        self.assertNotIn("-tag:v:1", cmd)
        self.assertEqual(cmd[cmd.index("-metadata:s:v:1") + 1], "handler_name=oh:original")


class PackageChildLaunchTests(unittest.TestCase):
    def test_installed_worker_uses_package_entry(self):
        service = object.__new__(QS.QueueService)
        service._python = "python"
        with tempfile.TemporaryDirectory() as tmp, patch.object(QS, "HERE", Path(tmp)), \
                patch.object(sys, "frozen", False, create=True):
            self.assertEqual(service._default_worker_command("job.json"),
                             ["python", "-m", "oh_streamer", "worker", "--job", "job.json"])

    def test_installed_player_uses_package_entry_and_user_log(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(QS, "HERE", Path(tmp)), \
                patch.object(sys, "frozen", False, create=True), \
                patch.object(QS.platform_support, "data_dir", return_value=Path(tmp) / "user"), \
                patch.object(QS.subprocess, "Popen") as spawn:
            QS._default_launch_player("clip.mkv", {"processed": True})
            argv = spawn.call_args.args[0]
            self.assertEqual(argv[1:4], ["-m", "oh_streamer", "player"])
            self.assertIn(str(Path(tmp) / "user" / "logs" / "oh_player.log"), argv)


class LaunchStartArgvTests(unittest.TestCase):
    def test_processed_queue_launch_forwards_subtitle_and_watch_metadata(self):
        payload = {"input": "clip.mkv", "processed": True, "queue_id": "q1", "subtitle_error": True,
                   "subtitles": [{"path": "zh.srt", "lang": "chi", "title": "AI", "default": True}]}
        with patch.object(QS.threading, "Thread") as thread, \
                patch.object(QS.queue_ipc, "send_to_player", return_value={"ok": True}) as send, \
                patch.object(QS.time, "sleep"):
            argv = self.launch(payload)
            self.assertNotIn("--processed", argv)
            thread.return_value.start.assert_called_once()
            thread.call_args.kwargs["target"]()
            send.assert_called_once_with(payload, None, timeout=1.5)

    def launch(self, payload):
        with tempfile.TemporaryDirectory() as tmp, patch.object(QS, "HERE", Path(tmp)), \
                patch.object(sys, "frozen", False, create=True), \
                patch.object(QS.platform_support, "data_dir", return_value=Path(tmp) / "user"), \
                patch.object(QS.subprocess, "Popen") as spawn:
            QS._default_launch_player("clip.mkv", payload)
            return spawn.call_args.args[0]

    def test_processed_launch_gets_start_option(self):
        argv = self.launch({"processed": True, "start": 12.5})
        self.assertEqual(argv[argv.index("--start") + 1], "12.500")
        self.assertIn("--processed", argv)

    def test_launch_without_start_or_zero_has_no_start_option(self):
        self.assertNotIn("--start", self.launch({"processed": True}))
        self.assertNotIn("--start", self.launch({"processed": True, "start": 0.0}))


class PureTests(unittest.TestCase):
    def test_no_torch_imported(self):
        import subprocess
        code = "import sys, oh_streamer.queue.queue_service; sys.exit(1 if {'torch', 'oh_streamer.player.oh_player'} & set(sys.modules) else 0)"
        env = dict(os.environ, PYTHONPATH=str(_STREAMER_ROOT / "src"))
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=str(_STREAMER_ROOT), env=env).returncode, 0)


if __name__ == "__main__":
    unittest.main()
