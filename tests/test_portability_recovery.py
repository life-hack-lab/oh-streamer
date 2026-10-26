"""Bounded recovery regressions using real owned processes and isolated state.

Run with the project venv. Localhost tests require permission to bind a loopback
socket; no remote network, models, GPU, player, or shutdown command is used.
"""
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
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

import psutil

import oh_streamer.processing.encoding_backend as EB
import portability_common as PC
import oh_streamer.queue.queue_ipc as queue_ipc
import oh_streamer.queue.queue_service as QS
import oh_streamer.queue.queue_worker as Q
import oh_streamer.player.single_instance as SI


def wait_until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def alive(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        base = _STREAMER_ROOT / ".test-tmp"
        base.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="recovery-", dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def service(self, **kwargs):
        svc = QS.QueueService(self.root / "queue", player_name="recovery-" + uuid.uuid4().hex,
                              set_execution_state=lambda flags: None,
                              run_shutdown=lambda args: self.fail("unexpected shutdown"), **kwargs)
        self.addCleanup(svc.stop)
        return svc

    def ready_item(self, svc):
        source = self.root / "source.mp4"
        source.write_bytes(b"source" * 100)
        item = svc.store.add(str(source), "file", None, None, {}, title="Recovery")
        svc.store.update(item.id, state="queued", source_path=str(source))
        return svc.store.get(item.id)

    def test_timeout_reaps_owned_descendants_and_keeps_partial_diagnostics(self):
        script = self.root / "owned.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "print(child.pid, flush=True)\n"
            "print('diagnostic before timeout', file=sys.stderr, flush=True)\n"
            "time.sleep(120)\n", encoding="utf-8")
        result = PC.run_command([sys.executable, script], timeout=1.5, log_path=self.root / "timeout.log")
        pid = int(result["stdout"].strip())
        self.addCleanup(lambda: QS.kill_tree(pid) if alive(pid) else None)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["returncode"], 0)
        self.assertLess(result["elapsed_s"], 8)
        self.assertTrue(wait_until(lambda: not alive(pid)), "owned child survived command timeout")
        self.assertIn("diagnostic before timeout", (self.root / "timeout.log").read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "real software fallback needs ffmpeg")
    def test_encoder_probe_timeout_falls_back_to_working_software(self):
        # Substitute only the unavailable hardware executable with a real hung
        # process. The timeout, selection, process cleanup, and fallback encode
        # still run through their public paths.
        script = self.root / "hung_encoder.py"
        pidfile = self.root / "encoder.pid"
        script.write_text(
            "import os, pathlib, sys, time\n"
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "time.sleep(120)\n", encoding="utf-8")
        real_run = subprocess.run
        def run_surrogate(argv, **kwargs):
            return real_run([sys.executable, str(script), str(pidfile)], **kwargs)
        with patch.object(EB.subprocess, "run", side_effect=run_surrogate):
            selected = EB.resolve_encoder("h264", lambda name: EB.encoder_usable(name, timeout=0.6),
                                          platform="darwin")
        self.assertEqual(selected, "libx264")
        self.assertTrue(pidfile.exists(), "hardware surrogate did not reach its probe")
        pid = int(pidfile.read_text())
        self.addCleanup(lambda: QS.kill_tree(pid) if alive(pid) else None)
        self.assertTrue(wait_until(lambda: not alive(pid)), "timed-out probe survived fallback")
        self.assertTrue(EB.encoder_usable(selected, timeout=5), "software fallback cannot encode")

    def test_startup_failure_releases_native_instance_lock(self):
        svc = self.service()
        name = queue_ipc.queue_instance_name(svc.player_name)
        with patch.object(SI, "state_dir", return_value=self.root / "state"), \
                patch.object(SI.Instance, "serve", side_effect=OSError("listener startup failed")):
            with self.assertRaisesRegex(OSError, "listener startup failed"):
                svc.start()
            replacement = SI.acquire(name)
            if replacement is not None:
                replacement.close()
            self.assertIsNotNone(replacement, "failed startup kept the queue's native instance lock")
            self.assertIsNone(svc._instance)

    def test_worker_protocol_bad_progress_does_not_hide_later_done(self):
        script = self.root / "events.py"
        script.write_text(
            "import json, pathlib, sys, time\n"
            "job = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
            "for line in ['not JSON', '[]', '{\"type\":\"progress\",\"frame\":\"bad\",\"frames\":10}', "
            "'{\"type\":\"progress\",\"frame\":1,\"frames\":{}}']:\n"
            "    print(line, flush=True)\n"
            "pathlib.Path(job['output']).write_bytes(b'finished')\n"
            "event = json.dumps({'type':'done', 'output':job['output']}) + '\\n'\n"
            "for part in [event[:9], event[9:]]:\n"
            "    sys.stdout.write(part); sys.stdout.flush(); time.sleep(.05)\n", encoding="utf-8")
        svc = self.service(worker_command=lambda job: [sys.executable, str(script), job])
        item = self.ready_item(svc)
        svc._maybe_start_worker()
        self.assertTrue(wait_until(lambda: svc.store.get(item.id).state in ("done", "failed")))
        actual = svc.store.get(item.id)
        self.assertEqual(actual.state, "done", actual.error)
        self.assertEqual(Path(actual.output_path).read_bytes(), b"finished")

    def test_worker_eof_without_done_fails_and_retains_resume_state(self):
        svc = self.service(worker_command=lambda job: [sys.executable, "-c", "print('worker interrupted')"])
        item = self.ready_item(svc)
        work = svc.store.item_dir(item.id) / "work"
        work.mkdir(parents=True)
        (work / "segments.json").write_text('{"resume":"keep"}', encoding="utf-8")
        svc._maybe_start_worker()
        self.assertTrue(wait_until(lambda: svc.store.get(item.id).state == "failed"))
        self.assertTrue(Path(item.source_path).exists())
        self.assertEqual((work / "segments.json").read_text(encoding="utf-8"), '{"resume":"keep"}')
        self.assertFalse((svc.store.item_dir(item.id) / "output.mkv").exists())

    def test_resumed_job_cap_pressure_preserves_segments_then_retries(self):
        script = self.root / "finish.py"
        script.write_text(
            "import json, pathlib, sys\n"
            "job = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
            "pathlib.Path(job['output']).write_bytes(b'finished')\n"
            "print(json.dumps({'type':'done', 'output':job['output']}), flush=True)\n", encoding="utf-8")
        svc = self.service(worker_command=lambda job: [sys.executable, str(script), job])
        item = self.ready_item(svc)
        work = svc.store.item_dir(item.id) / "work"
        work.mkdir(parents=True)
        segment = work / "closed.mkv"
        segment.write_bytes(b"committed segment" * 100)
        svc.store.update(item.id, state="processing")
        # A new service reloads the persisted interrupted job as queued.
        svc = self.service(worker_command=lambda job: [sys.executable, str(script), job])
        self.assertEqual(svc.store.get(item.id).state, "queued")
        svc.store.set_settings(cap_gb=0)
        svc._maybe_start_worker()
        self.assertEqual(svc.store.get(item.id).state, "queued")
        self.assertIsNone(svc._wk)
        self.assertIn(QS.SPACE_NOTICE, svc.get_state()["notice"])
        self.assertEqual(segment.read_bytes(), b"committed segment" * 100)
        self.assertFalse((svc.store.item_dir(item.id) / "job.json").exists())
        svc.set_settings({"cap_gb": 1})
        svc._maybe_start_worker()
        self.assertTrue(wait_until(lambda: svc.store.get(item.id).state == "done"))
        self.assertEqual(segment.read_bytes(), b"committed segment" * 100)
        self.assertTrue(Path(item.source_path).exists())

    def test_killed_manifest_commit_keeps_previous_resume_boundary(self):
        seg = self.root / "segments"
        seg.mkdir()
        (seg / "seg_000000.mkv").write_bytes(b"committed")
        key = {"n": 10}
        manifest = Q.Manifest.load(str(self.root / "segments.json"), key, str(seg))
        manifest.add(0, 10)
        script = self.root / "interrupted.py"
        script.write_text(
            "import pathlib, oh_streamer.queue.queue_worker as Q, sys, time\n"
            "root = pathlib.Path(sys.argv[1])\n"
            "m = Q.Manifest.load(str(root/'segments.json'), {'n':10}, str(root/'segments'))\n"
            "(root/'segments'/'seg_000001.mkv').write_bytes(b'uncommitted')\n"
            "def interrupted_replace(src, dst):\n"
            "    print('commit prepared', flush=True)\n"
            "    time.sleep(120)\n"
            "Q.os.replace = interrupted_replace\n"
            "m.add(1, 10)\n", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(_STREAMER_ROOT / "src"))
        result = PC.run_command([sys.executable, script, self.root], timeout=1.5, env=env)
        self.assertTrue(result["timed_out"])
        self.assertIn("commit prepared", result["stdout"])
        resumed = Q.Manifest.load(str(self.root / "segments.json"), key, str(seg))
        Q.clean_work_dir(str(seg), resumed)
        self.assertEqual(resumed.next_k, 1)
        self.assertEqual((seg / "seg_000000.mkv").read_bytes(), b"committed")
        self.assertFalse((seg / "seg_000001.mkv").exists())


class LoopbackRecoveryTests(unittest.TestCase):
    setUp = RecoveryTests.setUp
    def test_listener_rejects_incomplete_and_oversized_frames_then_recovers(self):
        messages = []
        with patch.object(SI, "state_dir", return_value=self.root / "state"):
            inst = SI.Instance("framing-" + uuid.uuid4().hex, None)
            def close():
                with patch.object(SI, "state_dir", return_value=self.root / "state"):
                    inst.close()
            self.addCleanup(close)
            inst.serve(lambda msg: messages.append(msg) or {"ok": True})
            port = inst.sock.getsockname()[1]
            def exchange(payload):
                with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
                    conn.settimeout(2)
                    conn.sendall(payload)
                    conn.shutdown(socket.SHUT_WR)
                    try:
                        return conn.recv(1024)
                    except ConnectionResetError:
                        return b""
            good = json.dumps({"token": inst.token, "input": "影片.mp4"}, ensure_ascii=False).encode("utf-8")
            self.assertEqual(exchange(good), b"", "EOF must not dispatch a request lacking its newline")
            self.assertEqual(messages, [])
            with patch.object(SI, "MAX_MESSAGE", 512):
                oversized = json.dumps({"token": inst.token, "input": "x" * 1024}).encode() + b"\n"
                self.assertEqual(exchange(oversized), b"")
            self.assertEqual(messages, [])
            self.assertEqual(exchange(b'{invalid JSON}\n'), b"")
            # Split in the middle of a multibyte character to exercise byte framing.
            cut = good.index("影".encode("utf-8")) + 1
            with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
                conn.sendall(good[:cut])
                conn.sendall(good[cut:] + b"\n")
                self.assertEqual(json.loads(conn.recv(1024)), {"ok": True})
            self.assertEqual(messages, [{"input": "影片.mp4"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
