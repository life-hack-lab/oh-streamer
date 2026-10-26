"""M7 fix A: the local ASR of the slice executor runs every subtitle dependency (VAD prepare, models) in the
subtitle runtime child, never in the queue process; children start through environment.subtitle_command with
PYTHONPATH / PYTHONIOENCODING / offline env; a kill of the process on_process received ends asr_batch at once.
No GPU, no models: fake Popen, or a sleeping python child this test starts and kills."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.queue import subtitle_local as sl
from oh_streamer.subtitles import environment, pipeline

ASR_ID = {"provider": "local", "asr_mode": "dual", "models": {"a": "x"}}
MT_ID = {"provider": "local", "backend": "sakura", "model": "m"}
SRC = Path(pipeline.__file__).resolve().parents[2]          # streamer/src
WORKER = "oh_streamer.subtitles.worker"


def in_host(*_a, **_k):
    raise AssertionError("subtitle dependency loaded in the queue process")


class FakeProc:
    """Popen stand-in: records argv / env; as the worker would, writes the documents of an asr_windows job."""
    calls = []

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs = list(argv), kwargs
        FakeProc.calls.append(self)
        self.returncode = 0
        lines = []
        job = json.loads(Path(self.argv[self.argv.index("--job") + 1]).read_text("utf-8"))
        if job.get("stage") == "asr_windows":
            docs = [{"segments": [{"start": 1.0, "end": 2.0, "text": f"w{i}"}], "speech_intervals": [[1.0, 2.0]],
                     "model": "A1+A2", "asr_warnings": []} for i, _ in enumerate(job["windows"])]
            Path(job["result"]).write_text(json.dumps(docs), "utf-8")
            lines.append(json.dumps({"type": "subtitle_progress", "phase": "asr", "percent": 50, "detail": "x"}))
        self.stdout = iter(line + "\n" for line in lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self, *a, **k):
        return self.returncode

    def poll(self):
        return self.returncode


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        FakeProc.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def deps(self, **kwargs):
        def extract(_source, path, start, end, **_k):
            path.write_bytes(b"RIFF")
        self.enterContext(mock.patch.object(sl, "local_identities",
                                            return_value={"asr_identity": ASR_ID, "mt_identity": MT_ID}))
        self.enterContext(mock.patch("oh_streamer.subtitles.runtime.resample_window_to_wav", side_effect=extract))
        return sl.default_local_deps(self.root / "item", self.root / "src.mp4", {"translation": {"model_file": "m"}},
                                     runtime_paths={"models": str(self.root / "models"), "server": "unused",
                                                    "python": "runtime-python"}, **kwargs)

    def assert_child_env(self, kwargs):
        env = kwargs.get("env") or {}
        self.assertIn(str(SRC), env.get("PYTHONPATH", "").split(os.pathsep))
        self.assertEqual(env.get("PYTHONIOENCODING"), "utf-8")
        self.assertEqual(env.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(env.get("TRANSFORMERS_OFFLINE"), "1")
        self.assertNotIn("OH_STREAMER_SUBTITLE_IDLE_UI_PIDS", env)


class HostIsolationTests(Base):
    def test_asr_batch_runs_vad_and_models_only_in_runtime_child(self):
        phases, procs = [], []
        deps = self.deps(on_phase=lambda *a: phases.append(a), on_process=procs.append)
        with mock.patch("oh_streamer.subtitles.asr.prepare_shared_spans", side_effect=in_host), \
                mock.patch("oh_streamer.subtitles.asr._load_vad", side_effect=in_host), \
                mock.patch.dict(os.environ, {"OH_STREAMER_SUBTITLE_IDLE_UI_PIDS": "1"}), \
                mock.patch.object(pipeline.subprocess, "Popen", side_effect=FakeProc):
            out = deps.asr_batch([(0.0, 10.0), (20.0, 30.0)], self.root / "work")
        self.assertEqual(len(FakeProc.calls), 1)                       # one runtime child for the whole turn
        proc = FakeProc.calls[0]
        job_path = proc.argv[proc.argv.index("--job") + 1]
        self.assertEqual(proc.argv, environment.subtitle_command("runtime-python", WORKER, "--job", job_path))
        self.assert_child_env(proc.kwargs)
        self.assertEqual(procs, [proc])
        self.assertTrue(any(p[0] == "asr" for p in phases))
        self.assertEqual([r["segments"][0]["text"] for r in out], ["w0", "w1"])
        self.assertEqual(out[1]["segments"][0]["start"], 21.0)         # window-local doc mapped to source time

    def test_child_error_and_oom_come_back(self):
        deps = self.deps()

        def failing(argv, **kwargs):
            proc = FakeProc(argv, **kwargs)
            proc.returncode = 1
            proc.stdout = iter([json.dumps({"type": "error", "message": "CUDA out of memory. Tried to allocate"})])
            return proc
        with mock.patch.object(pipeline.subprocess, "Popen", side_effect=failing):
            with self.assertRaises(sl.GpuOutOfMemory):
                deps.asr_batch([(0.0, 10.0)], self.root / "work")

        def oom_doc(argv, **kwargs):
            proc = FakeProc(argv, **kwargs)
            job = json.loads(Path(argv[argv.index("--job") + 1]).read_text("utf-8"))
            Path(job["result"]).write_text(json.dumps([{"error": "x", "segments": [],
                                                        "asr_warnings": ["A1: CUDA out of memory"]}]), "utf-8")
            return proc
        with mock.patch.object(pipeline.subprocess, "Popen", side_effect=oom_doc):
            with self.assertRaises(sl.GpuOutOfMemory):
                deps.asr_batch([(0.0, 10.0)], self.root / "work2")

        def plain_error(argv, **kwargs):
            proc = FakeProc(argv, **kwargs)
            proc.returncode = 1
            proc.stdout = iter([json.dumps({"type": "error", "message": "Silero VAD is unavailable"})])
            return proc
        with mock.patch.object(pipeline.subprocess, "Popen", side_effect=plain_error):
            with self.assertRaisesRegex(RuntimeError, "Silero VAD is unavailable"):
                deps.asr_batch([(0.0, 10.0)], self.root / "work3")

    def test_run_asr_child_env_and_command(self):
        with mock.patch.object(pipeline.subprocess, "Popen", side_effect=FakeProc):
            pipeline.run_asr_child({"folder": str(self.root / "asr"), "stage": "asr_model_windows"},
                                   python="runtime-python")
        proc = FakeProc.calls[0]
        job_path = proc.argv[proc.argv.index("--job") + 1]
        self.assertEqual(proc.argv, environment.subtitle_command("runtime-python", WORKER, "--job", job_path))
        self.assert_child_env(proc.kwargs)

    def test_frozen_child_reenters_executable(self):
        with mock.patch.object(environment.sys, "frozen", True, create=True), \
                mock.patch.object(pipeline.subprocess, "Popen", side_effect=FakeProc):
            pipeline.run_asr_child({"folder": str(self.root / "asr"), "stage": "asr_model_windows"},
                                   python="runtime-python")
        self.assertEqual(FakeProc.calls[0].argv[:2], [sys.executable, "--subtitle-worker"])


class CancelTests(Base):
    def test_killing_the_registered_process_ends_asr_batch_within_1s(self):
        killed = {}

        def on_process(proc):
            def kill():
                time.sleep(0.3)
                killed["t"] = time.monotonic()
                proc.kill()
            threading.Thread(target=kill, daemon=True).start()
        deps = self.deps(on_process=on_process)
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        with mock.patch.object(environment, "subtitle_command", return_value=sleeper):
            started = time.monotonic()
            with self.assertRaises(RuntimeError):
                deps.asr_batch([(0.0, 10.0)], self.root / "work")
        self.assertIn("t", killed, "on_process never received the child")
        self.assertLess(time.monotonic() - killed["t"], 1.0)
        self.assertLess(time.monotonic() - started, 20.0)

    def test_model_grandchild_exits_when_its_worker_is_killed(self):
        """worker.watch_parent: even a plain kill of the direct child (no tree kill) ends the model process."""
        import subprocess
        grandchild = ("import os, sys, time; from oh_streamer.subtitles.worker import watch_parent; "
                      "watch_parent(int(sys.argv[1])); time.sleep(60)")
        parent = ("import os, subprocess, sys, time; "
                  f"c = subprocess.Popen([sys.executable, '-c', {grandchild!r}, str(os.getpid())]); "
                  "print(c.pid, flush=True); time.sleep(60)")
        env = dict(os.environ, PYTHONPATH=str(SRC))
        proc = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE, text=True, env=env)
        try:
            child_pid = int(proc.stdout.readline())
            time.sleep(1.0)                                            # the grandchild armed its watcher
            proc.kill()
            proc.wait(5)
            import psutil
            child = psutil.Process(child_pid)
            child.wait(5)                                              # raises TimeoutExpired if it survived
        except psutil.NoSuchProcess:
            pass
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
