"""M7 fix B1: the local Sakura translation (OpenCC, llama-server, sentence options) runs in the subtitle runtime
child (worker stage mt_session), never in the queue process (lada venv: no opencc / silero_vad / torch).
No GPU, no models: the runtime child is a small stub script speaking the same stdin/stdout protocol."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import builtins
import json
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.queue import subtitle_local as sl
from oh_streamer.subtitles import environment, pipeline

BLOCKED = ("opencc", "silero_vad", "torch", "transformers", "faster_whisper", "llama_cpp", "pynvml")
ASR_ID = {"provider": "local", "asr_mode": "dual", "models": {"a": "x"}}
MT_ID = {"provider": "local", "backend": "sakura", "model": "m"}

STUB = textwrap.dedent('''
    import json, sys, time
    mode = sys.argv[1]
    job = json.load(open(sys.argv[sys.argv.index("--job") + 1], encoding="utf-8"))
    print(json.dumps({"type": "subtitle_progress", "phase": "prepare_translate", "percent": 10, "detail": "up"}),
          flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "close":
            break
        if mode == "hang":
            time.sleep(60)
        if mode == "oom":
            print(json.dumps({"type": "mt_error", "id": req["id"], "message": "CUDA out of memory. Tried"}), flush=True)
            continue
        ja = json.load(open(req["ja"], encoding="utf-8"))
        rows = [dict(s, text_zh="zh:" + s["text"], sentence=req.get("sentence")) for s in ja["segments"]]
        print(json.dumps({"type": "mt_rows", "id": req["id"], "rows": rows[:1]}), flush=True)
        json.dump({"segments": rows, "stage": job["stage"], "profile": job.get("profile")},
                  open(req["out"], "w", encoding="utf-8"))
        print(json.dumps({"type": "mt_done", "id": req["id"]}), flush=True)
''')


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stub = self.root / "stub_worker.py"
        self.stub.write_text(STUB, "utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def deps(self, mode="ok", **kwargs):
        self.enterContext(mock.patch.object(sl, "local_identities",
                                            return_value={"asr_identity": ASR_ID, "mt_identity": MT_ID}))
        self.enterContext(mock.patch.object(
            environment, "subtitle_command",
            lambda python, module, *args: [sys.executable, str(self.stub), mode, *args]))
        return sl.default_local_deps(
            self.root / "item", self.root / "src.mp4",
            {"translation": {"model_file": "m.gguf", "seed": 7}, "sentence": {"sentence_mode": True}},
            runtime_paths={"models": str(self.root / "models"), "server": "no-server.exe",
                           "python": "runtime-python"}, **kwargs)


JA = {"segments": [{"start": 1.0, "end": 2.0, "text": "a"}, {"start": 3.0, "end": 4.0, "text": "b"}]}


class HostImportAuditTests(Base):
    def test_translate_and_asr_batch_load_no_subtitle_packages_in_the_queue_process(self):
        attempts = []
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in BLOCKED:
                attempts.append(name)
                raise ImportError(f"{name} is not installed in the queue venv")
            return real_import(name, *args, **kwargs)

        deps = self.deps()
        sys_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in BLOCKED}
        try:
            with mock.patch.object(builtins, "__import__", guarded):
                batches = []
                with deps.mt_session() as translate:
                    rows = translate(JA, self.root / "w1", on_batch=batches.append)
                    translate(JA, self.root / "w2", sentence={"sentence_mode": False})
                self.assertEqual([r["text_zh"] for r in rows], ["zh:a", "zh:b"])
                self.assertEqual(len(batches), 1)
                # asr_batch: the ffmpeg cut is the only host work; the VAD / models run in the child
                def extract(_source, path, start, end, **_k):
                    path.write_bytes(b"RIFF")
                with mock.patch("oh_streamer.subtitles.runtime.resample_window_to_wav", side_effect=extract), \
                        mock.patch.object(pipeline, "run_production_asr_windows_child",
                                          return_value=[{"segments": [], "speech_intervals": []}]):
                    deps.asr_batch([(0.0, 10.0)], self.root / "asr")
        finally:
            sys.modules.update(sys_modules)
        self.assertEqual(attempts, [], f"queue process tried to import {attempts}")

    def test_translate_request_carries_sentence_override_and_job_options(self):
        deps = self.deps()
        with deps.mt_session() as translate:
            rows = translate(JA, self.root / "w", sentence={"sentence_mode": False})
        self.assertEqual(rows[0]["sentence"], {"sentence_mode": False})

    def test_oom_in_child_becomes_gpu_out_of_memory(self):
        deps = self.deps("oom")
        with self.assertRaises(sl.GpuOutOfMemory):
            with deps.mt_session() as translate:
                translate(JA, self.root / "w")


class CancelTests(Base):
    def test_killing_the_registered_process_ends_translate_within_1s(self):
        killed = {}

        def on_process(proc):
            def kill():
                time.sleep(0.8)
                killed["t"] = time.monotonic()
                proc.kill()
            threading.Thread(target=kill, daemon=True).start()
        deps = self.deps("hang", on_process=on_process)
        with self.assertRaises(RuntimeError):
            with deps.mt_session() as translate:
                translate(JA, self.root / "w")
        self.assertIn("t", killed, "on_process never received the runtime child")
        self.assertLess(time.monotonic() - killed["t"], 1.0)


if __name__ == "__main__":
    unittest.main()
