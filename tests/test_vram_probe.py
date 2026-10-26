"""T4.2/T4.3 fix 4: the pre-load free-VRAM probe (worker stage vram_probe, pipeline.run_vram_probe_child and the
LocalDeps.vram_probe of default_local_deps). Fake mem_get_info / fake worker job; no CUDA, no child process."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.queue import subtitle_local as sl
from oh_streamer.subtitles import pipeline, worker

MIB = 2**20


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class WorkerProbeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.result = Path(self._tmp.name) / "result.json"
        self.clock = Clock()

    def probe(self, frees_mib, need=5000.0, wait_s=10.0, poll_s=1.0):
        frees = list(frees_mib)
        calls = []

        def mem_get_info():
            calls.append(self.clock.t)
            value = frees.pop(0) if len(frees) > 1 else frees[0]
            return int(value * MIB), 8192 * MIB
        job = {"need_mib": need, "wait_s": wait_s, "poll_s": poll_s, "result": str(self.result)}
        out = worker.run_vram_probe(job, mem_get_info, self.clock.sleep, self.clock)
        self.assertEqual(json.loads(self.result.read_text("utf-8")), out)
        return out, calls

    def test_enough_free_vram_answers_at_once(self):
        out, calls = self.probe([6000.0])
        self.assertEqual((out["ok"], out["free_mib"], out["total_mib"], out["measured"]), (True, 6000.0, 8192.0, True))
        self.assertEqual(calls, [0.0])

    def test_short_vram_re_measures_until_the_clip_finishes(self):
        out, calls = self.probe([3000.0, 3500.0, 6000.0])
        self.assertTrue(out["ok"])
        self.assertEqual((out["free_mib"], out["waited_s"]), (6000.0, 2.0))
        self.assertEqual(calls, [0.0, 1.0, 2.0])

    def test_still_short_after_the_wait_is_not_ok(self):
        out, calls = self.probe([3000.0], wait_s=10.0)
        self.assertFalse(out["ok"])
        self.assertEqual(out["free_mib"], 3000.0)
        self.assertEqual(out["waited_s"], 10.0)
        self.assertEqual(len(calls), 11)                                    # every second for 10 s

    def test_zero_wait_measures_once(self):
        out, calls = self.probe([3000.0], wait_s=0.0)
        self.assertFalse(out["ok"])
        self.assertEqual(len(calls), 1)

    def test_a_failing_measurement_never_blocks(self):
        def boom():
            raise RuntimeError("no cuda")
        job = {"need_mib": 5000.0, "wait_s": 5.0, "result": str(self.result)}
        out = worker.run_vram_probe(job, boom, self.clock.sleep, self.clock)
        self.assertEqual((out["ok"], out["measured"]), (True, False))
        self.assertIn("no cuda", out["error"])
        self.assertTrue(self.result.is_file())

    def test_the_stage_is_known_to_main(self):
        self.assertIn('"vram_probe": "prepare_asr"', Path(worker.__file__).read_text("utf-8"))


class PipelineChildTests(unittest.TestCase):
    def test_child_job_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}

            def fake_run(job, job_path, log_path, on_phase=None, *, python=None, on_process=None, env_extra=None):
                seen.update(job=job, python=python, env_extra=env_extra)
                Path(job["result"]).write_text(json.dumps({"ok": False, "free_mib": 1000.0}), encoding="utf-8")
            with mock.patch.object(pipeline, "run_worker_job", fake_run):
                doc = pipeline.run_vram_probe_child(5313.0, workdir=Path(tmp) / "p", wait_s=10.0, python="py")
            self.assertEqual(doc, {"ok": False, "free_mib": 1000.0})
            self.assertEqual((seen["job"]["stage"], seen["job"]["need_mib"], seen["job"]["wait_s"]),
                             ("vram_probe", 5313.0, 10.0))
            self.assertIsNone(seen["env_extra"])                            # the probe must SEE the GPU

    def test_missing_result_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(pipeline, "run_worker_job", lambda *a, **k: None):
                with self.assertRaises(Exception):
                    pipeline.run_vram_probe_child(1.0, workdir=Path(tmp) / "p")


class DefaultDepsTests(unittest.TestCase):
    def test_default_local_deps_has_a_probe_that_uses_the_runtime_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_child(need, *, workdir, wait_s, python, on_phase, on_process):
                calls.append((need, wait_s, python, Path(workdir).parent.name))
                return {"ok": True}
            cfg = {"asr": {"mode": "dual"}, "translation": {"model_file": "sakura/x.gguf"}}
            paths = {"models": tmp, "python": "RUNTIME_PY"}
            deps = sl.default_local_deps(Path(tmp) / "sub", Path(tmp) / "a.mp4", cfg, runtime_paths=paths)
            self.assertIsNotNone(deps.vram_probe)
            with mock.patch.object(pipeline, "run_vram_probe_child", fake_child):
                self.assertEqual(deps.vram_probe(5000.0, 10.0), {"ok": True})
            self.assertEqual(calls, [(5000.0, 10.0, "RUNTIME_PY", "tmp")])


if __name__ == "__main__":
    unittest.main()
