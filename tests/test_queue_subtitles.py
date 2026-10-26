"""Pure queue subtitle helpers and read-only environment discovery tests."""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import io
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.queue import queue_store, subtitle_jobs
from oh_streamer.subtitles import cli, environment, glossary
from tests import test_queue_service as queue_test_support
import oh_streamer.queue.queue_service as queue_service


class SubtitleJobHelperTests(unittest.TestCase):
    def make_model(self, root):
        model = Path(root) / "anime-whisper"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"small fixture")
        return model

    def test_subtitle_paths_always_include_traditional_chinese_and_can_add_japanese(self):
        folder = Path("items/item-1/subtitles")
        self.assertEqual(subtitle_jobs.subtitle_paths(folder), [
            (str(folder / "sakura.zh-TW.srt"), "chi", "繁體中文（AI）"),
        ])
        self.assertEqual(subtitle_jobs.subtitle_paths(folder, japanese=True), [
            (str(folder / "sakura.zh-TW.srt"), "chi", "繁體中文（AI）"),
            (str(folder / "ja.srt"), "jpn", "日文（AI）"),
        ])

    def test_store_recovers_ready_subtitle_as_deferred_mux_after_restart(self):
        item = queue_store.Item.from_dict({
            "id": "subtitle-ready-on-disk",
            "state": "done",
            "subtitle": {"status": "ready", "percent": 100, "only": True},
        })

        self.assertEqual(item.state, "done")
        self.assertEqual(item.subtitle["status"], "mux_pending")
        self.assertIsNone(item.subtitle["percent"])
        self.assertTrue(item.subtitle["only"])


class SubtitleEnvironmentTests(unittest.TestCase):
    def make_environment(self, root):
        base = Path(root) / "testdata" / "subtitle_poc"
        python = base / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"fake python executable")

        models = base / "models"
        asr = models / "anime-whisper"
        asr.mkdir(parents=True)
        (asr / "config.json").write_text("{}", encoding="utf-8")
        (asr / "model.safetensors").write_bytes(b"asr weights")
        sakura = models / "sakura"
        sakura.mkdir(parents=True)
        (sakura / "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf").write_bytes(b"translation model")

        server = models / "llama-cpp" / "llama-server.exe"
        server.parent.mkdir(parents=True)
        server.write_bytes(b"fake server executable")
        return base, python, models, server

    def test_detect_reports_existing_development_runtime_and_choices(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-env-") as tmp:
            base, python, models, server = self.make_environment(tmp)
            with patch.object(environment, "_probe_python", return_value=True) as probe:
                found = environment.detect(root=tmp)

            self.assertTrue(found["available"], found["reason"])
            self.assertEqual(found["reason"], "")
            self.assertEqual(found["python"], str(python.resolve()))
            self.assertEqual(found["models"], str(models.resolve()))
            self.assertEqual(found["server"], str(server.resolve()))
            self.assertEqual(found["asr"], "whisper-ja-1.5B")
            self.assertEqual(found["asr_mode"], "dual")
            self.assertEqual(found["translation"], "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf")
            self.assertEqual([x["value"] for x in found["asr_options"]], ["anime-whisper"])
            self.assertEqual([x["value"] for x in found["translation_options"]], ["sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"])
            probe.assert_called_once()

    def test_detect_explains_missing_files_without_installing_or_downloading(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-env-missing-") as tmp:
            missing = str(Path(tmp) / "missing-python.exe")
            with patch.object(environment, "_probe_python") as probe, \
                    patch("oh_streamer.platform.platform_support.data_dir", return_value=Path(tmp) / "userdata"):
                found = environment.detect({"subtitle_python": missing}, root=tmp)

            self.assertFalse(found["available"])
            self.assertIn("找不到字幕執行環境的 Python", found["reason"])
            self.assertIn("找不到可用的 ASR 本地模型", found["reason"])
            self.assertIn("找不到選定的 Sakura GGUF", found["reason"])
            self.assertIn("找不到 llama-server", found["reason"])
            probe.assert_not_called()

    def test_detect_no_longer_blocks_frozen_runtime_when_files_exist(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-env-frozen-") as tmp:
            self.make_environment(tmp)
            with patch.object(environment, "_probe_python", return_value=True), \
                    patch.object(environment.sys, "frozen", True, create=True):
                found = environment.detect(root=tmp)

            self.assertNotIn("僅支援開發環境", found["reason"])
            self.assertTrue(found["available"], found["reason"])


class SubtitleQueueServiceTests(queue_test_support.ServiceTestBase):
    def setUp(self):
        class FakeInstance:
            def serve(self, _handler, log=None):
                self.log = log

            def close(self):
                pass

        acquire = patch.object(queue_test_support.QS.single_instance, "acquire", return_value=FakeInstance())
        acquire.start()
        try:
            super().setUp()
        finally:
            acquire.stop()
        # These tests exercise the legacy local pipeline; a new install defaults to cloud-first.
        self.svc.store.set_settings(subtitle_runtime_contract='legacy')

    def stop_scheduler(self):
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.svc._paused_all = False

    def enable_subtitles(self, item_id):
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "queued", "percent": 0})

    def test_playback_without_gpu_busy_allows_subtitle_start_and_does_not_cancel(self):
        item_id = self.ready()
        self.enable_subtitles(item_id)
        self.stop_scheduler()
        self.svc._playing = True
        self.svc._playing_ts = self.now[0]
        entered = threading.Event()
        release = threading.Event()

        def worker(_item, run):
            run.stage = "asr"
            entered.set()
            release.wait(5)

        try:
            with patch.object(self.svc, "_run_worker", side_effect=worker):
                self.svc._step()
                self.assertTrue(entered.wait(5))
                run = self.svc._wk
                self.svc._step()
                self.assertFalse(run.cancel.is_set())
                # The same worker is interrupted when the existing GPU-busy rule applies.
                self.svc._busy = True
                self.svc._busy_ts = self.now[0]
                self.svc._step()
                self.assertTrue(run.cancel.is_set())
        finally:
            release.set()
            if self.svc._wk:
                self.svc._wk.thread.join(5)
                self.svc._wk = None

    def test_subtitle_stage_admission_allows_playback_but_respects_busy_and_holding(self):
        item_id = self.ready()
        self.stop_scheduler()
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))
        self.svc._wk = run
        self.svc._playing = True
        self.svc._playing_ts = self.now[0]
        self.svc._subtitle_check(run)
        with patch.object(type(self.svc.perf), "holding", new_callable=property,
                          fget=lambda _: True):
            with self.assertRaises(queue_service._Cancelled):
                self.svc._subtitle_check(run)
        self.assertTrue(run.cancel.is_set())
        self.svc._wk = None

    def test_subtitle_only_retry_for_unwatched_item_can_queue_while_other_file_plays(self):
        item_id = self.done_item_with_source()
        self.svc._playing = True
        self.svc._playing_ts = self.now[0]
        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()):
            result = self.svc.rerun_subtitles(item_id)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "queued")

    def test_subtitle_only_tick_can_start_during_nonbusy_playback(self):
        item_id = self.done_item_with_source()
        self.svc.store.update(item_id, subtitle={"status": "mux_pending", "only": True})
        self.stop_scheduler()
        self.svc._playing = True
        self.svc._playing_ts = self.now[0]
        entered = threading.Event()
        with patch.object(self.svc, "_run_subtitle_only", side_effect=lambda *_: entered.set()):
            self.svc._step()
            self.assertTrue(entered.wait(5))
        self.assertEqual(self.svc._wk.item_id, item_id)
        self.svc._wk.thread.join(5)
        self.svc._wk = None

    def test_subtitle_only_start_is_mux_only_and_leaves_slice_work_to_the_executors(self):
        item_id = self.done_item_with_source("http://h/mux-only-start")
        self.stop_scheduler()
        started = []
        with patch.object(self.svc, "_run_subtitle_only", side_effect=lambda *_: started.append(1)):
            for status in ("queued", "waiting_gpu", "asr", "translate", "partial", "failed", "paused", "done"):
                self.svc.store.update(item_id, subtitle={"status": status, "only": True})
                self.svc._maybe_start_subtitle_only()
                self.assertIsNone(self.svc._wk, status)
            self.svc.store.update(item_id, subtitle={"status": "ready", "only": True})
            self.svc._maybe_start_subtitle_only()
            self.assertEqual(self.svc._wk.item_id, item_id)
            self.svc._wk.thread.join(5)
            self.svc._wk = None
        self.assertEqual(started, [1])

    def test_mux_embeds_only_the_primary_track_even_when_comparison_rows_exist(self):
        item_id = self.done_item_with_source("http://h/mux-primary")
        rows = [dict(id="primary", asr={}, translation={}, japanese_track=True), dict(id="cmp", asr={}, translation={})]
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True, "subtitle_configs": rows,
                                                  "subtitle_combos": ["primary", "cmp"]},
                              subtitle={"status": "mux_pending", "only": True})
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("sakura.zh-TW.srt", "ja.srt", "cmp.zh-TW.srt"):
            (folder / name).write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        jobs = []
        run = queue_service._Run(item_id, 0)
        with patch.object(self.svc, "_subtitle_check"), \
                patch.object(self.svc, "_subtitle_stage", side_effect=lambda it, r, job, env: jobs.append(job)), \
                patch.object(self.svc, "_subtitle_item_environment", return_value={}), \
                patch.object(subtitle_jobs, "replacement_guard", lambda path: __import__("contextlib").nullcontext()), \
                patch.object(subtitle_jobs, "replace_guarded_file", lambda src, dst: None), \
                patch.object(self.svc, "_safe_run_update"):
            self.svc._subtitle_mux(self.svc.store.get(item_id), run, self.svc.store.get(item_id).output_path)
        self.assertEqual([Path(t[0]).name for t in jobs[0]["tracks"]], ["sakura.zh-TW.srt", "ja.srt"])
        self.assertNotIn("cmp", " ".join(t[0] for t in jobs[0]["tracks"]))

    def test_mux_runs_inside_a_thumbnail_hold_and_works_without_a_thumb_manager(self):
        import contextlib
        item_id = self.done_item_with_source("http://h/mux-hold")
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "mux_pending", "only": True})
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sakura.zh-TW.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        events = []

        class Thumbs:
            @contextlib.contextmanager
            def hold(self_inner, iid):
                events.append(("hold", iid))
                try:
                    yield
                finally:
                    events.append(("release", iid))

        @contextlib.contextmanager
        def guard(path):
            events.append("guard")
            yield

        run = queue_service._Run(item_id, 0)
        real_thumbs = self.svc.thumbs
        for thumbs in (Thumbs(), None):
            events.clear()
            self.svc.thumbs = thumbs
            with patch.object(self.svc, "_subtitle_check"),                     patch.object(self.svc, "_subtitle_stage", side_effect=lambda *a: events.append("stage")),                     patch.object(self.svc, "_subtitle_item_environment", return_value={}),                     patch.object(subtitle_jobs, "replacement_guard", guard),                     patch.object(subtitle_jobs, "replace_guarded_file", lambda src, dst: None),                     patch.object(self.svc, "_safe_run_update"):
                self.svc._subtitle_mux(self.svc.store.get(item_id), run, self.svc.store.get(item_id).output_path)
            if thumbs is not None:
                self.assertEqual(events, [("hold", item_id), "stage", "guard", ("release", item_id)])
            else:
                self.assertEqual(events, ["stage", "guard"])
        self.svc.thumbs = real_thumbs                # tearDown stops it

    def test_progressive_payload_comes_from_the_published_slices_not_from_legacy_mode_flags(self):
        item_id = self.ready()
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        srt = folder / "block.zh-TW.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        self.svc._subtitle_update(item_id, status="partial", covered_s=300, audio_offset=1.5,
                                  block_srt=[str(srt)])
        payload = {"progressive": True}
        self.svc._add_external_subtitles(payload, self.svc.store.get(item_id))
        self.assertEqual(payload["subtitles"][0]["path"], str(srt))
        self.assertEqual((payload["subtitle_offset"], payload["subtitle_covered_s"], payload["subtitle_complete"]),
                         (1.5, 300, False))
        # nothing published yet: nothing to attach
        self.svc._subtitle_update(item_id, status="partial", covered_s=0, block_srt=None)
        empty = {"progressive": True}
        self.svc._add_external_subtitles(empty, self.svc.store.get(item_id))
        self.assertNotIn("subtitles", empty)

    def test_downloading_subtitle_item_starts_in_queue_order_without_waiting_gpu(self):
        # Every queued item is processed while it downloads: no hold, no 等待 GPU, queue order kept.
        downloading = self.ready()
        following = self.ready()
        self.enable_subtitles(downloading)
        self.svc.store.update(downloading, source_status="downloading")
        self.stop_scheduler()
        launched = []
        with patch.object(self.svc, "_source_ready", return_value=True),                 patch.object(self.svc, "_run_worker", side_effect=lambda item, _run: launched.append(item.id)):
            self.svc._maybe_start_worker()
            self.svc._wk.thread.join(5)
            self.assertEqual(launched, [downloading])
            self.assertNotEqual(self.svc.store.get(downloading).subtitle["status"], "waiting_gpu")
            self.assertEqual(self.svc.store.get(following).state, "queued")
            self.svc._wk = None
            self.svc.store.update(downloading, state="done")
            self.svc._maybe_start_worker()
            self.svc._wk.thread.join(5)
            self.assertEqual(launched, [downloading, following])
            self.svc._wk = None

    def available_environment(self):
        models = self.tmp / "models"
        model = models / "anime-whisper"
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "weights.safetensors").write_bytes(b"weights")
        return {
            "available": True, "reason": "", "python": _sys.executable,
            "models": str(models), "server": str(self.tmp / "llama-server.exe"),
            "asr": "anime-whisper", "translation": "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
        }

    def done_item_with_source(self, target="http://h/subtitle-source"):
        self.svc.pause_all()
        item_id = self.add(target)
        folder = self.svc.store.item_dir(item_id)
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "source.mp4"
        output = folder / "output.mkv"
        source.write_bytes(b"source video bytes")
        output.write_bytes(b"finished video bytes")
        self.svc.store.update(item_id, state="done", source_path=str(source), output_path=str(output),
                              size_bytes=output.stat().st_size)
        return item_id

    def test_missing_subtitle_environment_does_not_block_video_restoration(self):
        item_id = self.ready()
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "queued", "percent": 0, "error": ""})
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))

        with patch.object(self.svc, "_subtitle_environment", return_value={
            "available": False, "reason": "subtitle runtime is missing",
        }):
            self.svc._run_worker(item, run)

        finished = self.svc.store.get(item_id)
        self.assertEqual(finished.state, "done")
        self.assertTrue(Path(finished.output_path).is_file())
        # M7: the worker no longer runs subtitles; the executors report the missing runtime themselves.
        self.assertNotEqual(finished.subtitle["status"], "failed")

    def test_environment_probe_exception_does_not_block_service_initialization(self):
        root = self.tmp / "probe-error-service"
        with patch.object(environment, "detect", side_effect=OSError("runtime probe failed")):
            svc = queue_service.QueueService(root, player_name="subtitle-probe-error")
            found = svc._subtitle_environment()

        try:
            self.assertFalse(found["available"])
            self.assertIn("runtime probe failed", found["reason"])
            self.assertEqual(found["asr_options"], [])
        finally:
            svc.stop()

    def test_environment_probe_exception_does_not_block_video_restoration(self):
        item_id = self.ready()
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "queued", "percent": 0, "error": ""})
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))

        with patch.object(environment, "detect", side_effect=OSError("runtime probe failed")):
            self.svc._run_worker(item, run)

        finished = self.svc.store.get(item_id)
        self.assertEqual(finished.state, "done")
        self.assertTrue(Path(finished.output_path).is_file())
        self.assertNotEqual(finished.subtitle["status"], "failed")

    def test_finished_restoration_marks_subtitles_only_and_leaves_the_mux_to_the_scheduler(self):
        item_id = self.ready(state="processing")
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "ready", "percent": 100, "error": ""})
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))
        with patch.object(self.svc, "_subtitle_mux") as mux:
            self.svc._run_worker(item, run)
        mux.assert_not_called()
        finished = self.svc.store.get(item_id)
        self.assertEqual(finished.state, "done")
        self.assertTrue(finished.subtitle["only"])
        self.assertEqual(finished.subtitle["status"], "ready")

    def test_cancelled_mux_keeps_completed_video_and_scheduler_restarts_subtitle_only(self):
        item_id = self.ready(state="processing")
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "ready", "percent": 100, "error": ""})
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))

        # M7: the mux is no longer run by the restoration worker; an interrupted mux step is bookkept by
        # `_subtitle_interrupted`, and the scheduler restarts it.
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        with self.svc._lock:
            self.svc._paused_all = False
        self.svc.store.update(item_id, state="done", subtitle={"status": "mux", "percent": 100, "error": "", "only": True})
        run.reason = "auto"
        self.svc._subtitle_interrupted(self.svc.store.get(item_id), run)

        finished = self.svc.store.get(item_id)
        self.assertEqual(finished.state, "done")
        self.assertEqual(finished.subtitle["status"], "mux_pending")
        self.assertTrue(finished.subtitle["only"])
        self.assertIsNone(self.svc._wk)

        launched = threading.Event()
        subtitle_runs = []

        def run_only(_item, subtitle_run):
            subtitle_runs.append(subtitle_run)
            with self.svc._lock:
                if self.svc._wk is subtitle_run:
                    self.svc._wk = None
            launched.set()

        with patch.object(self.svc, "_run_subtitle_only", side_effect=run_only):
            self.svc._maybe_start_subtitle_only()
            self.assertTrue(launched.wait(5), "queued subtitle retry was not scheduled")

        self.assertEqual(len(subtitle_runs), 1)
        self.assertTrue(subtitle_runs[0].subtitle_only)
        self.assertEqual(self.svc.store.get(item_id).state, "done")

    def test_set_subtitles_validates_runtime_and_persists_item_toggle(self):
        item_id = self.ready()
        with patch.object(self.svc, "_subtitle_environment", return_value={
            "available": False, "reason": "subtitle runtime is missing",
        }):
            unavailable = self.svc.set_subtitles(item_id, True)
        self.assertFalse(unavailable["ok"])
        self.assertNotIn("subtitles", self.svc.store.get(item_id).settings)

        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()):
            enabled = self.svc.set_subtitles(item_id, True)
            self.assertTrue(enabled["ok"], enabled)
            item = self.svc.store.get(item_id)
            self.assertTrue(item.settings["subtitles"])
            self.assertEqual(item.subtitle["status"], "queued")

            disabled = self.svc.set_subtitles(item_id, False)
            self.assertTrue(disabled["ok"], disabled)
            item = self.svc.store.get(item_id)
            self.assertFalse(item.settings["subtitles"])
            self.assertEqual(item.subtitle["status"], "skipped")

        self.assertFalse(self.svc.set_subtitles(item_id, "yes")["ok"])

    def test_rerun_subtitles_queues_only_subtitle_work_for_a_done_item(self):
        item_id = self.done_item_with_source("http://h/rerun-only-subtitle")
        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()):
            result = self.svc.rerun_subtitles(item_id)

        self.assertTrue(result["ok"], result)
        item = self.svc.store.get(item_id)
        self.assertEqual(item.state, "done")
        self.assertTrue(item.settings["subtitles"])
        self.assertEqual(item.subtitle["status"], "queued")
        self.assertTrue(item.subtitle["only"])

    def test_subtitle_only_pause_resume_and_cancel(self):
        item_id = self.done_item_with_source("http://h/pause-only-subtitle")
        self.stop_scheduler()
        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()):
            self.assertTrue(self.svc.rerun_subtitles(item_id)["ok"])
        self.svc._subtitle_update(item_id, status="translate", percent=42, phase="translate")
        self.assertTrue(self.svc.pause(item_id)["ok"])
        item = self.svc.store.get(item_id)
        self.assertEqual((item.state, item.subtitle["status"], item.subtitle["percent"]), ("done", "paused", 42))
        self.assertTrue(item.subtitle["only"])
        # A stopped run must not turn "paused" into queued/cancelled.
        run = queue_service._Run(item_id, 0)
        run.reason = "pause"
        self.svc._subtitle_interrupted(item, run)
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "paused")
        self.assertNotIn("paused", subtitle_jobs.ACTIVE)

        self.assertTrue(self.svc.resume(item_id)["ok"])
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "queued")

        self.svc.pause(item_id)
        self.assertTrue(self.svc.cancel_subtitles(item_id)["ok"])
        item = self.svc.store.get(item_id)
        self.assertEqual(item.subtitle["status"], "cancelled")
        self.assertFalse(item.subtitle["only"])
        self.assertFalse(self.svc.cancel_subtitles(item_id)["ok"])

    def test_rerun_subtitles_clears_translation_partial(self):
        item_id = self.done_item_with_source("http://h/rerun-partial")
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / "sakura.translate.partial.json"
        partial.write_text("{}", encoding="utf-8")
        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()):
            self.assertTrue(self.svc.rerun_subtitles(item_id)["ok"])
        self.assertFalse(partial.exists())

    def test_queued_subtitle_only_redo_is_protected_from_capacity_eviction(self):
        item_id = self.done_item_with_source("http://h/subtitle-only-capacity")
        self.svc.store.update(item_id, watched=True, last_used=1.0)
        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()):
            result = self.svc.rerun_subtitles(item_id)

        self.assertTrue(result["ok"], result)
        self.assertIn(item_id, self.svc.store.protected_ids)
        # Refreshing player heartbeats must keep the in-flight only-redo reservation.
        self.svc._watched_ids()
        self.assertIn(item_id, self.svc.store.protected_ids)

        self.svc.store.set_settings(cap_gb=1e-8)  # 10 bytes, below the source + output fixture size.
        self.assertGreater(self.svc.store.usage_bytes(refresh=True), 10)
        self.assertFalse(self.svc.store.ensure_space(0))
        self.assertEqual(self.svc.store.get(item_id).state, "done")
        self.assertTrue(Path(self.svc.store.get(item_id).output_path).is_file())

    def test_cancelled_subtitle_stage_keeps_worker_lease_until_child_exits(self):
        item_id = self.done_item_with_source("http://h/cancel-subtitle")
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "mux_pending", "percent": 100, "error": "", "only": True})
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sakura.zh-TW.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))
        run.subtitle_only = True
        run.stage = "waiting_gpu"

        entered_wait = threading.Event()
        release_process = threading.Event()

        class BlockingProcess:
            def __init__(self):
                self.stdout = io.StringIO("")
                self.stdin = io.StringIO()
                self.pid = 12345
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                entered_wait.set()
                if not release_process.wait(timeout or 10):
                    raise TimeoutError("fake subtitle child was not released")
                self.returncode = 0
                return self.returncode

        child = BlockingProcess()
        # This test drives the supervisor manually. Stop the real scheduler so
        # pause_all from fixture setup cannot cancel it before fake Popen runs.
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        with self.svc._lock:
            self.svc._wk = run
        with patch.object(self.svc, "_subtitle_environment", return_value=self.available_environment()), \
                patch.object(queue_service.subprocess, "Popen", return_value=child) as spawn, \
                patch.object(queue_service, "kill_tree"):
            thread = threading.Thread(target=self.svc._run_subtitle_only, args=(item, run), daemon=True)
            thread.start()
            self.assertTrue(entered_wait.wait(5), "subtitle worker did not reach its blocking wait")

            self.svc._request_stop(run, "preempt")
            self.assertTrue(run.cancel.is_set())
            self.assertIs(self.svc._wk, run)
            self.assertIs(run.proc, child)

            # A queued subtitle-only run cannot acquire the one worker slot while the child is alive.
            self.svc._maybe_start_subtitle_only()
            self.assertIs(self.svc._wk, run)
            self.assertEqual(spawn.call_count, 1)

            release_process.set()
            thread.join(5)
            self.assertFalse(thread.is_alive())

        self.assertIsNone(self.svc._wk)
        self.assertIsNone(run.proc)
        # A mux the queue interrupted (preempt) is retried when the player is closed.
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "mux_pending")

    def test_perf_observation_with_delayed_window_overlapping_subtitles_is_ignored(self):
        self.svc._subtitle_windows[:] = [[990.0, 995.0]]
        self.svc.player_used_gpu_since = lambda _start: False
        self.now[0] = 1000.0
        recorded = []
        self.svc.perf.record_observation = lambda **values: recorded.append(values)

        self.svc._record_perf_obs({"frames": 20, "fps": 4, "mode": "full", "height": 1080})
        self.assertEqual(recorded, [])

        self.now[0] = 1001.0
        self.svc._record_perf_obs({"frames": 20, "fps": 4, "mode": "full", "height": 1080})
        self.assertEqual(len(recorded), 1)

    def test_playing_an_item_never_skips_pending_subtitles(self):
        """M7: the slice executors keep filling the table while it plays; play() no longer marks them skipped nor
        preempts the restoration worker."""
        item_id = self.ready()
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, settings={**item.settings, "subtitles": True},
                              subtitle={"status": "asr", "percent": 10, "error": ""})
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))
        run.stage = "restore"
        with self.svc._lock:
            self.svc._wk = run
        self.svc._stopping.set()
        try:
            self.assertTrue(self.svc.play(item_id)["ok"])
            self.assertNotEqual(run.reason, "preempt")
            self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "asr")
        finally:
            with self.svc._lock:
                self.svc._wk = None
            self.svc._stopping.clear()

    def test_reprocess_inherits_only_cache_and_slice_plan(self):
        """D6: the content-addressed stage cache and the slice table travel to the new item; the old ASR files and
        the combos folder do not, and no combos state is written."""
        item_id = self.done_item_with_source("http://h/reprocess-subtitle")
        old_folder = self.svc.store.item_dir(item_id) / "subtitles"
        (old_folder / "cache").mkdir(parents=True)
        (old_folder / "combos").mkdir()
        (old_folder / "cache" / "asr-1.json").write_text('{"segments": []}', encoding="utf-8")
        (old_folder / "slice-plan.json").write_text('{"v": 1}', encoding="utf-8")
        (old_folder / "ja.json").write_text("{}", encoding="utf-8")
        (old_folder / "asr_identity.json").write_text("{}", encoding="utf-8")
        (old_folder / "combos" / "manifest.json").write_text("{}", encoding="utf-8")

        result = self.svc.reprocess(item_id, {}, options={})
        self.assertTrue(result["ok"], result)
        new_folder = self.svc.store.item_dir(result["id"]) / "subtitles"
        self.assertEqual((new_folder / "cache" / "asr-1.json").read_text("utf-8"), '{"segments": []}')
        self.assertEqual((new_folder / "slice-plan.json").read_text("utf-8"), '{"v": 1}')
        for name in ("ja.json", "asr_identity.json", "combos"):
            self.assertFalse((new_folder / name).exists(), name)
        self.assertNotIn("combos", self.svc.store.get(result["id"]).subtitle)


    def test_ready_play_and_want_preserve_subtitles_and_restoration(self):
        item_id = self.ready()
        self.stop_scheduler()
        self.enable_subtitles(item_id)
        self.svc._subtitle_update(item_id, status="ready")
        run = queue_service._Run(item_id, 0)
        run.stage = "restore"
        self.svc._wk = run
        try:
            self.assertTrue(self.svc.play(item_id)["ok"])
            self.svc._ipc_want({"id": item_id, "k": 0})
            self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "ready")
            self.assertFalse(run.cancel.is_set())
        finally:
            self.svc._wk = None

    def test_wait_for_subtitles_persists_cancels_and_autoplays_once(self):
        item_id = self.ready()
        self.enable_subtitles(item_id)
        self.stop_scheduler()
        self.assertTrue(self.svc.play(item_id, 12, wait_for_subtitles=True)["waiting_subtitles"])
        restored = queue_store.Item.from_dict(self.svc.store.get(item_id).to_dict())
        self.assertTrue(restored.play_after_subtitles)
        self.assertEqual(restored.play_after_subtitles_start, 12)
        self.assertEqual(self.players, [])
        self.svc.cancel_play_after_subtitles(item_id)
        self.assertFalse(self.svc.store.get(item_id).play_after_subtitles)
        self.svc.play(item_id, 12, wait_for_subtitles=True)
        self.svc._subtitle_update(item_id, status="ready")
        self.svc._maybe_play_after_subtitles()
        self.svc._maybe_play_after_subtitles()
        self.assertEqual(len(self.players), 1)
        self.assertEqual(self.players[0]["start"], 12)
        self.assertFalse(self.svc.store.get(item_id).play_after_subtitles)
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "ready")

    def test_wait_failure_opens_without_subtitles_and_signals_osd(self):
        item_id = self.ready()
        self.enable_subtitles(item_id)
        self.stop_scheduler()
        self.svc.play(item_id, wait_for_subtitles=True)
        self.svc._subtitle_update(item_id, status="failed", error="fixture failure")
        self.svc._maybe_play_after_subtitles()
        self.assertTrue(self.players[-1]["subtitle_error"])
        self.assertNotIn("subtitles", self.players[-1])

    def test_restoration_finishing_before_first_slice_keeps_wait_for_subtitles(self):
        item_id = self.ready()
        self.stop_scheduler()
        self.enable_subtitles(item_id)
        self.svc.play(item_id, wait_for_subtitles=True)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))
        self.svc._run_worker(self.svc.store.get(item_id), run)

        item = self.svc.store.get(item_id)
        self.assertEqual(item.state, "done")
        self.assertEqual(item.subtitle["status"], "queued")
        self.assertTrue(item.play_after_subtitles)
        self.svc._maybe_play_after_subtitles()
        self.assertEqual(self.players, [])
        # the first finished slice (view) releases the wait
        self.svc._subtitle_update(item_id, status="partial", view={"v": 2, "state": "running", "slices_done": 1, "covered_s": 30.0})
        self.svc._maybe_play_after_subtitles()
        self.assertEqual(len(self.players), 1)

    def test_stuck_view_releases_the_wait_with_a_subtitle_error(self):
        item_id = self.ready()
        self.stop_scheduler()
        self.enable_subtitles(item_id)
        self.svc.play(item_id, wait_for_subtitles=True)
        self.assertTrue(self.svc.store.get(item_id).play_after_subtitles)
        self.svc._subtitle_update(item_id, view={"v": 2, "state": "stuck", "slices_done": 0, "covered_s": 0})
        self.svc._maybe_play_after_subtitles()
        self.assertTrue(self.players[-1]["subtitle_error"])

    def test_external_want_ends_wait_and_keeps_subtitles(self):
        item_id = self.ready()
        self.enable_subtitles(item_id)
        self.svc.play(item_id, wait_for_subtitles=True)
        self.svc._ipc_want({"id": item_id})
        item = self.svc.store.get(item_id)
        self.assertFalse(item.play_after_subtitles)
        self.assertEqual(item.subtitle["status"], "queued")        # never skipped

    def test_payload_ready_pending_and_embedded_track_rules(self):
        item_id = self.ready()
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir()
        (folder / "sakura.zh-TW.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n字幕\n", encoding="utf-8")
        for status in ("ready", "mux_pending", "done"):
            self.svc._subtitle_update(item_id, status=status, audio_offset=8.125)
            payload = self.svc._source_payload(self.svc.store.get(item_id))
            self.assertEqual(payload["subtitles"][0]["lang"], "chi")
            self.assertTrue(payload["subtitles"][0]["default"])
            self.assertEqual(payload["subtitle_offset"], 8.125)
        processed = {"processed": True}
        self.svc._add_external_subtitles(processed, self.svc.store.get(item_id))
        self.assertNotIn("subtitles", processed)
        for status in ("queued", "failed", "skipped"):
            self.svc._subtitle_update(item_id, status=status)
            self.assertNotIn("subtitles", self.svc._source_payload(self.svc.store.get(item_id)))

    def test_watched_mux_defers_without_starting_process(self):
        item_id = self.ready(state="done")
        self.enable_subtitles(item_id)
        self.svc._subtitle_update(item_id, status="ready")
        self.svc._watch[item_id] = self.now[0]
        run = queue_service._Run(item_id, 0)
        with patch.object(self.svc, "_subtitle_check"), patch.object(self.svc, "_subtitle_stage") as stage:
            self.assertFalse(self.svc._subtitle_mux(self.svc.store.get(item_id), run, "unused.mkv"))
        stage.assert_not_called()
        item = self.svc.store.get(item_id)
        self.assertEqual(item.state, "done")
        self.assertEqual(item.subtitle["status"], "mux_pending")
        self.assertEqual(queue_store.Item.from_dict(item.to_dict()).subtitle["status"], "mux_pending")

    def test_deferred_mux_runs_only_mux_for_matching_cache(self):
        item_id = self.ready(state="done")
        item = self.svc.store.get(item_id)
        self.svc.store.update(item_id, output_path=item.source_path,
                              subtitle={"status": "mux_pending", "only": True})
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, 0)
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        srt = folder / "sakura.zh-TW.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        with patch.object(self.svc, "_subtitle_mux", return_value=True) as mux:
            self.svc._run_subtitle_only(item, run)
        mux.assert_called_once_with(item, run, item.output_path)
        # The finished subtitle file is gone: nothing to mux, the item says so instead of silently waiting.
        srt.unlink()
        with patch.object(self.svc, "_subtitle_mux", return_value=True) as mux:
            self.svc._run_subtitle_only(item, run)
        mux.assert_not_called()
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "failed")

    def test_wait_priority_preempts_other_worker_without_watch_or_skip(self):
        first = self.ready()
        wanted = self.ready()
        self.enable_subtitles(wanted)
        self.stop_scheduler()
        self.svc.play(wanted, wait_for_subtitles=True)
        run = queue_service._Run(first, 0)
        self.svc._wk = run
        self.svc._watch[first] = self.now[0]
        try:
            self.svc._step()
            self.assertTrue(run.cancel.is_set())
            self.assertEqual(run.reason, "preempt")
            self.assertNotIn(wanted, self.svc._watch)
            self.assertEqual(self.svc.store.get(wanted).subtitle["status"], "queued")
        finally:
            self.svc._wk = None

    def test_play_during_mux_defers_instead_of_discarding_ready_subtitles(self):
        item_id = self.done_item_with_source("http://h/mux-play")
        self.stop_scheduler()
        self.enable_subtitles(item_id)
        self.svc._subtitle_update(item_id, status="mux", only=True)
        run = queue_service._Run(item_id, 0)
        run.stage = "mux"
        self.svc._wk = run
        try:
            self.assertTrue(self.svc.play(item_id)["ok"])
            current = self.svc.store.get(item_id)
            self.assertEqual(current.subtitle["status"], "mux_pending")
            self.assertEqual(current.state, "done")
            self.assertTrue(run.cancel.is_set())
            self.assertIn(item_id, self.svc._watched_ids())
        finally:
            self.svc._wk = None

    def test_pending_pause_and_resume_retains_mux_only_intent(self):
        item_id = self.done_item_with_source("http://h/pending-pause")
        self.stop_scheduler()
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(exist_ok=True)
        (folder / "sakura.zh-TW.srt").write_text("complete subtitles", encoding="utf-8")
        self.svc._subtitle_update(item_id, status="mux_pending", only=True)
        self.assertTrue(self.svc.pause(item_id)["ok"])
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "paused")
        self.assertTrue(self.svc.resume(item_id)["ok"])
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "mux_pending")
        self.assertEqual(self.svc.store.get(item_id).state, "done")

    def test_resume_without_the_final_srt_goes_back_to_the_executors(self):
        item_id = self.done_item_with_source("http://h/pending-pause-no-srt")
        self.stop_scheduler()
        self.svc._subtitle_update(item_id, status="mux_pending", only=True)
        self.assertTrue(self.svc.pause(item_id)["ok"])
        self.assertTrue(self.svc.resume(item_id)["ok"])
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "queued")

    def test_deferred_mux_admission_waits_for_lease_busy_and_perf(self):
        item_id = self.done_item_with_source("http://h/mux-admission")
        self.stop_scheduler()
        self.svc._subtitle_update(item_id, status="mux_pending", only=True)
        with patch.object(queue_service.threading, "Thread") as thread:
            self.svc._watch[item_id] = self.now[0]
            self.svc._maybe_start_subtitle_only()
            thread.assert_not_called()
            self.now[0] += self.svc.busy_expire + 1
            self.svc._busy, self.svc._busy_ts = True, self.now[0]
            self.svc._maybe_start_subtitle_only()
            thread.assert_not_called()
            self.svc._busy = False
            with patch.object(type(self.svc.perf), "holding", new_callable=property, fget=lambda _: True):
                self.svc._maybe_start_subtitle_only()
                thread.assert_not_called()
            self.svc._maybe_start_subtitle_only()
            thread.return_value.start.assert_called_once()
            self.svc._wk = None

    def test_processed_external_tracks_use_finished_audio_clock(self):
        item_id = self.done_item_with_source("http://h/output-clock")
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir()
        (folder / "sakura.zh-TW.srt").write_text("fixture", encoding="utf-8")
        self.svc._subtitle_update(item_id, status="mux_pending", audio_offset=8.4)
        payload = {"processed": True}
        with patch.object(self.svc, "_subtitle_audio_offset", return_value=.01):
            self.svc._add_external_subtitles(payload, self.svc.store.get(item_id))
        self.assertEqual(payload["subtitle_offset"], .01)

    def test_mux_pending_survives_source_cleanup_and_store_restart(self):
        """R11: no ready_identity any more; after a restart the mux is judged by status plus the final srt (and the
        slice table, when there is one)."""
        item_id = self.done_item_with_source("http://h/restart-mux")
        self.stop_scheduler()
        self.enable_subtitles(item_id)
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir()
        (folder / "sakura.zh-TW.srt").write_text("complete subtitles", encoding="utf-8")
        self.svc._subtitle_update(item_id, status="mux", only=True, audio_offset=8.4)
        original = self.svc.store.get(item_id)
        Path(original.source_path).unlink()
        self.svc.store.update(item_id, source_path="")
        restarted = queue_store.QueueStore(self.svc.store.item_dir(item_id).parent.parent)
        recovered = restarted.get(item_id)
        self.assertEqual(recovered.subtitle["status"], "mux_pending")
        self.assertTrue(recovered.subtitle["only"])
        self.assertEqual(recovered.subtitle["audio_offset"], 8.4)
        self.assertTrue(self.svc._subtitle_mux_ready(recovered))
        (folder / "sakura.zh-TW.srt").unlink()
        self.assertFalse(self.svc._subtitle_mux_ready(recovered))

class SubtitleGlossaryTests(unittest.TestCase):
    def test_generic_glossary_mode_ignores_per_item_glossary(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-generic-glossary-") as tmp:
            folder = Path(tmp)
            source = folder / "ja.json"
            source.write_text(json.dumps({"segments": [], "speech_intervals": []}), encoding="utf-8")
            model = folder / "sakura.gguf"
            model.write_bytes(b"fixture model metadata only")
            per_item_glossary = folder / "glossary.txt"
            per_item_glossary.write_text("forced per-item glossary phrase", encoding="utf-8")

            with patch.object(glossary, "load_glossary", wraps=glossary.load_glossary) as load:
                result = cli.translate_file(
                    source, folder, model="sakura", model_path=model, llama_server="unused",
                    sentence_mode=True, generic_glossary_only=True,
                )

        self.assertEqual(result["segments"], [])
        self.assertEqual(load.call_count, 1)
        self.assertNotIn(per_item_glossary, [Path(call.args[0]) for call in load.call_args_list])


if __name__ == "__main__":
    unittest.main()
