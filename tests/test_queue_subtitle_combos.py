"""Queue integration of subtitle combos (backend only): fake worker processes, no GPU, no models."""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import io
import json
import os
import shutil
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.queue import queue_app, queue_store, subtitle_jobs
from oh_streamer.subtitles import combos as subtitle_combos
from tests import test_queue_service as queue_test_support
import oh_streamer.queue.queue_service as queue_service

IDS = ["anime-sakura1", "ja15b-sakura1"]
SRT = "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"


def write_manifest(folder, statuses, stage_error="boom"):
    """A manifest as the scheduler writes it; zh files only for done combos."""
    presets, _ = subtitle_combos.load_presets()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for combo_id, status in statuses.items():
        zh = f"{combo_id}.zh-TW.srt"
        if status == "done":
            (folder / zh).write_text(SRT, encoding="utf-8")
        entries.append({
            "id": combo_id, "label": presets[combo_id]["label"], "source": "builtin", "status": status,
            "failed_stage": None if status == "done" else "translation",
            "error": None if status == "done" else stage_error,
            "files": {"zh": zh if status == "done" else None, "ja": None},
            "cue_count": 1 if status == "done" else None, "stages": {},
        })
    manifest = {"format": subtitle_combos.MANIFEST_FORMAT, "schema_version": 1, "updated_at": "t",
                "video": {"audio_sha256": "abc"}, "combos": entries, "mux": None}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def progress(statuses, stage_error="boom"):
    presets, _ = subtitle_combos.load_presets()
    return {"type": "subtitle_combos", "items": [
        {"id": i, "label": presets[i]["label"], "status": s,
         "failed_stage": "translation" if s == "failed" else None,
         "error": stage_error if s == "failed" else None, "cue_count": 1 if s == "done" else None, "stages": {}}
        for i, s in statuses.items()]}


class FakeProc:
    def __init__(self, events, rc=0, hook=None):
        self.events, self.rc, self.hook = events, rc, hook
        self.stdin = io.StringIO()
        self.pid = 4242
        self.returncode = None
        self.stdout = self._lines()

    def _lines(self):
        for index, event in enumerate(self.events):
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if self.hook:
                self.hook(index)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.rc
        return self.rc


class ComboQueueBase(queue_test_support.ServiceTestBase):
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
        # Keep the real scheduler out of the way; tests drive _run_subtitle_only by hand.
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.svc._paused_all = False
        user_file = self.tmp / "no-user-combos.json"
        env = patch.dict(os.environ, {"OH_STREAMER_SUBTITLE_COMBOS": str(user_file)})
        env.start()
        self.addCleanup(env.stop)
        self.user_file = user_file
        self.jobs = []
        self.script = {}
        for patcher in (
            patch.object(subtitle_jobs, "replacement_guard", lambda path: __import__("contextlib").nullcontext()),
            patch.object(subtitle_jobs, "replace_guarded_file", lambda src, dst: os.replace(src, dst)),
            patch.object(queue_service, "kill_tree"),
            # Fake model dirs hold no weights; ready_identity must not need real model files.
            patch.object(subtitle_jobs.runtime, "local_model_fingerprint", return_value="fake-model"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def environment(self, **extra):
        models = self.tmp / "models"
        models.mkdir(exist_ok=True)
        return {"available": False, "reason": "no sakura gguf", "runtime_available": True, "runtime_reason": "",
                "python": _sys.executable, "models": str(models), "server": str(self.tmp / "llama-server.exe"),
                "asr": "anime-whisper", "translation": "", **extra}

    def legacy_environment(self):
        return {"available": True, "reason": "", "python": _sys.executable, "models": str(self.tmp / "models"),
                "server": str(self.tmp / "llama-server.exe"), "asr": "anime-whisper",
                "translation": "sakura/x.gguf"}

    def done_item(self, target="http://h/combos-source", combos=None, subtitles=True):
        self.svc.pause_all()
        item_id = self.add(target)
        folder = self.svc.store.item_dir(item_id)
        folder.mkdir(parents=True, exist_ok=True)
        source, output = folder / "source.mp4", folder / "output.mkv"
        source.write_bytes(b"source video bytes")
        output.write_bytes(b"finished video bytes")
        item = self.svc.store.get(item_id)
        settings = {**item.settings, "subtitles": subtitles}
        if combos is not None:
            settings["subtitle_combos"] = combos
        self.svc.store.update(item_id, state="done", source_path=str(source), output_path=str(output),
                              size_bytes=output.stat().st_size, settings=settings,
                              subtitle={"status": "queued", "percent": 0, "error": "", "only": True})
        return item_id

    def combos_folder(self, item_id):
        return self.svc.store.item_dir(item_id) / "subtitles" / "combos"

    def fake_popen(self, combo_statuses=None, combo_rc=0, combo_hook=None, events=None):
        """Replacement for subprocess.Popen: acts like the worker for the stages the job asks for."""
        def popen(command, **_kw):
            job = json.loads(Path(command[command.index("--job") + 1]).read_text("utf-8"))
            self.jobs.append(job)
            stage = job["stage"]
            if stage == "combos":
                statuses = combo_statuses if combo_statuses is not None else dict.fromkeys(job["combos"], "done")
                write_manifest(job["folder"], statuses)
                lines = events if events is not None else [
                    {"type": "subtitle_progress", "phase": "combos", "percent": None, "detail": "start"},
                    progress(dict.fromkeys(statuses, "running")),
                    progress(statuses),
                    {"type": "done", "combos": dict(statuses)},
                ]
                return FakeProc(lines, combo_rc, combo_hook)
            if stage == "asr":
                folder = Path(job["folder"])
                (folder / "ja.json").write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "x"}]}), "utf-8")
                return FakeProc([{"type": "done"}])
            if stage == "translate":
                folder = Path(job["folder"])
                (folder / "sakura.zh-TW.json").write_text(json.dumps({"segments": [{"text": "你好"}]}), "utf-8")
                (folder / "sakura.zh-TW.srt").write_text(SRT, "utf-8")
                return FakeProc([{"type": "done"}])
            if stage == "mux":
                Path(job["output"]).write_bytes(b"muxed video bytes")
                return FakeProc([{"type": "done", "output": job["output"]}])
            raise AssertionError(stage)
        return popen

    def run_subtitle_only(self, item_id, popen, environment=None):
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, self.svc._generations.get(item_id, 0))
        run.subtitle_only = True
        run.stage = "waiting_gpu"
        with self.svc._lock:
            self.svc._wk = run
        env = environment or self.environment()
        with patch.object(self.svc, "_subtitle_environment", return_value=env), \
                patch.object(queue_service.subprocess, "Popen", side_effect=popen):
            self.svc._run_subtitle_only(item, run)
        return run

    def sub(self, item_id):
        return self.svc.store.get(item_id).subtitle


class SelectionAndValidationTests(ComboQueueBase):
    def test_list_subtitle_combos_shape_includes_builtin_and_user_presets(self):
        self.user_file.write_text(json.dumps({"schema_version": 1, "combos": [
            {"id": "my-turbo", "label": "turbo＋Sakura", "asr": {"model": "turbo"}}]}), "utf-8")
        listed = self.svc.list_subtitle_combos()
        by_id = {c["id"]: c for c in listed}
        self.assertEqual(set(by_id["anime-sakura1"]), {"id", "label", "builtin", "asr", "translation", "approx_gpu_hint",
                                                      "available", "missing", "unavailable_reason", "cloud"})
        self.assertTrue(by_id["anime-sakura1"]["builtin"])
        self.assertEqual(by_id["anime-sakura1"]["asr"], "anime-whisper")
        self.assertEqual(by_id["anime-sakura1"]["translation"], "sakura-v1")
        self.assertIn("1.9", by_id["anime-sakura1"]["approx_gpu_hint"])
        self.assertFalse(by_id["my-turbo"]["builtin"])
        self.assertEqual(listed[0]["id"], "ai-default")

    def test_set_subtitle_combos_validates_and_persists(self):
        item_id = self.done_item(subtitles=False)
        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            bad = self.svc.set_subtitle_combos(item_id, ["anime-sakura1", "nope"])
            self.assertFalse(bad["ok"])
            self.assertIn("nope", bad["error"])
            self.assertFalse(self.svc.set_subtitle_combos(item_id, "anime-sakura1")["ok"])
            self.assertFalse(self.svc.set_subtitle_combos(item_id, [1])["ok"])
            self.assertFalse(self.svc.set_subtitle_combos("missing", IDS)["ok"])
            self.assertNotIn("subtitle_combos", self.svc.store.get(item_id).settings)

            ok = self.svc.set_subtitle_combos(item_id, IDS + [IDS[0]])
        self.assertEqual(ok, {"ok": True, "ids": IDS})
        item = self.svc.store.get(item_id)
        self.assertEqual(item.settings["subtitle_combos"], IDS)
        # M7: no per-combo progress dict any more (the slice table tracks every comparison row)
        self.assertNotIn("combos", item.subtitle)
        plans = item.settings["subtitle_track_plans"]
        # this legacy picker lists the rows as given: the first one is the primary, the rest comparison rows
        self.assertIn("primary", plans)
        self.assertIn(IDS[1], plans)

        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            self.assertTrue(self.svc.set_subtitle_combos(item_id, [])["ok"])
        item = self.svc.store.get(item_id)
        self.assertNotIn("subtitle_combos", item.settings)
        self.assertNotIn("combos", item.subtitle)

    def test_user_preset_ids_are_accepted(self):
        self.user_file.write_text(json.dumps({"schema_version": 1, "combos": [
            {"id": "my-turbo", "label": "turbo＋Sakura", "asr": {"model": "turbo"}}]}), "utf-8")
        item_id = self.done_item()
        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            self.assertTrue(self.svc.set_subtitle_combos(item_id, ["my-turbo"])["ok"])

    def test_set_subtitle_combos_refused_while_subtitle_step_runs(self):
        item_id = self.done_item(combos=IDS)
        self.svc._subtitle_update(item_id, status="mux")
        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            result = self.svc.set_subtitle_combos(item_id, ["anime-sakura1"])
        self.assertFalse(result["ok"])
        self.assertEqual(self.svc.store.get(item_id).settings["subtitle_combos"], IDS)
        # also refused while the mux step holds the worker lease
        self.svc._subtitle_update(item_id, status="mux_pending")
        run = queue_service._Run(item_id, 0)
        run.subtitle_only = True
        run.stage = "mux"
        with self.svc._lock:
            self.svc._wk = run
        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            self.assertFalse(self.svc.set_subtitle_combos(item_id, ["anime-sakura1"])["ok"])
        with self.svc._lock:
            self.svc._wk = None
        # and while an executor thread of the item does not stop in time
        self.svc._subtitle_stop_executors = lambda *args, **kwargs: False
        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            self.assertFalse(self.svc.set_subtitle_combos(item_id, ["anime-sakura1"])["ok"])
        self.assertEqual(self.svc.store.get(item_id).settings["subtitle_combos"], IDS)

    def test_set_subtitle_combos_needs_the_subtitle_runtime(self):
        item_id = self.done_item(subtitles=False)
        broken = self.environment(runtime_available=False, runtime_reason="找不到 llama-server")
        with patch.object(self.svc, "_subtitle_environment", return_value=broken):
            result = self.svc.set_subtitle_combos(item_id, IDS)
        self.assertEqual((result["ok"], result["error"]), (False, "找不到 llama-server"))

    def test_default_combos_setting_validated_and_used_for_new_items(self):
        self.assertEqual(self.svc.store.get_settings()["subtitle_combos_default"], [])
        self.assertFalse(self.svc.set_default_subtitle_combos(["zzz"])["ok"])
        self.assertEqual(self.svc.store.get_settings()["subtitle_combos_default"], [])
        self.assertEqual(self.svc.set_default_subtitle_combos(IDS), {"ok": True, "ids": IDS})
        self.assertEqual(self.svc.store.get_settings()["subtitle_combos_default"], IDS)
        # set_settings goes through the same validation
        self.assertFalse(self.svc.set_settings({"subtitle_combos_default": ["zzz"]})["ok"])
        self.assertTrue(self.svc.set_settings({"subtitle_combos_default": ["anime-gt7b"]})["ok"])
        self.assertTrue(self.svc.set_default_subtitle_combos([])["ok"])
        self.assertEqual(self.svc.store.get_settings()["subtitle_combos_default"], [])

    def test_add_resolved_applies_default_and_explicit_selection(self):
        def resolved(token):
            return self.svc._tokens.put(types.SimpleNamespace(
                target="http://h/" + token, kind="url", headers={}, quals=None, title=token, cover="", heights=[]))

        from oh_streamer.queue.subtitle_jobs import derive_subtitle_snapshot
        self.svc.set_default_subtitle_combos(["anime-gt7b"])
        tok = resolved("t1")
        result = self.svc.add_resolved([tok], {"subtitles": True})
        self.assertTrue(result["ok"], result)
        item = self.svc.store.get(result["ids"][0])
        # nothing is frozen at add: the item follows the global settings (default comparison on)
        self.assertEqual({k for k in item.settings if k.startswith("subtitle_")}, set())
        self.assertEqual(derive_subtitle_snapshot(self.svc.store.get_settings(), item.settings)["subtitle_combos"],
                         ["primary", "anime-gt7b"])
        self.assertNotIn("combos", item.subtitle)            # M7: comparison state lives in the manifest, not here

        tok = resolved("t2")      # differs from the global value: kept as a per-video override
        result = self.svc.add_resolved([tok], {"subtitles": True, "subtitle_comparison_enabled": False})
        configured = self.svc.store.get(result["ids"][0]).settings
        self.assertEqual(configured["subtitle_overrides"], {"subtitle_comparison_enabled": False})
        derived = derive_subtitle_snapshot(self.svc.store.get_settings(), configured)
        self.assertEqual(derived["subtitle_combos"], ["primary"])
        self.assertEqual(len(derived["subtitle_comparison_rows"]), 1)

        tok = resolved("t3")      # same as the global value: no override stored
        result = self.svc.add_resolved([tok], {"subtitles": True, "subtitle_comparison_enabled": True})
        self.assertNotIn("subtitle_overrides", self.svc.store.get(result["ids"][0]).settings)

        tok = resolved("t4")
        result = self.svc.add_resolved([tok], {"subtitles": True, "subtitle_comparison_enabled": "invalid"})
        self.assertFalse(result["ok"])

        tok = resolved("t5")
        result = self.svc.add_resolved([tok], {})
        self.assertNotIn("subtitle_combos", self.svc.store.get(result["ids"][0]).settings)

    def test_subtitle_snapshot_keeps_intake_pixel_model_overrides(self):
        self.svc.store.set_settings(detection_model="global-detection", restoration_model="global-restoration")
        token = self.svc._tokens.put(types.SimpleNamespace(target="http://h/override", kind="url", headers={}, quals=None, title="override", cover="", heights=[]))
        with patch.object(self.svc, "_model_options", return_value={"detection_model": [], "restoration_model": []}):
            result = self.svc.add_resolved([token], {"subtitles": True, "detection_model": "chosen-detection", "restoration_model": "chosen-restoration"})
        self.assertTrue(result["ok"], result)
        settings = self.svc.store.get(result["ids"][0]).settings
        self.assertEqual(settings["detection_model"], "chosen-detection")
        self.assertEqual(settings["restoration_model"], "chosen-restoration")

    def test_bridge_exposes_combo_methods(self):
        api = queue_app.Api(types.SimpleNamespace(service=self.svc))
        self.assertEqual(api.list_subtitle_combos()[0]["id"], "ai-default")
        self.assertFalse(api.set_subtitle_combos("missing", IDS)["ok"])
        self.assertEqual(api.set_default_subtitle_combos(IDS), {"ok": True, "ids": IDS})


class PipelineTests(ComboQueueBase):
    def mux_pending_item(self, combos=IDS):
        """A finished video whose subtitles are done and waiting for the mux (what the executors leave behind)."""
        item_id = self.done_item(combos=combos)
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sakura.zh-TW.srt").write_text(SRT, "utf-8")
        self.svc._subtitle_update(item_id, status="mux_pending", only=True, percent=100)
        return item_id

    def test_mux_embeds_only_the_primary_track_and_keeps_comparison_tracks_as_sidecars(self):
        item_id = self.mux_pending_item()
        write_manifest(self.combos_folder(item_id), dict.fromkeys(IDS, "done"))
        self.run_subtitle_only(item_id, self.fake_popen())
        self.assertEqual([j["stage"] for j in self.jobs], ["mux"])
        mux = self.jobs[0]
        self.assertEqual([t[1] for t in mux["tracks"]], ["chi"])
        self.assertTrue(mux["tracks"][0][0].endswith("sakura.zh-TW.srt"))
        self.assertNotIn("dispositions", mux)
        presets, _ = subtitle_combos.load_presets()
        # tracks an older version embedded for these combos are replaced, never duplicated
        self.assertEqual(mux["replace_titles"], [presets[i]["label"] for i in IDS])
        sub = self.sub(item_id)
        self.assertEqual((sub["status"], sub["embedded"], sub["only"]), ("done", True, False))
        self.assertEqual(Path(self.svc.store.get(item_id).output_path).read_bytes(), b"muxed video bytes")
        self.assertTrue((self.combos_folder(item_id) / "manifest.json").is_file(), "sidecars stay")
        self.assertIsNone(self.svc._wk)

    def test_mux_without_comparison_rows_has_no_replace_titles(self):
        item_id = self.mux_pending_item(combos=[])
        self.run_subtitle_only(item_id, self.fake_popen())
        mux = self.jobs[0]
        self.assertEqual(len(mux["tracks"]), 1)
        self.assertNotIn("replace_titles", mux)
        self.assertEqual(self.sub(item_id)["status"], "done")

    def test_a_missing_subtitle_file_fails_the_item_without_spawning(self):
        item_id = self.mux_pending_item()
        (self.svc.store.item_dir(item_id) / "subtitles" / "sakura.zh-TW.srt").unlink()
        self.run_subtitle_only(item_id, self.fake_popen())
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.sub(item_id)["status"], "failed")
        self.assertEqual(self.svc.store.get(item_id).state, "done")

    def test_rerun_keeps_every_content_addressed_cache_and_resets_the_slice_table(self):
        item_id = self.done_item(combos=IDS)
        cache = self.combos_folder(item_id) / "stages" / "asr" / "k" / "ja.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}", "utf-8")
        slice_cache = self.svc.store.item_dir(item_id) / "subtitles" / "cache" / "asr" / "k.json"
        slice_cache.parent.mkdir(parents=True)
        slice_cache.write_text("{}", "utf-8")
        plan = self.svc.store.item_dir(item_id) / "subtitles" / "slice-plan.json"
        plan.write_text("{}", "utf-8")
        self.svc._subtitle_update(item_id, status="done", only=False, covered_s=60, block_srt=["x.srt"])
        with patch.object(self.svc, "_subtitle_environment", return_value=self.environment()):
            result = self.svc.rerun_subtitles(item_id)
        self.assertTrue(result["ok"], result)
        self.assertTrue(cache.is_file())
        self.assertTrue(slice_cache.is_file())
        self.assertFalse(plan.exists(), "a new slice table is built from the cache")
        self.assertNotIn("block_srt", self.sub(item_id))
        sub = self.sub(item_id)
        self.assertEqual((sub["status"], sub["only"]), ("queued", True))

    def test_legacy_rerun_still_clears_translation_files(self):
        item_id = self.done_item()
        legacy = self.svc.store.item_dir(item_id) / "subtitles" / "sakura.zh-TW.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text("{}", "utf-8")
        self.svc._subtitle_update(item_id, status="done", only=False)
        with patch.object(self.svc, "_subtitle_environment", return_value=self.legacy_environment()):
            self.assertTrue(self.svc.rerun_subtitles(item_id)["ok"])
        self.assertFalse(legacy.exists())

    def test_reprocess_does_not_carry_combos_folder_or_state(self):
        """M7 D6: only subtitles/cache and slice-plan.json are inherited; the combos folder is rebuilt by the
        comparison executor, and no item.subtitle["combos"] state is written."""
        item_id = self.done_item(combos=IDS)
        folder = self.combos_folder(item_id)
        write_manifest(folder, {IDS[0]: "done", IDS[1]: "failed"})

        result = self.svc.reprocess(item_id, {}, options={})
        self.assertTrue(result["ok"], result)
        new = self.svc.store.get(result["id"])
        self.assertFalse(self.combos_folder(new.id).exists())
        self.assertEqual(new.settings["subtitle_combos"], IDS)
        self.assertNotIn("combos", new.subtitle)
        # the original is untouched
        self.assertTrue((folder / "manifest.json").is_file())


class StoreSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="combo-store-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_old_store_without_combo_fields_loads_with_defaults(self):
        (self.tmp / "queue.json").write_text(json.dumps({
            "version": 1, "settings": {"subtitle_asr": "anime-whisper", "cap_gb": 5},
            "items": [{"id": "old1", "target": "x", "settings": {"subtitles": True},
                       "subtitle": {"status": "done", "percent": 100}}]}), "utf-8")
        store = queue_store.QueueStore(self.tmp)
        self.assertEqual(store.get_settings()["subtitle_combos_default"], [])
        item = store.get("old1")
        self.assertEqual(item.subtitle, {"status": "done", "percent": 100})
        self.assertNotIn("subtitle_combos", item.settings)

    def test_default_list_is_not_shared_between_stores(self):
        first = queue_store.QueueStore(self.tmp / "a")
        first.get_settings()["subtitle_combos_default"].append("x")
        second = queue_store.QueueStore(self.tmp / "b")
        self.assertEqual(second.get_settings()["subtitle_combos_default"], [])
        self.assertEqual(queue_store.DEFAULT_SETTINGS["subtitle_combos_default"], [])

    def test_old_combos_state_is_dropped_on_load_and_interrupted_step_requeued(self):
        state = {"selected": IDS, "total": 2, "done": 1, "failed": 0, "items": [
            {"id": IDS[0], "label": "a", "status": "done", "failed_stage": None, "error": None, "cue_count": 3},
            {"id": IDS[1], "label": "b", "status": "running", "failed_stage": None, "error": None, "cue_count": None}]}
        store = queue_store.QueueStore(self.tmp)
        store.set_settings(subtitle_runtime_contract='legacy')
        item = store.add("http://h/x", "url", None, None, {"subtitles": True, "subtitle_combos": IDS}, title="x")
        store.update(item.id, subtitle={"status": "combos", "percent": None, "phase": "combos",
                                        "detail": "d", "combos": state})
        store.set_settings(subtitle_combos_default=["anime-gt7b"])

        again = queue_store.QueueStore(self.tmp)
        loaded = again.get(item.id)
        self.assertEqual(loaded.settings["subtitle_combos"], IDS)
        self.assertEqual(again.get_settings()["subtitle_combos_default"], ["anime-gt7b"])
        self.assertEqual(loaded.subtitle["status"], "queued")  # interrupted 'combos' step is requeued
        self.assertNotIn("combos", loaded.subtitle)           # M7: progress is read from the manifest

    def test_corrupt_combo_fields_are_dropped_not_fatal(self):
        (self.tmp / "queue.json").write_text(json.dumps({"version": 1, "settings": {}, "items": [
            {"id": "bad1", "settings": {"subtitle_combos": "anime-sakura1"}, "subtitle": {"combos": "oops"}},
            {"id": "bad2", "subtitle": {"combos": {"items": [1, {"id": 5}, {"id": "ok", "status": "weird"}]}}}]}), "utf-8")
        store = queue_store.QueueStore(self.tmp)
        self.assertNotIn("combos", store.get("bad1").subtitle)
        self.assertNotIn("subtitle_combos", store.get("bad1").settings)
        self.assertNotIn("combos", store.get("bad2").subtitle)


class PlayerHandoffTests(ComboQueueBase):
    def test_lookup_and_play_payload_name_the_combos_folder_only_when_a_manifest_exists(self):
        target = "http://h/handoff"
        item_id = self.done_item(target=target, combos=IDS)
        found = self.svc._lookup(target, None)
        self.assertEqual(found["state"], "done")
        self.assertNotIn("subtitle_combos_dir", found)
        write_manifest(self.combos_folder(item_id), {IDS[0]: "done"})
        found = self.svc._lookup(target, None)
        self.assertEqual(Path(found["subtitle_combos_dir"]), self.combos_folder(item_id))
        payload = self.svc._add_combos_dir({"input": "x"}, self.svc.store.get(item_id))
        self.assertEqual(Path(payload["subtitle_combos_dir"]), self.combos_folder(item_id))


class InterruptTests(ComboQueueBase):
    def test_an_interrupted_mux_is_retried_later_and_slice_work_is_left_to_the_executors(self):
        item_id = self.done_item(combos=IDS)
        item = self.svc.store.get(item_id)
        run = queue_service._Run(item_id, 0)
        run.reason = "paused_all"
        self.svc._subtitle_update(item_id, status="mux", only=True)
        self.svc._subtitle_interrupted(item, run)
        self.assertEqual(self.sub(item_id)["status"], "mux_pending")
        # statuses the executors own are not touched by a stopped GPU lease
        for status in ("queued", "partial", "asr", "translate", "cancelled", "failed"):
            self.svc._subtitle_update(item_id, status=status)
            self.svc._subtitle_interrupted(item, run)
            self.assertEqual(self.sub(item_id)["status"], status)


if __name__ == "__main__":
    unittest.main()
