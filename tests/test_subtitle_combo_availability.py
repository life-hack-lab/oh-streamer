import tempfile
import unittest
from pathlib import Path

from oh_streamer.subtitles import combos
from oh_streamer.queue import subtitle_jobs


def asr(models, name):
    d = Path(models) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")


def gguf(models, rel):
    p = Path(models) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


class ComboAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.models.mkdir()
        self.server = Path(self.tmp.name) / "llama-server.exe"
        self.presets, _ = combos.load_presets(Path(self.tmp.name) / "none.json")

    def tearDown(self):
        self.tmp.cleanup()

    def install(self, *, ja=False, anime=False, gt7=False, server=False):
        if ja: asr(self.models, "whisper-ja-1.5B")
        if anime: asr(self.models, "anime-whisper")
        if gt7: gguf(self.models, combos.TRANSLATION_FILES["sakura-galtransl-7b-v3.7"])
        if server: self.server.write_bytes(b"x")

    def avail(self, combo_id):
        return combos.combo_availability(self.presets[combo_id], self.models, self.server)

    def test_requirements_dual_single_and_server(self):
        ids = [r["id"] for r in combos.combo_requirements(self.presets["ai-default"])]
        self.assertEqual(ids, ["whisper-ja-1.5B", "anime-whisper", "sakura-galtransl-7b-v3.7", "llama-server"])
        ids = [r["id"] for r in combos.combo_requirements(self.presets["anime-sakura1"])]
        self.assertEqual(ids, ["anime-whisper", "sakura-v1", "llama-server"])

    def test_correction_llm_adds_candidates_and_gguf(self):
        rows = combos.combo_requirements(self.presets["anime-llmfix-sakura1"])
        paths = [r["path"] for r in rows]
        self.assertIn("whisper-ja-1.5B", paths)
        self.assertIn("anime-whisper", paths)
        self.assertIn(combos.CORRECTION_FILES["qwen3"], paths)

    def test_cloud_parts_need_nothing(self):
        cloud = combos.normalize_combo({"id": "c1", "label": "c", "asr": {"mode": "single", "model": "whisper-large-v3"},
                                        "translation": {"model_id": "user-groq-x", "batch_size": 30}})
        self.assertEqual(combos.combo_requirements(cloud), [])
        self.assertTrue(combos.combo_availability(cloud, self.models, "")["available"])
        mixed = combos.normalize_combo({"id": "c2", "label": "m", "asr": {"mode": "single", "model": "whisper-large-v3"}})
        self.assertEqual([r["id"] for r in combos.combo_requirements(mixed)], ["sakura-galtransl-7b-v3.7", "llama-server"])

    def test_missing_labels_and_hint(self):
        self.install(ja=True)
        info = self.avail("ai-default")
        self.assertFalse(info["available"])
        self.assertEqual(info["missing"], ["anime-whisper", "GalTransl 7B v3.7", "llama-server"])
        self.assertIn("設定 → 字幕 → 本機 下載本機字幕模型", info["reason"])
        self.install(anime=True, gt7=True, server=True)
        self.assertTrue(self.avail("ai-default")["available"])

    def test_fallback_dual_to_single_and_none(self):
        self.install(ja=True, gt7=True, server=True)
        res = combos.resolve_available_combo(self.presets["ai-default"], self.models, self.server, presets=self.presets)
        self.assertEqual(res["fallback"]["to_id"], "ai-single")
        self.assertEqual(res["combo"]["asr"]["mode"], "single")
        self.assertIn("暫時改用", res["fallback"]["message"])
        self.assertEqual(self.presets["ai-default"]["asr"]["mode"], "dual")   # input untouched
        # only anime-whisper + gt7b -> anime-gt7b
        (self.models / "whisper-ja-1.5B" / "model.safetensors").unlink()
        asr(self.models, "anime-whisper")
        res = combos.resolve_available_combo(self.presets["ai-default"], self.models, self.server, presets=self.presets)
        self.assertEqual(res["fallback"]["to_id"], "anime-gt7b")
        # nothing available: unchanged
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        res = combos.resolve_available_combo(self.presets["ai-default"], empty, self.server, presets=self.presets)
        self.assertIsNone(res["fallback"])
        self.assertIs(res["combo"], self.presets["ai-default"])

    def test_snapshot_falls_back_without_rewriting_setting(self):
        self.install(anime=True, gt7=True, server=True)
        settings = {"subtitles": True, "subtitle_models": str(self.models), "subtitle_server": str(self.server)}
        snap = subtitle_jobs.snapshot_subtitle_settings(settings)
        self.assertEqual(snap["subtitle_configs"][0]["asr"]["model"], "anime-whisper")
        self.assertEqual(snap["subtitle_configs"][0]["asr"]["mode"], "single")
        self.assertEqual(snap["subtitle_combo_fallback"]["to_id"], "anime-gt7b")
        self.assertNotIn("subtitle_primary", settings)

    def test_list_combos_marks_unavailable(self):
        self.install(anime=True, gt7=True, server=True)
        settings = {"subtitle_models": str(self.models), "subtitle_server": str(self.server)}
        rows = {r["id"]: r for r in subtitle_jobs.list_combos(settings)}
        self.assertTrue(rows["anime-gt7b"]["available"])
        self.assertFalse(rows["ai-default"]["available"])
        self.assertEqual(rows["ai-default"]["missing"], ["whisper-ja-1.5B"])
        self.assertIn("下載", rows["ai-default"]["unavailable_reason"])

    def test_start_check_fails_row_with_missing_message(self):
        from types import SimpleNamespace
        from oh_streamer.queue import subtitle_host_gpu as gh
        cls = next(c for c in vars(gh).values() if isinstance(c, type) and hasattr(c, "_subtitle_require_local_models"))
        # the models folder / llama-server come from the global settings (never stored per item)
        fake = SimpleNamespace(store=SimpleNamespace(get_settings=lambda: {"subtitle_models": str(self.models),
                                                                           "subtitle_server": str(self.server)}))
        item = SimpleNamespace(settings={"subtitle_configs": [self.presets["anime-gt7b"]]})
        with self.assertRaises(RuntimeError) as ctx:
            cls._subtitle_require_local_models(fake, item)
        self.assertIn("缺少 anime-whisper", str(ctx.exception))
        self.install(anime=True, gt7=True, server=True)
        cls._subtitle_require_local_models(fake, item)

    def _start_check_fixture(self, rows):
        from types import SimpleNamespace
        from oh_streamer.queue import subtitle_host_gpu as gh
        cls = next(c for c in vars(gh).values() if isinstance(c, type) and hasattr(c, "_subtitle_require_local_models"))
        updates = []
        store = SimpleNamespace(get_settings=lambda: {"subtitle_models": str(self.models),
                                                      "subtitle_server": str(self.server)}, update=lambda item_id, **kw: updates.append((item_id, kw)))
        fake = SimpleNamespace(store=store)
        item = SimpleNamespace(id="it1", settings={"subtitle_configs": rows})
        return cls, fake, item, updates

    def test_missing_comparison_row_marks_only_that_track(self):
        self.install(anime=True, gt7=True, server=True)       # ai-default (dual: needs whisper-ja) is missing
        primary = self.presets["anime-gt7b"]
        other = dict(self.presets["ai-default"], id="cmp1")
        cls, fake, item, updates = self._start_check_fixture([primary, other])
        cls._subtitle_require_local_models(fake, item)         # primary fine: no raise
        self.assertEqual(len(updates), 1)
        plans = updates[0][1]["settings"]["subtitle_track_plans"]
        self.assertFalse(plans["cmp1"]["ready"])
        self.assertIn("缺少 whisper-ja-1.5B", plans["cmp1"]["reasons"][0])
        self.assertIn("下載", plans["cmp1"]["reasons"][0])
        self.assertIsNone(plans["primary"])      # legacy item: no frozen primary plan, untouched

    def test_comparison_row_recovers_after_models_downloaded(self):
        self.install(anime=True, gt7=True, server=True)
        primary = self.presets["anime-gt7b"]
        other = dict(self.presets["ai-default"], id="cmp1")
        cls, fake, item, updates = self._start_check_fixture([primary, other])
        cls._subtitle_require_local_models(fake, item)
        item.settings = updates[-1][1]["settings"]
        self.assertFalse(item.settings["subtitle_track_plans"]["cmp1"]["ready"])
        cls._subtitle_require_local_models(fake, item)          # still missing: marked once, not appended again
        self.assertEqual(len(updates), 1)
        self.install(ja=True)
        cls._subtitle_require_local_models(fake, item)
        plan = updates[-1][1]["settings"]["subtitle_track_plans"]["cmp1"]
        self.assertFalse(any(e.get("code") == "models_missing" for e in plan.get("errors") or []))

    def test_available_comparison_rows_untouched(self):
        self.install(anime=True, gt7=True, server=True)
        cls, fake, item, updates = self._start_check_fixture(
            [self.presets["anime-gt7b"], dict(self.presets["anime-gt7b"], id="cmp1")])
        cls._subtitle_require_local_models(fake, item)
        self.assertEqual(updates, [])

    def test_missing_primary_still_raises_even_with_comparison(self):
        cls, fake, item, updates = self._start_check_fixture(
            [self.presets["anime-gt7b"], dict(self.presets["anime-gt7b"], id="cmp1")])
        with self.assertRaises(RuntimeError):
            cls._subtitle_require_local_models(fake, item)
        self.assertEqual(updates, [])


if __name__ == "__main__":
    unittest.main()
