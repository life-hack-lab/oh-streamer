import tempfile
import unittest
from pathlib import Path

from oh_streamer.queue.subtitle_jobs import SubtitleQueueMixin
from oh_streamer.subtitles import model_download as md


def _item(id, cat, default=False, size=100):
    if cat == "runtime":
        spec = md.FileSpec("http://x/" + id, f"dl/{id}.zip", size, "0" * 64, extract_to="llama-cpp", unpacked=10)
        return md.CatalogItem(id, id, "r1", default, (spec,), marker="llama-cpp/llama-server.exe", category=cat, description="d")
    spec = md.FileSpec("http://x/" + id, f"{id}/f.bin", size, "0" * 64)
    return md.CatalogItem(id, id, "r1", default, (spec,), category=cat, description="d")


ITEMS = (_item("a1", "asr", True), _item("a2", "asr", True), _item("t1", "translation", True),
         _item("t2", "translation"), _item("rt", "runtime", True))


class FakeStore:
    def __init__(self, models):
        self.models = models

    def get_settings(self):
        return {"subtitle_models": str(self.models)}


class FakeJob:
    instances = []

    def __init__(self, models_dir, ids, items=None, **kw):
        self.ids, self.state, self.cancelled = list(ids), "running", False
        FakeJob.instances.append(self)

    def start(self):
        return self

    def cancel(self):
        self.cancelled = True
        self.state = "cancelled"

    def snapshot(self):
        return dict(state=self.state, item="", item_label="", bytes_done=0, bytes_total=1, speed=0, eta=None, error="")


class Svc(SubtitleQueueMixin):
    def __init__(self, models):
        self.store = FakeStore(models)
        self._model_dl_state = dict(lock=__import__("threading").Lock(), job=None, items=ITEMS, cls=FakeJob, seen=True)
        self.invalidated = 0

    def _invalidate_subtitle_environment(self):
        self.invalidated += 1


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc = Svc(Path(self.tmp.name))
        FakeJob.instances.clear()
        import os
        self._env = os.environ.pop("OH_STREAMER_SUBTITLE_MODELS", None)
        self.addCleanup(lambda: os.environ.__setitem__("OH_STREAMER_SUBTITLE_MODELS", self._env) if self._env else None)

    def test_in_use_refuses_runtime_when_local_run(self):
        self.svc._processes_in_folder = lambda folder: []
        self.svc._subtitle_local = object()
        plan = self.svc.plan_subtitle_model_download(["a1", "t1"])
        self.assertFalse(plan["ok"])
        self.assertIn("llama-server", plan["message"])
        res = self.svc.start_subtitle_model_download(["a1", "t1"])
        self.assertFalse(res["ok"])
        self.assertIn("本機字幕正在使用 llama-server", res["error"])
        self.assertEqual(FakeJob.instances, [])
        self.svc._subtitle_local = None
        self.assertTrue(self.svc.plan_subtitle_model_download(["a1", "t1"])["ok"])

    def test_in_use_refuses_when_process_in_folder(self):
        seen = []
        self.svc._processes_in_folder = lambda folder: seen.append(Path(folder).name) or ["x/llama-server.exe"]
        res = self.svc.start_subtitle_model_download(["a1", "t1"])
        self.assertFalse(res["ok"])
        self.assertEqual(seen, ["llama-cpp"])

    def test_in_use_ignores_installed_runtime_and_model_only(self):
        self.svc._subtitle_local = object()
        self.svc._processes_in_folder = lambda folder: ["x"]
        rt = Path(self.tmp.name) / "llama-cpp"
        rt.mkdir()
        (rt / "llama-server.exe").write_bytes(b"x")
        # runtime considered installed via a fake catalog entry without marker requirement
        items = tuple(i if i.id != "rt" else md.CatalogItem("rt", "rt", "r1", True, (), category="runtime") for i in ITEMS)
        self.svc._model_dl_state["items"] = items
        plan = self.svc.plan_subtitle_model_download(["a1", "t1"])
        self.assertTrue(plan["ok"])

    def test_process_scan_matches_folder_only(self):
        import os, psutil
        folder = Path(os.path.dirname(psutil.Process().exe()))
        self.assertTrue(Svc._processes_in_folder(folder))
        self.assertEqual(Svc._processes_in_folder(Path(self.tmp.name) / "nope"), [])

    def test_catalog_groups(self):
        res = self.svc.subtitle_model_downloads()
        self.assertEqual([r["id"] for r in res["groups"]["asr"]], ["a1", "a2"])
        self.assertEqual([r["id"] for r in res["groups"]["runtime"]], ["rt"])
        self.assertEqual(res["default_ids"], ["a1", "a2", "t1", "rt"])
        self.assertIsNone(res["progress"])

    def test_real_catalog_has_categories(self):
        cats = {i.id: i.category for i in md.catalog()}
        self.assertEqual(cats["llama-server"], "runtime")
        self.assertEqual(cats["anime-whisper"], "asr")
        self.assertEqual(cats["sakura-galtransl-14b-v3.8"], "translation")
        self.assertTrue(all(i.description for i in md.catalog()))

    def test_real_catalog_has_license_and_model_card(self):
        for item in md.catalog():
            self.assertTrue(item.license_id and item.license_label, item.id)
            self.assertTrue(item.card_url.startswith("https://"), item.id)
        by_id = {i.id: i for i in md.catalog()}
        for sakura in ("sakura-galtransl-7b-v3.7", "sakura-galtransl-14b-v3.8", "sakura-7b-qwen2.5-v1.0"):
            self.assertEqual(by_id[sakura].license_id, "CC-BY-NC-SA-4.0")
            self.assertIn("非商業", by_id[sakura].license_label)
        self.assertEqual(by_id["anime-whisper"].license_id, "MIT")
        self.assertEqual(by_id["whisper-ja-1.5B"].license_id, "unspecified")
        rows = {r["id"]: r for rows in self.svc.subtitle_model_downloads()["groups"].values() for r in rows}
        self.assertIn("license_label", next(iter(rows.values())))

    def test_open_model_card_only_catalog(self):
        from unittest import mock
        from oh_streamer.queue.queue_app import Api
        api = Api.__new__(Api)
        with mock.patch("webbrowser.open") as opened:
            self.assertFalse(api.open_model_card("../evil")["ok"])
            opened.assert_not_called()
            self.assertTrue(api.open_model_card("anime-whisper")["ok"])
            opened.assert_called_once_with("https://huggingface.co/litagin/anime-whisper")

    def test_plan_rules(self):
        self.assertFalse(self.svc.plan_subtitle_model_download(["t1"])["ok"])           # no ASR
        self.assertFalse(self.svc.plan_subtitle_model_download(["a1"])["ok"])           # no translation
        plan = self.svc.plan_subtitle_model_download(["a1", "t1"])
        self.assertTrue(plan["ok"])
        self.assertIn("rt", plan["ids"])                                                # runtime forced in
        self.assertEqual(plan["download_bytes"], 300)

    def test_installed_counts_for_rules_and_is_skipped(self):
        orig = md._item_installed
        md._item_installed = lambda m, i: i.id in ("a2", "rt")
        self.addCleanup(lambda: setattr(md, "_item_installed", orig))
        plan = self.svc.plan_subtitle_model_download(["t1"])
        self.assertTrue(plan["ok"])
        self.assertNotIn("rt", plan["ids"])

    def test_start_enforces_rules_and_single_job(self):
        res = self.svc.start_subtitle_model_download(["a1"])
        self.assertFalse(res["ok"])
        self.assertEqual(FakeJob.instances, [])
        res = self.svc.start_subtitle_model_download(["a1", "t1"])
        self.assertTrue(res["ok"])
        self.assertEqual(FakeJob.instances[0].ids, ["a1", "t1", "rt"])
        again = self.svc.start_subtitle_model_download(["a1", "t1"])
        self.assertFalse(again["ok"])
        self.assertEqual(len(FakeJob.instances), 1)

    def test_unknown_id_rejected(self):
        self.assertFalse(self.svc.start_subtitle_model_download(["nope"])["ok"])

    def test_cancel_then_restart_and_done_invalidates(self):
        self.svc.start_subtitle_model_download(["a1", "t1"])
        self.svc.cancel_subtitle_model_download()
        self.assertTrue(FakeJob.instances[0].cancelled)
        self.assertEqual(self.svc.subtitle_model_download_progress()["progress"]["state"], "cancelled")
        self.assertEqual(self.svc.invalidated, 0)
        self.assertTrue(self.svc.start_subtitle_model_download(["a1", "t1"])["ok"])
        FakeJob.instances[1].state = "done"
        self.svc.subtitle_model_download_progress()
        self.svc.subtitle_model_download_progress()
        self.assertEqual(self.svc.invalidated, 1)


if __name__ == "__main__":
    unittest.main()
