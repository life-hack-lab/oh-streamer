"""Cloud subtitles must not depend on the dev subtitle venv (frozen build / bare checkout)."""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oh_streamer.queue.queue_store import QueueStore  # noqa: E402
from oh_streamer.queue.subtitle_jobs import SubtitleQueueMixin  # noqa: E402
from oh_streamer.subtitles import environment  # noqa: E402


def load_frozen_entry():
    spec = importlib.util.spec_from_file_location("streamer_packaging_frozen_entry_subtitles",
                                                  ROOT / "packaging" / "frozen_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLOUD_ROW = {"id": "primary", "asr": {"model": "user-groq"}, "translation": {"model_id": "user-azure"}}


class FrozenCloudRuntimeTests(unittest.TestCase):
    def setUp(self):
        environment._probe_runtime_python.cache_clear()
        environment._probe_python.cache_clear()

    def test_frozen_without_venv_keeps_cloud_runtime_but_blocks_local(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(environment.sys, "frozen", True, create=True), \
                patch.object(environment, "_probe_python", return_value=False), \
                patch("oh_streamer.platform.platform_support.data_dir", return_value=Path(tmp) / "userdata"):
            found = environment.detect({}, root=tmp)
        self.assertTrue(found["runtime_available"], found["runtime_reason"])
        self.assertEqual(found["runtime_python"], sys.executable)
        self.assertFalse(found["available"])
        self.assertNotIn("僅支援開發環境", found["reason"])
        self.assertIn("找不到可用的 ASR 本地模型", found["reason"])

    def test_dev_checkout_without_venv_runs_cloud_on_current_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(environment, "_probe_python", return_value=False):
            found = environment.detect({}, root=tmp)
        self.assertTrue(found["runtime_available"], found["runtime_reason"])
        self.assertEqual(found["runtime_python"], str(Path(sys.executable).resolve()))
        self.assertFalse(found["available"])
        self.assertIn("字幕執行環境缺少必要套件", found["reason"])

    def test_working_venv_stays_the_runtime_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "venv-python.exe"
            venv.write_bytes(b"fake")
            with patch.object(environment, "_probe_runtime_python", return_value=True):
                python, ok = environment.resolve_runtime_python(str(venv))
        self.assertTrue(ok)
        self.assertEqual(python, str(venv.resolve()))

    def test_item_snapshot_with_missing_venv_is_cloud_ready_when_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp) / "queue")
            store.set_settings(subtitle_runtime_contract="legacy")  # snapshot rows as given
            missing = str(Path(tmp) / "testdata/subtitle_poc/.venv/Scripts/python.exe")
            item = store.add("source", "file", None, None, {
                "subtitles": True, "subtitle_configs": [CLOUD_ROW], "subtitle_python": missing})
            svc = SimpleNamespace(_subtitle_environment=lambda: dict(
                available=False, reason="找不到可用的 ASR 本地模型", python=missing, models="", server="",
                runtime_available=False, runtime_reason="找不到可執行的字幕 Python"))
            with patch.object(environment.sys, "frozen", True, create=True):
                env = SubtitleQueueMixin._subtitle_item_environment(svc, item)
        self.assertTrue(env["runtime_available"])
        self.assertEqual(env["runtime_reason"], "")
        self.assertEqual(env["runtime_python"], sys.executable)
        self.assertFalse(env["available"])  # local stays gated by its own reason

    def test_frozen_child_commands_reenter_the_executable_and_route_back(self):
        entry = load_frozen_entry()
        with patch.object(environment.sys, "frozen", True, create=True):
            worker = environment.subtitle_command("ignored", "oh_streamer.subtitles.worker", "--job", Path("j.json"))
        self.assertEqual(worker, [sys.executable, "--subtitle-worker", "--job", "j.json"])
        self.assertEqual(entry.route(worker[1:]), ("oh_streamer.subtitles.worker", ["--job", "j.json"]))

    def test_dev_child_command_uses_module_flag(self):
        self.assertEqual(environment.subtitle_command("py", "oh_streamer.subtitles.worker", "--job", "j"),
                         ["py", "-m", "oh_streamer.subtitles.worker", "--job", "j"])
        self.assertEqual(environment.subtitle_command("", "oh_streamer.subtitles.worker")[:2], [sys.executable, "-m"])


class CustomModelsDirectoryTests(unittest.TestCase):
    def paths(self, settings, env=None):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env or {}, clear=False):
            if not env:
                os.environ.pop("OH_STREAMER_SUBTITLE_MODELS", None)
            return Path(environment.runtime_paths(settings, root=tmp)["models"]), Path(tmp)

    def test_default_is_unchanged_when_unset_or_blank(self):
        for settings in ({}, {"custom_models_directory": ""}, {"custom_models_directory": "   "}):
            with self.subTest(settings=settings):
                with tempfile.TemporaryDirectory() as tmp:
                    dev = Path(tmp) / "testdata/subtitle_poc/models"
                    dev.mkdir(parents=True)
                    with patch.dict(os.environ, {}, clear=False):
                        for k in ("OH_STREAMER_SUBTITLE_MODELS", "OH_STREAMER_INSTALL_ROOT"):
                            os.environ.pop(k, None)
                        models = Path(environment.runtime_paths(settings, root=tmp)["models"])
                    self.assertEqual(models, dev.resolve())

    def test_custom_directory_used_when_set(self):
        with tempfile.TemporaryDirectory() as custom:
            models, _ = self.paths({"custom_models_directory": custom})
            self.assertEqual(models, Path(custom).resolve())

    def test_precedence_env_then_subtitle_models_then_custom(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            settings = {"subtitle_models": b, "custom_models_directory": c}
            self.assertEqual(self.paths(settings, {"OH_STREAMER_SUBTITLE_MODELS": a})[0], Path(a).resolve())
            self.assertEqual(self.paths(settings)[0], Path(b).resolve())
            self.assertEqual(self.paths({"subtitle_models": "", "custom_models_directory": c})[0], Path(c).resolve())

    def test_queue_settings_store_and_detect_use_it(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as custom, \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OH_STREAMER_SUBTITLE_MODELS", None)
            store = QueueStore(Path(tmp) / "queue")
            self.assertEqual(store.get_settings()["custom_models_directory"], "")
            store.set_settings(custom_models_directory=custom)
            settings = store.get_settings()
            self.assertEqual(settings["custom_models_directory"], custom)
            self.assertEqual(environment.detect(settings, root=tmp)["models"], str(Path(custom).resolve()))


if __name__ == "__main__":
    unittest.main()


class SharedEnvironmentDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.userdata = self.root / "userdata"
        env = {k: v for k, v in os.environ.items() if k not in (
            "OH_STREAMER_INSTALL_ROOT", "OH_STREAMER_SUBTITLE_PYTHON", "OH_STREAMER_SUBTITLE_MODELS")}
        patcher = patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        data = patch("oh_streamer.platform.platform_support.data_dir", return_value=self.userdata)
        data.start()
        self.addCleanup(data.stop)
        self.addCleanup(self.tmp.cleanup)

    def make_dev(self):
        base = self.root / "testdata/subtitle_poc"
        py = base / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
        py.parent.mkdir(parents=True)
        py.write_bytes(b"x")
        (base / "models").mkdir()
        return py, base / "models"

    def test_default_models_dir_is_under_user_data(self):
        self.assertEqual(environment.default_models_dir(), self.userdata / "subtitle_models")

    def test_installed_uses_current_python_and_user_models_even_if_dev_files_exist(self):
        self.make_dev()
        os.environ["OH_STREAMER_INSTALL_ROOT"] = str(self.root / "install")
        paths = environment.runtime_paths({}, root=self.root)
        self.assertEqual(paths["python"], str(Path(sys.executable).resolve()))
        self.assertEqual(paths["models"], str((self.userdata / "subtitle_models").resolve()))

    def test_dev_prefers_poc_venv_and_models_when_present(self):
        py, models = self.make_dev()
        paths = environment.runtime_paths({}, root=self.root)
        self.assertEqual(paths["python"], str(py.resolve()))
        self.assertEqual(paths["models"], str(models.resolve()))

    def test_dev_falls_back_when_poc_files_missing(self):
        paths = environment.runtime_paths({}, root=self.root)
        self.assertEqual(paths["python"], str(Path(sys.executable).resolve()))
        self.assertEqual(paths["models"], str((self.userdata / "subtitle_models").resolve()))

    def test_overrides_beat_installed_defaults(self):
        os.environ["OH_STREAMER_INSTALL_ROOT"] = str(self.root / "install")
        os.environ["OH_STREAMER_SUBTITLE_PYTHON"] = str(self.root / "py.exe")
        paths = environment.runtime_paths({"subtitle_models": str(self.root / "m")}, root=self.root)
        self.assertEqual(paths["python"], str((self.root / "py.exe").resolve()))
        self.assertEqual(paths["models"], str((self.root / "m").resolve()))

    def test_runtime_default_model_path_follows_same_rule(self):
        from oh_streamer.subtitles import runtime
        os.environ["OH_STREAMER_INSTALL_ROOT"] = str(self.root / "install")
        self.assertEqual(runtime.default_model_path("anime-whisper"),
                         (self.userdata / "subtitle_models").resolve() / "anime-whisper")
