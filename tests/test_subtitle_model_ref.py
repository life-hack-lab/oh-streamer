import tempfile
import unittest
from pathlib import Path

from oh_streamer.subtitles import configuration, environment, fallback
from oh_streamer.queue import subtitle_view


class ModelRefTests(unittest.TestCase):
    def test_model_inside_the_models_folder_is_stored_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = {"subtitle_models": str(root)}
            row = {"id": "m", "kind": "translation", "provider": "local", "recipe": "sakura-v1",
                   "model_path": str(root / "m.gguf")}
            ref = environment.model_ref(row, settings)
            self.assertEqual(ref["model_name"], "m.gguf")
            self.assertNotIn("model_path", ref)
            self.assertEqual(Path(environment.model_file(ref, settings)), root / "m.gguf")
            dual = environment.model_ref(dict(row, recipe="__dual__", model_path=str(root)), settings)
            self.assertEqual(dual["model_name"], ".")
            self.assertEqual(Path(environment.model_file(dual, settings)), root)

    def test_model_outside_the_folder_keeps_its_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            file = Path(other) / "mine.gguf"
            file.write_bytes(b"x")
            row = {"id": "u", "kind": "translation", "provider": "local", "model_path": str(file)}
            ref = environment.model_ref(row, {"subtitle_models": tmp})
            self.assertEqual(ref["model_path"], str(file))
            self.assertNotIn("model_name", ref)
            self.assertEqual(environment.model_file(ref, {"subtitle_models": tmp}), str(file))

    def test_legacy_absolute_path_falls_back_to_the_current_models_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "m.gguf").write_bytes(b"x")
            legacy = {"provider": "local", "model_path": "Z:/old-place/m.gguf"}
            self.assertEqual(Path(environment.model_file(legacy, {"subtitle_models": tmp})), root.resolve() / "m.gguf")
            self.assertIsNone(environment.model_file({"provider": "local"}, {}))

    def test_identity_survives_moving_the_models_folder(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for root in (a, b):
                (Path(root) / "m.gguf").write_bytes(b"12345")
            row = {"id": "m", "kind": "translation", "provider": "local", "recipe": "sakura-v1",
                   "model_name": "m.gguf"}
            first = fallback.identity(row, {"subtitle_models": a})
            second = fallback.identity(row, {"subtitle_models": b})
            self.assertEqual(first["model_name"], "m.gguf")
            self.assertNotIn("model_path", first)
            self.assertEqual({k: v for k, v in first.items() if k != "revision"},
                             {k: v for k, v in second.items() if k != "revision"})
            self.assertEqual(fallback.identity(dict(row, model_name="gone.gguf"), {"subtitle_models": a})["revision"],
                             "missing")

    def test_resolve_stores_the_model_name_not_the_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = {"subtitle_models": tmp}
            profile = configuration.migrate({}, {})
            profile["active_mode"] = "local"
            profile["profiles"]["local"].update(asr="whisper-ja-1.5B", translation="sakura-galtransl-7b-v3.7")
            resolved = configuration.resolve(profile, settings)
            self.assertTrue(any(r and r.get("provider") == "local" for r in resolved["models"].values()))
            for row in resolved["models"].values():
                if row and row.get("provider") == "local":
                    self.assertIn("model_name", row)
                    self.assertNotIn("model_path", row)

    def test_view_names_a_local_row_by_its_model_name(self):
        row = {"id": "x", "kind": "translation", "provider": "local", "model_name": "Sub/foo.gguf"}
        self.assertEqual(subtitle_view._row_name(row), "foo.gguf")


if __name__ == "__main__":
    unittest.main()
