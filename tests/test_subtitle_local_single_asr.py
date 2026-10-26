"""for_row of the local executor: single-model local ASR bindings (user's local mode: whisper-ja-1.5B)."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.queue import subtitle_local as sl
from oh_streamer.subtitles.providers.http import ProviderError


class SingleAsrForRowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.folder = self.root / "item"
        self.folder.mkdir()
        self.source = self.root / "v.mp4"
        self.source.write_bytes(b"x")
        self.models = self.root / "models"
        self.models.mkdir()
        self.config = {"asr": {"mode": "dual"}, "translation": {"model_file": "m.gguf"}}

    def deps(self):
        return sl.default_local_deps(self.folder, self.source, self.config,
                                     runtime_paths={"models": self.models, "server": "unused"})

    def row(self, alias="whisper-ja-1.5b", dir_name="whisper-ja-1.5B", **extra):
        # shape of model_registry.list_models() builtin local ASR rows as frozen into the plan
        return {"id": dir_name, "kind": "asr", "provider": "local", "recipe": alias,
                "model_path": str(self.models / dir_name), "builtin": True, **extra}

    def test_single_row_runs_child_with_single_mode(self):
        calls = []

        def child(jobs, **kwargs):
            calls.append(kwargs)
            return [{"segments": [], "speech_intervals": [], "model": "x", "asr_warnings": []} for _ in jobs]

        with patch("oh_streamer.subtitles.runtime.resample_window_to_wav",
                   side_effect=lambda _s, path, *a, **k: Path(path).write_bytes(b"")), \
                patch("oh_streamer.subtitles.pipeline.run_production_asr_windows_child", side_effect=child):
            sub = self.deps().for_row(self.row())
            sub.asr_batch([(0.0, 30.0)], self.root / "w")
        self.assertEqual(calls[0]["asr_mode"], "single")
        self.assertEqual(calls[0]["asr_model"], "whisper-ja-1.5B")
        self.assertEqual(Path(calls[0]["models_dir"]), self.models)

    def test_dual_and_single_identities_differ(self):
        dual = self.deps().for_row({"id": "__dual__", "kind": "asr", "provider": "local", "recipe": "__dual__",
                                    "model_path": str(self.models)})
        single = self.deps().for_row(self.row())
        other = self.deps().for_row(self.row("anime-whisper", "anime-whisper"))
        self.assertNotEqual(dual.asr_identity, single.asr_identity)
        self.assertNotEqual(single.asr_identity, other.asr_identity)
        self.assertNotEqual(dual.asr_model_id, single.asr_model_id)
        self.assertNotEqual(single.asr_model_id, other.asr_model_id)

    def test_unsupported_recipes_stay_unsupported(self):
        for row in (self.row("my-custom-decode", "custom"), self.row(model_path=""),
                    self.row("whisper-ja-1.5b", "elsewhere")):
            with self.subTest(row=row):
                with self.assertRaises(ProviderError) as ctx:
                    self.deps().for_row(row)
                self.assertEqual(ctx.exception.kind, "unsupported")


if __name__ == "__main__":
    unittest.main()
