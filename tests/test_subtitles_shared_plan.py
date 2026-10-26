"""CPU shared-plan integration: one VAD pass, exact spans, empty ASR records."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles import asr, runtime


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy runtime unavailable")
class SharedPlanTests(unittest.TestCase):
    def test_prepare_once_and_run_two_models_on_identical_pcm_spans_including_empty(self):
        import numpy as np
        audio = np.full(7 * 16000, .02, dtype=np.float32)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                     inference_mode=nullcontext)
        seen = []
        def resample(source, output, **options):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fixture pcm")
            return len(audio) / 16000
        def pipeline(*args):
            calls = []
            def run(payload, **options):
                calls.append(len(payload["array"]))
                return {"text": "ここへ来て" if len(calls) != 2 else ""}
            seen.append(calls)
            return run, None, None, "float32"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(runtime, "source_identity", return_value=("source", {"fixture": True})), \
                    patch.object(runtime, "resample_to_wav", side_effect=resample) as resampler, \
                    patch.object(runtime, "load_wav_float32", return_value=audio), \
                    patch.object(asr, "_detect_speech_intervals", return_value=([{"start": 2., "end": 3.}], .1, "fixture")) as vad, \
                    patch.object(runtime, "local_model_fingerprint", return_value="model"), \
                    patch.object(asr, "_make_pipeline", side_effect=pipeline):
                plan = asr.prepare_shared_spans("fixture.mp4", root / "shared", cache_dir=root / "cache")
                self.assertEqual(plan["groups"], [{"start": 0., "end": 2., "gap": True},
                                                {"start": 2., "end": 3., "gap": False},
                                                {"start": 3., "end": 7., "gap": True}])
                cached = asr.prepare_shared_spans("fixture.mp4", root / "shared", cache_dir=root / "cache",
                                                  quality_config={"f7": {"avg_logprob_min": -.9}})
                self.assertEqual(cached["plan_key"], plan["plan_key"])
                first = asr.run_asr("fixture.mp4", root / "a1", model="whisper-ja-1.5b",
                                    cache_dir=root / "cache", segmentation="utterance", shared_plan=plan)
                second = asr.run_asr("fixture.mp4", root / "a2", cache_dir=root / "cache",
                                     segmentation="utterance", shared_plan=plan["plan_path"])
                self.assertEqual(resampler.call_count, 1)
                self.assertEqual(vad.call_count, 1)
                self.assertEqual(seen, [[32000, 16000, 64000], [32000, 16000, 64000]])
                self.assertEqual([(row["start"], row["end"]) for row in first["raw_segments"]],
                                 [(row["start"], row["end"]) for row in second["raw_segments"]])
                self.assertEqual(len(first["raw_segments"]), 3)
                self.assertEqual(first["raw_segments"][1]["text"], "")
                self.assertEqual(len(first["segments"]), 2)
                self.assertTrue(first["metrics"]["shared_vad_reused"])
                self.assertEqual(first["metrics"]["vad_seconds"], 0)
                self.assertEqual(first["shared_plan_key"], second["shared_plan_key"])

    def test_plan_identity_rejects_changed_source_settings_or_tampered_spans(self):
        import numpy as np
        audio = np.zeros(2 * 16000, dtype=np.float32)
        fake_torch = SimpleNamespace()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pcm = root / "audio" / "source.16k-mono.wav"
            pcm.parent.mkdir()
            pcm.write_bytes(b"fixture")
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(runtime, "source_identity", return_value=("source", {})), \
                    patch.object(runtime, "resample_to_wav", return_value=2), \
                    patch.object(runtime, "load_wav_float32", return_value=audio), \
                    patch.object(asr, "_detect_speech_intervals", return_value=([], .1, "fixture")), \
                    patch.object(runtime, "local_model_fingerprint", return_value="model"):
                plan = asr.prepare_shared_spans("fixture", root / "shared", cache_dir=root)
                changed = dict(plan, groups=[{"start": 0, "end": 1, "gap": True}])
                with self.assertRaisesRegex(ValueError, "identity"):
                    asr.run_asr("fixture", root / "out", segmentation="utterance", shared_plan=changed)
                with self.assertRaisesRegex(ValueError, "settings"):
                    asr.run_asr("fixture", root / "out", segmentation="utterance", shared_plan=plan, vad_threshold=.5)
                with patch.object(runtime, "source_identity", return_value=("other-source", {})):
                    with self.assertRaisesRegex(ValueError, "source"):
                        asr.run_asr("fixture", root / "out", segmentation="utterance", shared_plan=plan)
                settings = dict(plan["settings"], version="old")
                with self.assertRaisesRegex(ValueError, "version"):
                    asr._read_shared_plan(dict(plan, settings=settings))


if __name__ == "__main__":
    unittest.main()
