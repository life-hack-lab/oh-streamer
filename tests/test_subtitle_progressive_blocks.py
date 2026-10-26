"""Windowed audio/VAD for a source that is still downloading (block-plan tests were removed in M7b).

ffmpeg is used only for a one-second synthetic tone; ASR/translation processes and torch are fakes.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np      # imported before sys.modules is patched below, so it is not reloaded

from oh_streamer.subtitles import asr, runtime


class WindowedAudioTests(unittest.TestCase):
    def test_window_wav_keeps_absolute_timestamps(self):
        try:
            ffmpeg = runtime.resolve_ffmpeg()
        except Exception:
            self.skipTest("ffmpeg unavailable")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "tone.wav"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                            "-ac", "1", "-ar", "16000", str(source)], check=True)
            out = Path(folder) / "window.wav"
            self.assertAlmostEqual(runtime.resample_window_to_wav(source, out, 4, 7), 7.0, delta=0.05)
            audio = runtime.load_wav_float32(out)
            self.assertEqual(float(np.abs(audio[:int(3.9 * 16000)]).max()), 0.0, "front padding is silence")
            self.assertGreater(float(np.abs(audio[int(4.5 * 16000):]).max()), 0.05)

    def test_shared_spans_for_a_window_are_analysed_locally_and_shifted(self):
        with tempfile.TemporaryDirectory() as folder:
            seen = {}

            def fake_window(source, destination, start, end, **kw):
                seen["window"] = (start, end)
                import wave
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(destination), "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(16000)
                    out.writeframes(b"\0\0" * int(end * 16000))
                return float(end)

            def fake_detect(audio, **kw):
                seen["analysed_seconds"] = len(audio) / 16000
                return [dict(start=1.0, end=2.0)], 0.0, "fake"

            with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}), \
                    patch.object(runtime, "resample_window_to_wav", side_effect=fake_window), \
                    patch.object(runtime, "source_identity", return_value=("key", {"kind": "file"})), \
                    patch.object(asr, "_detect_speech_intervals", side_effect=fake_detect), \
                    patch.object(asr, "utterance_groups", return_value=[(1.0, 2.0)]), \
                    patch.object(asr, "gap_inference_groups", return_value=[]):
                plan = asr.prepare_shared_spans("playlist.m3u8", Path(folder) / "shared", cache_dir=Path(folder) / "cache",
                                                window=(10.0, 13.0))
            self.assertEqual(seen["window"], (10.0, 13.0))
            self.assertEqual(seen["analysed_seconds"], 3.0, "only the window is analysed")
            self.assertEqual(plan["speech_intervals"], [dict(start=11.0, end=12.0)])
            self.assertEqual([(g["start"], g["end"]) for g in plan["groups"]], [(11.0, 12.0)])
            self.assertEqual(plan["audio_duration_seconds"], 13.0)
            self.assertEqual(plan["settings"]["window"], [10.0, 13.0])


if __name__ == "__main__":
    unittest.main()
