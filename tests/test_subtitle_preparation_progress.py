"""CPU-only measured progress tests for local subtitle preparation."""

import contextlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
_TEMP_ROOT = _STREAMER_ROOT / ".test-tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from oh_streamer.subtitles import asr, cli, core, runtime
from oh_streamer.subtitles.llm import LlamaServerClient


class FakeProcess:
    def __init__(self):
        self.pid = 4567
        self.return_code = None

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = 0

    def kill(self):
        self.return_code = -9

    def wait(self, timeout=None):
        self.return_code = 0 if self.return_code is None else self.return_code
        return self.return_code


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class SubtitlePreparationProgressTests(unittest.TestCase):
    def test_duration_uses_format_fallback_and_accounts_for_audio_start_time(self):
        probe = {
            "streams": [{"duration": "N/A", "start_time": "3.0"}],
            "format": {"duration": "43.0"},
        }
        with patch.object(runtime.shutil, "which", return_value="ffprobe"), patch.object(
            runtime.subprocess,
            "run",
            return_value=runtime.subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                stdout=json.dumps(probe),
                stderr="",
            ),
        ):
            duration = runtime._probe_audio_duration("fixture.mkv", "ffmpeg", None)

        self.assertEqual(duration, 40.0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_real_ffmpeg_resample_reports_media_time_progress(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            source = root / "source.wav"
            target = root / "audio" / "source.16k-mono.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(runtime.SAMPLE_RATE)
                audio.writeframes(b"\0\0" * runtime.SAMPLE_RATE * 2)

            updates = []
            duration = runtime.resample_to_wav(source, target, on_progress=lambda current, total: updates.append(
                (current, total)
            ))
            target_is_valid = runtime.valid_pcm_wav(target)

        self.assertAlmostEqual(duration, 2.0, places=2)
        self.assertTrue(target_is_valid)
        self.assertGreaterEqual(len(updates), 2)
        self.assertEqual(updates[0], (0.0, 2.0))
        self.assertEqual(updates[-1], (duration, duration))
        self.assertTrue(all(current <= following for (current, _), (following, _) in zip(
            updates,
            updates[1:],
        )))
        self.assertIsNone(runtime._time_from_ffmpeg_progress("out_time_us=N/A"))
        self.assertEqual(runtime._time_from_ffmpeg_progress("out_time_us=2500000"), 2.5)
        self.assertEqual(runtime._time_from_ffmpeg_progress("out_time=00:01:02.500"), 62.5)

    def test_asr_combines_audio_vad_and_model_milestones_and_throttles_silero(self):
        class FakeVadModel:
            def eval(self):
                return self

            def to(self, _device):
                return self

        def get_speech_timestamps(
            waveform,
            _model,
            *,
            progress_tracking_callback=None,
            **_kwargs,
        ):
            if progress_tracking_callback is not None:
                for percent in range(101):
                    progress_tracking_callback(percent)
            return [{"start": 0.1, "end": 1.8}]

        class FakePipe:
            def __call__(self, _payload, **_kwargs):
                return {
                    "chunks": [
                        {"text": "こんにちは", "timestamp": (0.1, 1.0)},
                    ],
                }

        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": type("Cuda", (), {"is_available": staticmethod(lambda: False)}),
                "from_numpy": staticmethod(lambda array: array),
                "inference_mode": staticmethod(contextlib.nullcontext),
            },
        )

        phase_events = []
        old_progress = []

        def resample(_source, _destination, *, on_progress):
            for current in (0.0, 0.5, 1.0, 1.5, 2.0):
                on_progress(current, 2.0)
            return 2.0

        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch.object(runtime, "local_model_fingerprint", return_value="local-model"),
                patch.object(runtime, "source_identity", return_value=("source-key", {"kind": "file"})),
                patch.object(runtime, "resample_to_wav", side_effect=resample),
                patch.object(runtime, "load_wav_float32", return_value=[0.0] * (runtime.SAMPLE_RATE * 2)),
                patch.object(asr, "_load_vad", return_value=(FakeVadModel(), get_speech_timestamps, "fixture VAD")),
                patch.object(asr, "_make_pipeline", return_value=(FakePipe(), None, None, "float32")),
            ):
                result = asr.run_asr(
                    "fixture.wav",
                    root / "out",
                    model_path=root / "model",
                    cache_dir=root / "cache",
                    force=True,
                    vad_rescue_chunk_seconds=0,
                    on_phase=lambda phase, percent, detail: phase_events.append(
                        (phase, percent, detail)
                    ),
                    on_progress=lambda completed, total: old_progress.append((completed, total)),
                )

        prep_values = [
            percent for phase, percent, _detail in phase_events
            if phase == "prepare_asr" and percent is not None
        ]
        self.assertTrue(prep_values)
        self.assertTrue(all(left <= right for left, right in zip(prep_values, prep_values[1:])))
        self.assertEqual(prep_values[-1], 100)
        self.assertTrue(any("音訊擷取與 PCM 載入完成" in detail for _, _, detail in phase_events))
        self.assertTrue(any("處理器已就緒" in detail for _, _, detail in phase_events))
        self.assertLess(len(phase_events), 80)
        asr_events = [(percent, detail) for phase, percent, detail in phase_events if phase == "asr"]
        self.assertEqual(asr_events[0][0], 0)
        self.assertEqual(asr_events[-1][0], 100)
        self.assertEqual(old_progress, [(1, 1)])
        self.assertEqual(result["segments"][0]["text"], "こんにちは")

    def test_translation_server_readiness_and_translation_progress_are_separate(self):
        class FakeClient:
            def __init__(self, *_args, on_phase=None, **_kwargs):
                self.on_phase = on_phase
                self.metrics = {}

            def __enter__(self):
                for percent, detail in (
                    (None, "正在檢查本機翻譯模型"),
                    (25, "本機翻譯模型與伺服器執行檔已驗證"),
                    (40, "本機伺服器啟動參數已準備完成"),
                    (60, "本機翻譯伺服器程序已啟動"),
                    (100, "本機翻譯伺服器已就緒"),
                ):
                    self.on_phase("prepare_translate", percent, detail)
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def translate(self, *_args, **_kwargs):
                return "1. 你好"

        events = []
        old_progress = []
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            source = root / "ja.json"
            model = root / "model.gguf"
            source.write_text(
                json.dumps(
                    {
                        "segments": [{"start": 0, "end": 1, "text": "こんにちは"}],
                        "speech_intervals": [[0, 1]],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            model.write_bytes(b"local fixture")
            with (
                patch("oh_streamer.subtitles.llm.LlamaServerClient", FakeClient),
                patch.object(cli, "assert_gpu_safe", return_value={"required": False}),
                patch.object(cli, "NvmlMemorySampler") as sampler,
                patch.object(core, "to_taiwan_traditional", side_effect=lambda text, converter=None: text),
            ):
                sampler.return_value.__enter__.return_value.to_dict.return_value = {}
                result = cli.translate_file(
                    source,
                    root / "out",
                    model="sakura",
                    model_path=model,
                    llama_server="mock-server",
                    force=True,
                    on_phase=lambda phase, percent, detail: events.append((phase, percent, detail)),
                    on_progress=lambda completed, total: old_progress.append((completed, total)),
                )

        preparation = [
            percent for phase, percent, _detail in events
            if phase == "prepare_translate" and percent is not None
        ]
        self.assertEqual(preparation, [25, 40, 60, 100])
        translation = [
            percent for phase, percent, _detail in events
            if phase == "translate" and percent is not None
        ]
        self.assertEqual(translation[0], 0)
        self.assertEqual(translation[-1], 100)
        self.assertEqual(old_progress, [(1, 1)])
        self.assertEqual(result["segments"][0]["text"], "你好")

    def test_llama_server_progress_reports_real_startup_milestones_once(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"fixture")
            events = []
            process = FakeProcess()
            client = LlamaServerClient(
                "sakura",
                model_path,
                "llama-server",
                port=8767,
                on_phase=lambda phase, percent, detail: events.append((phase, percent, detail)),
            )
            with patch("oh_streamer.subtitles.llm.shutil.which", return_value="llama-server"), patch(
                "oh_streamer.subtitles.llm.subprocess.Popen",
                return_value=process,
            ), patch("oh_streamer.subtitles.llm.urlopen", return_value=FakeResponse()):
                client.start()
                count_when_ready = len(events)
                client.start()
                self.assertEqual(len(events), count_when_ready)
                client.close()

        self.assertEqual(
            [(percent) for phase, percent, _detail in events if phase == "prepare_translate"],
            [None, 25, 40, 60, 100],
        )


if __name__ == "__main__":
    unittest.main()
