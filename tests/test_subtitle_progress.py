"""CPU-only contracts for queue subtitle phase and progress reporting."""

from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import inspect
import json
import math
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.queue import queue_store, subtitle_jobs
from oh_streamer.subtitles import asr, cli, core, runtime, worker, pipeline


def _temporary_directory(prefix):
    test_tmp = _STREAMER_ROOT / ".test-tmp"
    test_tmp.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=test_tmp)


class SubtitleProgressFieldsTests(unittest.TestCase):
    def test_allowed_stage_phases_are_normalized_for_the_scheduler(self):
        allowed = {
            "asr": ("prepare_asr", "asr"),
            "translate": ("prepare_translate", "translate"),
            "mux": ("mux",),
        }
        for stage, phases in allowed.items():
            for phase in phases:
                with self.subTest(stage=stage, phase=phase):
                    fields = subtitle_jobs.progress_fields(
                        stage,
                        {"phase": phase, "percent": 37.5, "detail": "fixture"},
                    )
                    self.assertEqual(
                        fields,
                        {"status": stage, "phase": phase, "percent": 37.5, "detail": "fixture"},
                    )

    def test_old_events_without_phase_remain_compatible_and_none_percent_is_valid(self):
        self.assertEqual(
            subtitle_jobs.progress_fields("asr", {"percent": None}),
            {"status": "asr", "phase": "asr", "percent": None, "detail": ""},
        )
        self.assertEqual(
            subtitle_jobs.progress_fields("translate", {"percent": 12}),
            {"status": "translate", "phase": "translate", "percent": 12.0, "detail": ""},
        )
        self.assertEqual(
            subtitle_jobs.progress_fields("mux", {"phase": "mux", "detail": 17}),
            {"status": "mux", "phase": "mux", "percent": None, "detail": "17"},
        )

    def test_invalid_phases_and_nonfinite_or_out_of_range_percentages_are_ignored(self):
        invalid = [
            {"phase": None, "percent": 0},
            {"phase": [], "percent": 0},
            {"phase": {}, "percent": 0},
            {"phase": "prepare_translate", "percent": 0},
            {"phase": "unknown", "percent": 0},
            {"phase": "asr", "percent": float("nan")},
            {"phase": "asr", "percent": float("inf")},
            {"phase": "asr", "percent": float("-inf")},
            {"phase": "asr", "percent": "nan"},
            {"phase": "asr", "percent": "not a number"},
            {"phase": "asr", "percent": -0.01},
            {"phase": "asr", "percent": 100.01},
        ]
        for message in invalid:
            with self.subTest(message=message):
                self.assertIsNone(subtitle_jobs.progress_fields("asr", message))
        self.assertIsNone(subtitle_jobs.progress_fields("waiting_gpu", {"percent": 0}))


class SubtitleProgressPersistenceTests(unittest.TestCase):
    class Harness:
        def __init__(self, root):
            self._lock = threading.RLock()
            self._tombstones = set()
            self.store = queue_store.QueueStore(Path(root))
            self.store.set_settings(subtitle_runtime_contract='legacy')

    def test_status_transition_clears_stale_phase_and_detail(self):
        with _temporary_directory("subtitle-progress-state-") as tmp:
            harness = self.Harness(Path(tmp) / "queue")
            item = harness.store.add(
                "input.mkv",
                "file",
                {},
                None,
                {"subtitles": True},
                state="done",
            )
            harness.store.update(
                item.id,
                subtitle={
                    "status": "asr",
                    "phase": "prepare_asr",
                    "detail": "extracting audio",
                    "percent": None,
                },
            )

            subtitle_jobs.SubtitleQueueMixin._subtitle_update(
                harness,
                item.id,
                status="waiting_gpu",
                percent=0,
            )

            saved = harness.store.get(item.id).subtitle
            self.assertEqual(saved["status"], "waiting_gpu")
            self.assertEqual(saved["phase"], "")
            self.assertEqual(saved["detail"], "")

    def test_progress_update_with_explicit_phase_preserves_new_phase(self):
        with _temporary_directory("subtitle-progress-phase-") as tmp:
            harness = self.Harness(Path(tmp) / "queue")
            item = harness.store.add(
                "input.mkv", "file", {}, None, {"subtitles": True}, state="done"
            )
            subtitle_jobs.SubtitleQueueMixin._subtitle_update(
                harness,
                item.id,
                **subtitle_jobs.progress_fields(
                    "translate",
                    {"phase": "prepare_translate", "percent": None, "detail": "launch"},
                ),
            )

            saved = harness.store.get(item.id).subtitle
            self.assertEqual(saved["status"], "translate")
            self.assertEqual(saved["phase"], "prepare_translate")
            self.assertEqual(saved["detail"], "launch")
            self.assertIsNone(saved["percent"])

    def test_restart_resets_active_saved_phase_for_legacy_queue_items(self):
        item = queue_store.Item.from_dict(
            {
                "id": "active-subtitle-phase",
                "state": "done",
                "subtitle": {
                    "status": "translate",
                    "phase": "prepare_translate",
                    "detail": "starting translation",
                    "percent": 23,
                },
            }
        )
        self.assertEqual(item.subtitle["status"], "queued")
        self.assertEqual(item.subtitle["percent"], 0)
        self.assertEqual(item.subtitle["phase"], "")
        self.assertEqual(item.subtitle["detail"], "")


class SubtitleWorkerPhaseDispatchTests(unittest.TestCase):
    def _run_worker_stage(self, stage, monkeypatched_api, api_name, phase):
        with _temporary_directory(f"subtitle-dispatch-{stage}-") as tmp:
            folder = Path(tmp)
            job_path = folder / f"{stage}.job.json"
            job = {
                "stage": stage,
                "folder": str(folder),
                "source": str(folder / "source.wav"),
                "asr": "anime-whisper",
                "model_path": str(folder / "model"),
                "server": "http://127.0.0.1:1",
                "show_interjections": False,
            }
            job_path.write_text(json.dumps(job), encoding="utf-8")
            emitted = []
            observed_kwargs = {}

            def stage_api(*_args, **kwargs):
                observed_kwargs.update(kwargs)
                kwargs["on_phase"](phase, 12.5, "preparation detail")
                kwargs["on_phase"](stage, 50, "stage progress")

            with patch.object(monkeypatched_api, api_name, side_effect=stage_api), \
                    patch.object(
                        worker, "emit", side_effect=lambda **event: emitted.append(event)
                    ), \
                    patch.object(worker.sys, "argv", ["subtitle-worker", "--job", str(job_path)]):
                worker.main()

            self.assertIs(observed_kwargs["on_phase"], worker.report_phase)
            self.assertTrue(observed_kwargs.get("gpu_managed"))
            self.assertNotIn("on_progress", observed_kwargs)
            phase_events = [event for event in emitted if event.get("type") == "subtitle_progress"]
            self.assertTrue(
                any(event.get("phase") == phase and event.get("detail") == "preparation detail"
                    for event in phase_events),
                phase_events,
            )
            self.assertTrue(
                any(event.get("phase") == stage and event.get("percent") == 50
                    and event.get("detail") == "stage progress"
                    for event in phase_events),
                phase_events,
            )
            self.assertEqual(emitted[-1].get("type"), "done")

    def test_asr_worker_forwards_phase_callback(self):
        self._run_worker_stage("asr", pipeline, "run_production_asr", "prepare_asr")

    def test_translation_worker_forwards_phase_callback(self):
        self._run_worker_stage("translate", cli, "translate_file", "prepare_translate")


class SubtitleApiPhaseCompatibilityTests(unittest.TestCase):
    def test_asr_and_translation_keep_legacy_progress_callbacks(self):
        for api in (asr.run_asr, cli.translate_file):
            with self.subTest(api=api.__name__):
                parameters = inspect.signature(api).parameters
                self.assertIn("on_progress", parameters)
                self.assertIn("on_phase", parameters)

    def test_asr_cache_hit_reports_preparation_and_asr_without_loading_models(self):
        with _temporary_directory("subtitle-asr-phase-cache-") as tmp:
            root = Path(tmp)
            events = []
            cached = {
                "cache_key": "cache-key",
                "segments": [],
                "raw_segments": [],
                "speech_intervals": [],
                "metrics": {},
            }
            spec = {"model_id": "local-test", "torch_dtype": "float32", "generate_kwargs": {}}
            with patch.object(asr.runtime, "resolve_model", return_value=("local-test", spec)), \
                    patch.object(
                        asr.runtime, "local_model_fingerprint", return_value="fingerprint"
                    ), \
                    patch.object(
                        asr.runtime,
                        "source_identity",
                        return_value=("source-key", "source.mkv"),
                    ), \
                    patch.object(asr.runtime, "make_job_key", return_value="cache-key"), \
                    patch.object(asr, "_cache_matches", return_value=cached):
                result = asr.run_asr(
                    root / "source.mkv",
                    root / "output",
                    cache_dir=root / "cache",
                    model_path=root / "model",
                    on_progress=lambda *_args: self.fail("cache hit should not report ASR batches"),
                    on_phase=lambda *event: events.append(event),
                )

            self.assertTrue(result["metrics"]["cache_hit"])
            self.assertEqual([event[0] for event in events], ["prepare_asr", "prepare_asr", "asr"])
            self.assertIsNone(events[0][1])
            self.assertEqual(events[-2][1], 100)
            self.assertTrue(events[-2][2])
            self.assertEqual(events[-1][1], 100)
            self.assertTrue(events[-1][2])

    def test_translation_cache_hit_reports_both_phases_without_starting_server(self):
        with _temporary_directory("subtitle-translate-phase-cache-") as tmp:
            root = Path(tmp)
            source = root / "ja.json"
            source.write_text("{}", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            (output / "sakura.zh-TW.json").write_text(
                json.dumps({"cache_key": "cache-key", "segments": []}),
                encoding="utf-8",
            )
            events = []
            with patch.object(cli, "translation_key", return_value="cache-key"), \
                    patch(
                        "oh_streamer.subtitles.llm.LlamaServerClient",
                        side_effect=AssertionError("cache hit started server"),
                    ):
                result = cli.translate_file(
                    source,
                    output,
                    model="sakura",
                    model_path=root / "model.gguf",
                    llama_server="unused",
                    on_progress=lambda *_args: self.fail(
                        "cache hit should not report translation batches"
                    ),
                    on_phase=lambda *event: events.append(event),
                )

            self.assertEqual(result["cache_key"], "cache-key")
            self.assertEqual(
                [event[0] for event in events],
                ["prepare_translate", "prepare_translate", "translate"],
            )
            self.assertIsNone(events[0][1])
            self.assertEqual(events[-2][1], 100)
            self.assertTrue(events[-2][2])
            self.assertEqual(events[-1][1], 100)
            self.assertTrue(events[-1][2])


class SubtitleMuxProgressTests(unittest.TestCase):
    def test_mux_worker_emits_bounded_monotonic_progress_with_nonzero_media_clock(self):
        with _temporary_directory("subtitle-mux-progress-") as tmp:
            folder = Path(tmp)
            source = folder / "source.mkv"
            output = folder / "output.mkv"
            srt = folder / "sakura.zh-TW.srt"
            srt.write_text(
                "1\n00:00:00,250 --> 00:00:00,900\n第一句測試\n\n"
                "2\n00:00:01,100 --> 00:00:01,700\n第二句測試\n",
                encoding="utf-8",
            )
            encoded = subprocess.run(
                [
                    runtime.resolve_ffmpeg(),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=10:duration=2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=2",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-c:a",
                    "aac",
                    "-output_ts_offset",
                    "4",
                    str(source),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(encoded.returncode, 0, encoded.stderr)
            source_probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of",
                        "json", str(source),
                    ],
                    encoding="utf-8",
                )
            )
            self.assertGreater(float(source_probe["format"]["start_time"]), 3.5)
            self.assertGreater(
                float(
                    next(
                        stream for stream in source_probe["streams"]
                        if stream["codec_type"] == "audio"
                    )["start_time"]
                ),
                3.5,
            )
            job_path = folder / "mux.job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "stage": "mux",
                        "video": str(source),
                        "output": str(output),
                        "tracks": [(str(srt), "chi", "繁體中文（AI）")],
                    }
                ),
                encoding="utf-8",
            )
            events = []
            parsed_srt_calls = []
            original_parse_srt = core.parse_srt

            def parsed_srt(text):
                parsed_srt_calls.append(True)
                return original_parse_srt(text)

            def emit(**event):
                if event.get("type") == "subtitle_progress" and event.get("percent") == 100:
                    self.assertEqual(
                        len(parsed_srt_calls),
                        2,
                        "100% must follow extracted and source SRT validation",
                    )
                    self.assertTrue(output.is_file())
                events.append(event)

            with patch.object(core, "parse_srt", side_effect=parsed_srt), \
                    patch.object(worker, "emit", side_effect=emit), \
                    patch.object(worker.sys, "argv", ["subtitle-worker", "--job", str(job_path)]):
                worker.main()

            progress = [event for event in events if event.get("type") == "subtitle_progress"]
            self.assertTrue(progress)
            self.assertTrue(all(event.get("phase") == "mux" for event in progress), progress)
            percentages = [
                event["percent"] for event in progress if event.get("percent") is not None
            ]
            self.assertEqual(percentages[0], 0)
            self.assertEqual(percentages[-3:], [90, 95, 100])
            self.assertTrue(
                all(math.isfinite(value) and 0 <= value <= 100 for value in percentages),
                percentages,
            )
            self.assertEqual(percentages, sorted(percentages), percentages)
            self.assertEqual(len(parsed_srt_calls), 2)
            self.assertEqual(events[-1].get("type"), "done")
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
