"""CPU-only tests for the isolated subtitle mux stage."""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import os
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from oh_streamer.queue.subtitle_jobs import replace_guarded_file, replacement_guard
from oh_streamer.subtitles import cli, worker, pipeline


def ffprobe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class SubtitleMuxTests(unittest.TestCase):
    def test_remux_command_uses_stream_copy_and_maps_explicit_subtitle_inputs(self):
        command = worker.remux_command(
            "source.mkv", "output.mkv",
            [("traditional.srt", "chi", "繁體中文（AI）"), ("japanese.srt", "jpn", "日文（AI）")],
            offset=3.25, existing_subtitles=(4, 7),
        )

        self.assertEqual(command[0], worker.runtime.resolve_ffmpeg())
        self.assertIn("-copyts", command)
        self.assertEqual(command[command.index("-i") + 1], "source.mkv")
        self.assertEqual(command.count("-itsoffset"), 2)
        self.assertEqual(command[command.index("-itsoffset") + 1], "3.250000")
        self.assertIn("-map", command)
        self.assertIn("-0:4", command)
        self.assertIn("-0:7", command)
        self.assertIn("1:0", command)
        self.assertIn("2:0", command)
        self.assertEqual(command[command.index("-c") + 1], "copy")
        self.assertIn("-avoid_negative_ts", command)

    def test_remux_command_without_subtitles_only_maps_the_source(self):
        command = worker.remux_command("source.mkv", "output.mkv", [])

        self.assertEqual(command[command.index("-map") + 1], "0")
        self.assertEqual(command.count("-map"), 1)
        self.assertNotIn("-itsoffset", command)
        self.assertNotIn("-disposition:s:0", command)
        self.assertEqual(command[command.index("-c") + 1], "copy")

    def test_mux_preserves_nonzero_source_clock_and_applies_audio_offset_to_soft_subtitle(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-mux-") as tmp:
            folder = Path(tmp)
            source = folder / "source.mkv"
            output = folder / "output.mkv"
            srt = folder / "sakura.zh-TW.srt"
            jpn_srt = folder / "sakura.jpn.srt"
            srt.write_text(
                "1\n00:00:00,250 --> 00:00:00,900\n第一句測試\n\n"
                "2\n00:00:01,100 --> 00:00:01,700\n第二句測試\n",
                encoding="utf-8",
            )
            jpn_srt.write_text("1\n00:00:00,400 --> 00:00:00,800\n日本語字幕\n", encoding="utf-8")

            # Keep the fixture tiny and CPU-only. The shifted PTS exercises inputs whose
            # audio/video clock does not begin at zero.
            command = [
                worker.runtime.resolve_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-output_ts_offset", "4", str(source),
            ]
            encoded = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(encoded.returncode, 0, encoded.stderr)

            before = ffprobe(source)
            av_before = [s for s in before["streams"] if s["codec_type"] in ("video", "audio")]
            audio_start = float(next(s for s in av_before if s["codec_type"] == "audio")["start_time"])
            self.assertGreater(audio_start, 3.5)
            self.assertGreater(float(before["format"]["start_time"]), 3.5)

            worker.mux({
                "video": str(source), "output": str(output),
                "tracks": [
                    (str(srt), "chi", "繁體中文（AI）"),
                    (str(jpn_srt), "jpn", "日文（AI）"),
                ],
            })

            after = ffprobe(output)
            av_after = [s for s in after["streams"] if s["codec_type"] in ("video", "audio")]
            self.assertEqual(
                [(s["codec_type"], s["codec_name"]) for s in av_after],
                [(s["codec_type"], s["codec_name"]) for s in av_before],
            )
            subtitles = [s for s in after["streams"] if s["codec_type"] == "subtitle"]
            self.assertEqual(len(subtitles), 2)
            self.assertEqual(subtitles[0]["tags"]["language"], "chi")
            self.assertEqual(subtitles[0]["tags"]["title"], "繁體中文（AI）")
            self.assertEqual(subtitles[0]["disposition"]["default"], 1)
            self.assertEqual(subtitles[1]["tags"]["language"], "jpn")
            self.assertEqual(subtitles[1]["tags"]["title"], "日文（AI）")
            self.assertEqual(subtitles[1]["disposition"]["default"], 0)
            self.assertAlmostEqual(float(subtitles[0]["start_time"]), audio_start + 0.25, delta=0.03)
            self.assertGreater(float(after["format"]["start_time"]), 3.5)

    def test_mux_without_generated_subtitles_preserves_the_source_streams(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-mux-empty-") as tmp:
            folder = Path(tmp)
            source = folder / "source.mkv"
            output = folder / "output.mkv"
            command = [
                worker.runtime.resolve_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", str(source),
            ]
            encoded = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(encoded.returncode, 0, encoded.stderr)

            before = ffprobe(source)
            worker.mux({"video": str(source), "output": str(output), "tracks": []})
            after = ffprobe(output)

            self.assertEqual(
                [(stream["codec_type"], stream["codec_name"]) for stream in after["streams"]],
                [(stream["codec_type"], stream["codec_name"]) for stream in before["streams"]],
            )
            self.assertFalse(any(stream["codec_type"] == "subtitle" for stream in after["streams"]))
            self.assertAlmostEqual(float(after["format"]["duration"]), float(before["format"]["duration"]), delta=0.1)

    def test_mux_replaces_existing_ai_subtitle_tracks_and_preserves_user_track(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-mux-rerun-") as tmp:
            folder = Path(tmp)
            source = folder / "source.mkv"
            first_output = folder / "first.mkv"
            second_output = folder / "second.mkv"
            ai_srt = folder / "ai.srt"
            user_srt = folder / "user.srt"
            ai_srt.write_text("1\n00:00:00,100 --> 00:00:00,600\n舊翻譯\n", encoding="utf-8")
            user_srt.write_text("1\n00:00:00,200 --> 00:00:00,500\n使用者字幕\n", encoding="utf-8")

            command = [
                worker.runtime.resolve_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=1",
                "-f", "srt", "-i", str(user_srt), "-map", "0:v:0", "-map", "1:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:s", "srt",
                "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=User supplied",
                str(source),
            ]
            encoded = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(encoded.returncode, 0, encoded.stderr)

            worker.mux({
                "video": str(source), "output": str(first_output),
                "tracks": [(str(ai_srt), "chi", "繁體中文（AI）")], "audio_offset": 0,
            })
            ai_srt.write_text("1\n00:00:00,100 --> 00:00:00,600\n新翻譯\n", encoding="utf-8")
            worker.mux({
                "video": str(first_output), "output": str(second_output),
                "tracks": [(str(ai_srt), "chi", "繁體中文（AI）")], "audio_offset": 0,
            })

            subtitles = [s for s in ffprobe(second_output)["streams"] if s["codec_type"] == "subtitle"]
            self.assertEqual(len(subtitles), 2)
            self.assertEqual(
                {s.get("tags", {}).get("title") for s in subtitles},
                {"User supplied", "繁體中文（AI）"},
            )
            extracted = subprocess.run(
                [worker.runtime.resolve_ffmpeg(), "-nostdin", "-v", "error", "-i", str(second_output),
                 "-map", "0:s:1", "-f", "srt", "pipe:1"],
                capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(extracted.returncode, 0, extracted.stderr)
            self.assertIn("新翻譯", extracted.stdout)
            self.assertNotIn("舊翻譯", extracted.stdout)


class SubtitleReplacementGuardTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows share-mode replacement contract")
    def test_open_reader_blocks_guard_and_guard_blocks_new_readers_until_replace(self):
        with tempfile.TemporaryDirectory(
                prefix="subtitle-replacement-guard-with-a-deliberately-long-path-name-") as tmp:
            video = Path(tmp) / "output.mkv"
            subtitle_folder = Path(tmp) / "subtitles"
            subtitle_folder.mkdir()
            replacement = subtitle_folder / "mux.partial.mkv"
            video.write_bytes(b"old video placeholder")
            replacement.write_bytes(b"new video")

            with video.open("rb"):
                with self.assertRaises(RuntimeError):
                    with replacement_guard(video):
                        self.fail("replacement guard should not pass an existing reader")

            with replacement_guard(video):
                with self.assertRaises(PermissionError):
                    video.open("rb")
                replace_guarded_file(replacement, video)

            self.assertEqual(video.read_bytes(), b"new video")
            self.assertFalse(replacement.exists())
            self.assertEqual(list(Path(tmp).glob("output.mkv*")), [video])


class SubtitleWorkerDispatchTests(unittest.TestCase):
    def test_model_child_receives_shared_plan_vad_settings(self):
        from oh_streamer.subtitles import asr
        with tempfile.TemporaryDirectory(prefix="subtitle-worker-model-") as tmp:
            folder = Path(tmp)
            job = folder / "asr.job.json"
            job.write_text(json.dumps({
                "stage": "asr_model", "source": "source.wav", "folder": str(folder),
                "asr": "whisper-ja-1.5B", "model_path": "model", "shared_plan": "plan.json",
                "vad_threshold": .4, "vad_rescue_chunk_seconds": 45,
            }), encoding="utf-8")
            with patch.object(asr, "run_asr") as run_asr, \
                    patch.object(worker, "emit") as emit, \
                    patch.object(worker.sys, "argv", ["subtitle-worker", "--job", str(job)]):
                worker.main()
        self.assertEqual(run_asr.call_args.kwargs["shared_plan"], "plan.json")
        self.assertEqual(run_asr.call_args.kwargs["vad_threshold"], .4)
        self.assertEqual(run_asr.call_args.kwargs["vad_rescue_chunk_seconds"], 45)
        self.assertTrue(any(call.kwargs.get("phase") == "asr" and call.kwargs.get("percent") == 100
                            for call in emit.call_args_list))

    def test_translate_stage_forces_generic_glossary_mode(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-worker-glossary-") as tmp:
            folder = Path(tmp)
            item_glossary = folder / "glossary.txt"
            item_glossary.write_text("must be ignored", encoding="utf-8")
            job = folder / "translate.job.json"
            job.write_text(json.dumps({
                "stage": "translate", "folder": str(folder),
                "model_path": str(folder / "sakura.gguf"), "server": "unused",
                "show_interjections": False, "glossary": str(item_glossary),
            }), encoding="utf-8")

            with patch.object(worker.sys, "argv", ["subtitle-worker", "--job", str(job)]), \
                    patch.object(cli, "translate_file") as translate:
                worker.main()

        self.assertTrue(translate.call_args.kwargs["generic_glossary_only"])
        self.assertTrue(translate.call_args.kwargs["gpu_managed"])

    def test_asr_stage_uses_queue_gpu_admission(self):
        with tempfile.TemporaryDirectory(prefix="subtitle-worker-asr-") as tmp:
            folder = Path(tmp)
            job = folder / "asr.job.json"
            job.write_text(json.dumps({
                "stage": "asr", "source": str(folder / "source.wav"), "folder": str(folder),
                "asr": "small", "model_path": str(folder / "model.bin"),
            }), encoding="utf-8")
            with patch.object(pipeline, "run_production_asr") as run_asr, \
                    patch.object(worker.sys, "argv", ["subtitle-worker", "--job", str(job)]):
                worker.main()

        self.assertTrue(run_asr.call_args.kwargs["gpu_managed"])
        self.assertEqual(run_asr.call_args.kwargs["asr_mode"], "dual")


if __name__ == "__main__":
    unittest.main()
