"""Clip timing, input mapping and an actual local FFmpeg trim."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_clip


class ClipTests(unittest.TestCase):
    def test_timestamp(self):
        for text, expected in (("90", 90), ("01:30.5", 90.5), ("1:02:03", 3723), ("0", 0)):
            self.assertEqual(make_clip.timestamp(text), expected)
        for text in ("-1", "nan", "inf", "1:60", "1:60:00", "1:2:3:4", "bad"):
            with self.assertRaises(argparse.ArgumentTypeError):
                make_clip.timestamp(text)

    def test_separate_audio_and_hls_input_options(self):
        cmd = make_clip.ffmpeg_command("ffmpeg", {
            "url": "https://cdn.test/video.m3u8", "audio_url": "https://cdn.test/audio.m4a",
            "http_headers": {"Referer": "https://site.test/"}}, 3.5, 2, Path("out.mp4"), 18,
            ["-allowed_extensions", "ALL"])
        self.assertEqual(cmd.count("-ss"), 2)
        self.assertEqual(cmd.count("-headers"), 2)
        self.assertIn("1:a:0", cmd)
        self.assertEqual(cmd.count("-allowed_extensions"), 1)
        self.assertNotIn("-vf", cmd)
        self.assertNotIn("-r", cmd)

    def test_external_resolver_and_quality(self):
        with patch.object(make_clip.source_resolve, "find_resolver", return_value={"command": ["resolver"]}), \
                patch.object(make_clip.source_resolve, "resolve_external", return_value=(
                    "https://cdn.test/master.m3u8", {"Referer": "https://site.test/"}, "Title")), \
                patch.object(make_clip.source_resolve, "resolve_with_ytdlp", return_value=([
                    {"url": "https://cdn.test/720.m3u8", "height": 720, "label": "720p", "http_headers": {}}], "master", None)):
            selected = make_clip.resolve_input("https://site.test/video", {}, "720p")
        self.assertEqual(selected["title"], "Title")
        self.assertEqual(selected["http_headers"]["Referer"], "https://site.test/")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_actual_trim_keeps_resolution_frame_rate_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, output = Path(directory) / "source.mp4", Path(directory) / "out.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25",
                            "-f", "lavfi", "-i", "sine=frequency=440", "-t", "3", "-c:v", "libx264",
                            "-c:a", "aac", str(source)], check=True)
            self.assertEqual(make_clip.main([str(source), "--start", "0.7", "--duration", "1.2", "-o", str(output)]), 0)
            info = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(info["actual_duration_seconds"], 1.2, delta=0.1)
            video = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"], video["avg_frame_rate"]), (160, 90, "25/1"))
            before = output.read_bytes()
            self.assertEqual(make_clip.main([str(source), "--start", "0", "--duration", "1", "-o", str(output)]), 1)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(make_clip.main([str(source), "--start", "30", "--duration", "1", "-o", str(Path(directory) / "empty.mp4")]), 1)
            self.assertFalse((Path(directory) / "empty.mp4").exists())


if __name__ == "__main__":
    unittest.main()
