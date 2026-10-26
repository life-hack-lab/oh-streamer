"""Real CPU ffmpeg integration for concise/full subtitle mux and reruns."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles import worker
from oh_streamer.subtitles.core import parse_srt
from oh_streamer.subtitles.tracks import CONCISE_TITLE, FULL_TITLE


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], encoding="utf-8"))


def extract(path, stream_index):
    return parse_srt(subprocess.check_output([
        worker.runtime.resolve_ffmpeg(), "-nostdin", "-v", "error", "-copyts", "-i", str(path),
        "-map", f"0:{stream_index}", "-f", "srt", "pipe:1"
    ], encoding="utf-8"))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class QualityMuxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="quality-dual-mux-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.concise = self.write("concise.srt", "1\n00:00:00,250 --> 00:00:00,900\n共同譯文一\n\n"
                                  "2\n00:00:01,100 --> 00:00:01,700\n共同譯文二\n")
        self.full = self.write("full.srt", self.concise.read_text(encoding="utf-8") +
                               "\n3\n00:00:01,800 --> 00:00:02,100\n啾、啾噗……\n\n"
                               "4\n00:00:02,200 --> 00:00:02,700\n（低信心）完整軌補句\n")
        self.japanese = self.write("ja.srt", "1\n00:00:00,250 --> 00:00:00,900\n日本語一\n\n"
                                   "2\n00:00:01,100 --> 00:00:01,700\n日本語二\n")
        self.user = self.write("user.srt", "1\n00:00:00,400 --> 00:00:01,000\n元の日本語字幕\n")
        self.tracks = [(str(self.concise), "chi", CONCISE_TITLE), (str(self.full), "chi", FULL_TITLE),
                       (str(self.japanese), "jpn", "日文（AI）")]

    def write(self, filename, text):
        path = self.folder / filename
        path.write_text(text, encoding="utf-8")
        return path

    def source(self, origin):
        path = self.folder / f"source-{origin}.mkv"
        subprocess.run([
            worker.runtime.resolve_ffmpeg(), "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=s=64x48:r=10:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
            "-i", str(self.user), "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-c:s", "srt",
            "-metadata:s:s:0", "language=jpn", "-metadata:s:s:0", "title=使用者日文",
            "-disposition:s:0", "0", "-output_ts_offset", str(origin), str(path)
        ], check=True)
        return path

    def assert_cues_equal(self, actual, expected, offset):
        self.assertEqual(len(actual), len(expected))
        for actual_cue, expected_cue in zip(actual, expected):
            self.assertEqual(actual_cue.text, expected_cue.text)
            self.assertAlmostEqual(actual_cue.start, expected_cue.start + offset, delta=.002)
            self.assertAlmostEqual(actual_cue.end, expected_cue.end + offset, delta=.002)

    def verify(self, output, offset, original_user):
        streams = [s for s in probe(output)["streams"] if s["codec_type"] == "subtitle"]
        self.assertEqual(len(streams), 4)
        by_title = {s["tags"]["title"]: s for s in streams}
        self.assertEqual(set(by_title), {"使用者日文", CONCISE_TITLE, FULL_TITLE, "日文（AI）"})
        for path, lang, title in self.tracks:
            stream = by_title[title]
            self.assertEqual(stream["tags"]["language"], lang)
            self.assertEqual(stream["disposition"]["default"], int(title == CONCISE_TITLE))
            self.assert_cues_equal(extract(output, stream["index"]),
                                   parse_srt(Path(path).read_text(encoding="utf-8")), offset)
        self.assertEqual(by_title["使用者日文"]["tags"]["language"], "jpn")
        self.assert_cues_equal(extract(output, by_title["使用者日文"]["index"]), original_user, 0)

    def test_actual_two_chinese_tracks_and_japanese_preserve_all_cues_with_source_clock(self):
        for origin in (0, 4):
            with self.subTest(origin=origin):
                source = self.source(origin)
                before = probe(source)
                offset = float(next(s for s in before["streams"] if s["codec_type"] == "audio")["start_time"])
                original_user = extract(source, next(s for s in before["streams"] if s["codec_type"] == "subtitle")["index"])
                output = self.folder / f"output-{origin}.mkv"
                worker.mux({"video": str(source), "output": str(output), "tracks": self.tracks})
                self.verify(output, offset, original_user)

    def test_rerun_replaces_both_chinese_tracks_and_generated_japanese_without_duplicates(self):
        source = self.source(4)
        before = probe(source)
        original_user = extract(source, next(s for s in before["streams"] if s["codec_type"] == "subtitle")["index"])
        first, second = self.folder / "first.mkv", self.folder / "second.mkv"
        worker.mux({"video": str(source), "output": str(first), "tracks": self.tracks, "audio_offset": 4})
        self.concise.write_text(self.concise.read_text(encoding="utf-8").replace("共同譯文", "更新譯文"), encoding="utf-8")
        self.full.write_text(self.full.read_text(encoding="utf-8").replace("共同譯文", "更新譯文"), encoding="utf-8")
        worker.mux({"video": str(first), "output": str(second), "tracks": self.tracks, "audio_offset": 4})
        self.verify(second, 4, original_user)


if __name__ == "__main__":
    unittest.main()
