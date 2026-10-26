"""Real CPU FFmpeg regression: progressive HLS windows keep the media's zero origin."""
import array
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles import runtime, cloud_chunks


class ProgressiveAudioWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ffmpeg = runtime.resolve_ffmpeg()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))
        cls.temp = tempfile.TemporaryDirectory(prefix="hls-audio-window-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        source = cls.root / "clock.wav"
        # Every second has a distinct frequency: an extraction from the wrong live segment
        # cannot pass merely by returning/padding the right number of samples.
        samples = array.array("h", (int(9000 * math.sin(2 * math.pi * (220 + 37 * (i // 16000))
                                                     * (i % 16000) / 16000))
                                   for i in range(40 * 16000)))
        if sys.byteorder != "little":
            samples.byteswap()
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(samples.tobytes())
        cls.mp4 = cls.root / "clock.m4a"
        cls.run_ffmpeg(["-i", str(source), "-c:a", "aac", "-b:a", "96k", str(cls.mp4)])
        cls.playlist = cls.root / "prefix.m3u8"
        cls.run_ffmpeg(["-i", str(cls.mp4), "-c", "copy", "-hls_time", "4",
                        "-hls_playlist_type", "event", "-hls_segment_filename", str(cls.root / "seg_%03d.ts"),
                        str(cls.playlist)])
        # Only the first 7 complete segments are announced. Later segments stay unannounced,
        # as they would in the fixture's staging directory. No ENDLIST means downloading.
        lines = cls.playlist.read_text("utf-8").splitlines()
        prefix, segments = [], 0
        for line in lines:
            if line.startswith("#EXT-X-ENDLIST"):
                break
            if segments == 7:
                break
            prefix.append(line)
            if line and not line.startswith("#"):
                segments += 1
        cls.playlist.write_text("\n".join(prefix) + "\n", "utf-8")
        announced = {line for line in prefix if line and not line.startswith("#")}
        for segment in cls.root.glob("seg_*.ts"):
            if segment.name not in announced:
                segment.unlink()  # later download segments really do not exist yet

    @classmethod
    def run_ffmpeg(cls, args):
        subprocess.run([cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
                       capture_output=True, check=True, timeout=20,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    @staticmethod
    def samples(path):
        with wave.open(str(path), "rb") as audio:
            values = array.array("h", audio.readframes(audio.getnframes()))
        if sys.byteorder != "little":
            values.byteswap()
        return values

    def assert_clock(self, values, output_offset, source_start):
        # Probe in the middle of known one-second markers, away from AAC boundary transients.
        for probe in (0.5, 2.5, 4.5):
            begin = int((output_offset + probe - 0.2) * 16000)
            part = values[begin:begin + 6400]
            crossings = sum(a <= 0 < b for a, b in zip(part, part[1:]))
            measured = crossings / 0.4
            expected = 220 + 37 * int(source_start + probe)
            self.assertAlmostEqual(measured, expected, delta=4.0)

    def test_event_prefix_windows_have_correct_length_and_source_position(self):
        for start in (0.0, 5.25, 20.25):
            with self.subTest(start=start):
                out = self.root / f"window-{start}.wav"
                duration = runtime.resample_window_to_wav(self.playlist, out, start, start + 6,
                                                          ffmpeg=self.ffmpeg, timeout=15, pad=False)
                self.assertAlmostEqual(duration, 6.0, delta=0.01)
                self.assert_clock(self.samples(out), 0, start)

    def test_padded_event_window_adds_only_requested_leading_silence(self):
        out = self.root / "padded.wav"
        duration = runtime.resample_window_to_wav(self.playlist, out, 5.25, 11.25,
                                                  ffmpeg=self.ffmpeg, timeout=15, pad=True)
        self.assertAlmostEqual(duration, 11.25, delta=0.01)
        values = self.samples(out)
        self.assertTrue(all(value == 0 for value in values[:int(5.25 * 16000)]))
        self.assert_clock(values, 5.25, 5.25)

    def test_mp4_window_uses_the_same_clock_without_hls_input_options(self):
        out = self.root / "mp4.wav"
        duration = runtime.resample_window_to_wav(self.mp4, out, 5.25, 11.25,
                                                  ffmpeg=self.ffmpeg, timeout=15, pad=False)
        # Existing local-file input seeking may clip an AAC frame. This change deliberately
        # preserves that path and its cache semantics; precise HLS lengths are asserted above.
        self.assertAlmostEqual(duration, 6.0, delta=0.08)
        self.assert_clock(self.samples(out), 0, 5.25)

    def test_cloud_flac_window_has_the_same_clock_and_exact_length(self):
        out = self.root / "cloud.flac"
        self.assertEqual(cloud_chunks.extract_chunk_audio(self.playlist, out, 5.25, 6.0,
                                                          ffmpeg=self.ffmpeg, fmt="flac"), out)
        self.assertEqual(out.read_bytes()[:4], b"fLaC")
        decoded = self.root / "cloud-roundtrip.wav"
        self.run_ffmpeg(["-i", str(out), "-c:a", "pcm_s16le", str(decoded)])
        self.assertAlmostEqual(runtime.wav_duration(decoded), 6.0, delta=0.01)
        self.assert_clock(self.samples(decoded), 0, 5.25)
        self.assertEqual(list(self.root.glob(".hls-chunk-*")), [])

    def test_cloud_wav_window_retains_exact_hls_samples(self):
        out = self.root / "cloud.wav"
        cloud_chunks.extract_chunk_audio(self.playlist, out, 20.25, 6.0, ffmpeg=self.ffmpeg, fmt="wav")
        self.assertAlmostEqual(runtime.wav_duration(out), 6.0, delta=0.01)
        self.assert_clock(self.samples(out), 0, 20.25)

    def test_cloud_hls_decode_failure_propagates_without_wave_fallback(self):
        out = self.root / "failed.flac"
        with patch.object(runtime, "resample_window_to_wav", side_effect=RuntimeError("decode failed")), \
                patch("wave.open") as fallback:
            with self.assertRaisesRegex(RuntimeError, "Failed to extract HLS"):
                cloud_chunks.extract_chunk_audio(self.playlist, out, 5.25, 6.0, ffmpeg=self.ffmpeg)
        fallback.assert_not_called()
        self.assertFalse(out.exists())
        self.assertEqual(list(self.root.glob(".hls-chunk-*")), [])


if __name__ == "__main__":
    unittest.main()
