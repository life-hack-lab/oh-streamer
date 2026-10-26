"""extract_chunk_audio / resample_window_to_wav read a range-cache (.rcache) source through its runtime_url."""
from pathlib import Path
import array
import subprocess
import sys
import tempfile
import unittest
import wave

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from oh_streamer.sources import media_input, range_cache as RC
from oh_streamer.subtitles import cloud_chunks, runtime
from tests.test_range_cache import Upstream


def _rms(path) -> float:
    with wave.open(str(path), "rb") as w:
        samples = array.array("h", w.readframes(w.getnframes()))
    return (sum(s * s for s in samples) / max(1, len(samples))) ** 0.5


class RcacheMediaInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ffmpeg = runtime.resolve_ffmpeg()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"no ffmpeg: {exc}")
        cls.tmp = tempfile.TemporaryDirectory()
        src = Path(cls.tmp.name) / "src.mp4"
        # 10 s: a 440 Hz tone during the first 5 s, silence afterwards.
        subprocess.run([cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=10",
                        "-f", "lavfi", "-i", "aevalsrc=if(lt(t\\,5)\\,0.5*sin(2*PI*440*t)\\,0):s=16000:d=10",
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(src)],
                       check=True, capture_output=True)
        cls.data = src.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.up = Upstream(self.data)
        self.addCleanup(self.up.close)
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.session, _ = RC.prepare_session(Path(self.work.name) / "range-cache", url=self.up.url,
                                             total_bytes=len(self.data), request_key="k", block_size=16 * 1024)
        self.addCleanup(self.session.close)
        self.marker = str(self.session.playlist_path)

    def test_suffix_helper(self):
        self.assertTrue(media_input.needs_runtime_url(self.marker))
        self.assertTrue(media_input.needs_runtime_url("X:/a/source.RCACHE?x=1"))
        self.assertFalse(media_input.needs_runtime_url("a/index.m3u8"))
        self.assertFalse(media_input.needs_runtime_url("a/video.mp4"))
        self.assertEqual(media_input.ffmpeg_input("a/video.mp4"), ("a/video.mp4", []))

    def test_extract_chunk_audio_explicit_source_url(self):
        out = Path(self.work.name) / "loud.flac"
        cloud_chunks.extract_chunk_audio(self.marker, out, 2.0, 2.0, fmt="flac",
                                         source_url=self.session.runtime_url)
        self.assertGreater(out.stat().st_size, 0)
        wav = Path(self.work.name) / "loud.wav"
        subprocess.run([self.ffmpeg, "-loglevel", "error", "-y", "-i", str(out), str(wav)], check=True)
        with wave.open(str(wav), "rb") as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (16000, 1))
            self.assertAlmostEqual(w.getnframes() / 16000.0, 2.0, delta=0.15)
        self.assertGreater(_rms(wav), 3000)

    def test_extract_chunk_audio_resolves_live_session_and_seeks(self):
        quiet = Path(self.work.name) / "quiet.wav"
        cloud_chunks.extract_chunk_audio(self.marker, quiet, 6.0, 2.0, fmt="wav")   # no source_url: live session
        self.assertLess(_rms(quiet), 300)
        loud = Path(self.work.name) / "loud.wav"
        cloud_chunks.extract_chunk_audio(self.marker, loud, 3.0, 1.0, fmt="wav")
        self.assertGreater(_rms(loud), 3000)

    def test_resample_window_to_wav(self):
        out = Path(self.work.name) / "w.wav"
        runtime.resample_window_to_wav(self.marker, out, 4.0, 6.0, pad=False, source_url=self.session.runtime_url)
        self.assertAlmostEqual(runtime.wav_duration(out), 2.0, delta=0.15)

    def test_missing_session_raises(self):
        with self.assertRaises(RuntimeError):
            cloud_chunks.extract_chunk_audio(str(Path(self.work.name) / "nope" / "source.rcache"),
                                             Path(self.work.name) / "x.wav", 0, 1, fmt="wav")


if __name__ == "__main__":
    unittest.main()
