"""Real shared-source overlap with local HTTP and FFmpeg, without a GPU/model."""
import collections
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oh_streamer.queue import queue_download
from oh_streamer.sources import hls_cache


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RealHLSOverlap(unittest.TestCase):
    def test_closed_output_before_download_complete_and_one_fetch_per_segment(self):
        with tempfile.TemporaryDirectory(prefix="hls_overlap_") as temporary:
            root = Path(temporary)
            fixture = root / "upstream"
            fixture.mkdir()
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-t", "10", "-c:v", "libx264", "-preset", "ultrafast",
                "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-f", "hls",
                "-hls_time", "1", "-hls_list_size", "0", "-hls_playlist_type", "vod",
                str(fixture / "source.m3u8"),
            ], check=True, timeout=20, capture_output=True)
            release, blocked, ready = threading.Event(), threading.Event(), threading.Event()
            counts, results = collections.Counter(), {}

            class Handler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(fixture), **kwargs)

                def do_GET(self):
                    name = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
                    counts[name] += 1
                    if name == "source7.ts":
                        blocked.set()
                        release.wait(25)
                    super().do_GET()

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            url = f"http://127.0.0.1:{server.server_address[1]}/source.m3u8"

            def source_ready(handle):
                results["handle"] = handle
                ready.set()

            def download():
                try:
                    results["download"] = queue_download.download(
                        url, {}, None, str(root / "cache"), lambda progress: None,
                        threading.Event(), on_source_ready=source_ready,
                    )
                except BaseException as exc:
                    results["error"] = repr(exc)

            downloader = threading.Thread(target=download, daemon=True)
            downloader.start()
            process = None
            try:
                self.assertTrue(ready.wait(15), results)
                self.assertTrue(blocked.wait(15), results)
                handle = results["handle"]
                self.assertEqual(handle.snapshot()["status"], "downloading")
                self.assertTrue(downloader.is_alive())
                segments = root / "segments"
                segments.mkdir()
                # Encode acts as a processing consumer; queue scheduling and Lada's
                # completion rules are tested separately by the worker/service suite.
                with (root / "encode.log").open("w") as encode_log:
                    process = subprocess.Popen([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", handle.runtime_url, "-map", "0:v:0", "-an",
                        "-c:v", "libx264", "-preset", "ultrafast", "-g", "30",
                        "-sc_threshold", "0", "-bf", "0", "-f", "segment",
                        "-segment_time", "1", "-reset_timestamps", "1",
                        str(segments / "seg_%03d.mkv"),
                    ], stdout=subprocess.DEVNULL, stderr=encode_log)
                    deadline = time.monotonic() + 12
                    while time.monotonic() < deadline and not (segments / "seg_001.mkv").exists():
                        if process.poll() is not None:
                            self.fail((root / "encode.log").read_text())
                        time.sleep(.05)
                    self.assertTrue((segments / "seg_001.mkv").exists(),
                                    "first segment must close while tail download is blocked")
                    self.assertTrue(downloader.is_alive())
                    prefix_probe = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "json", str(segments / "seg_000.mkv"),
                    ], capture_output=True, text=True, check=True, timeout=10)
                    self.assertGreater(float(json.loads(prefix_probe.stdout)["format"]["duration"]), .9)
                    release.set()
                    downloader.join(15)
                    self.assertFalse(downloader.is_alive())
                    self.assertNotIn("error", results)
                    self.assertEqual(process.wait(timeout=15), 0)
                self.assertEqual(handle.snapshot()["status"], "complete")
                final = root / "source-copy.mkv"
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", handle.runtime_url, "-map", "0", "-c", "copy", str(final),
                ], check=True, capture_output=True, timeout=15)
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-show_streams", "-show_format",
                    "-of", "json", str(final),
                ], capture_output=True, text=True, check=True, timeout=10)
                info = json.loads(probe.stdout)
                self.assertEqual({s["codec_type"] for s in info["streams"]}, {"audio", "video"})
                self.assertAlmostEqual(float(info["format"]["duration"]), 10, delta=.1)
                self.assertEqual([counts[f"source{i}.ts"] for i in range(10)], [1] * 10)
                identity = handle.identity
                hls_cache.close_session(handle.playlist_path)
                reopened = hls_cache.open_session(handle.playlist_path)
                self.assertIsNotNone(reopened)
                self.assertEqual(reopened.identity, identity)
            finally:
                release.set()
                downloader.join(35)
                if process and process.poll() is None:
                    process.kill()
                    process.wait()
                if "handle" in results:
                    hls_cache.close_session(results["handle"].playlist_path)
                server.shutdown()
                server.server_close()
                server_thread.join(2)


if __name__ == "__main__":
    unittest.main()
