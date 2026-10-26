"""HLS demand scheduling and real loopback downloader requests (no GPU)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from oh_streamer.sources import hls_cache as HC
from oh_streamer.queue import queue_download as QD


def playlist(count=8, duration=600):
    return ("#EXTM3U\n" + "".join(f"#EXTINF:{duration},\ns{i}.ts\n" for i in range(count))
            + "#EXT-X-ENDLIST\n").encode()


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        manifest = HC.parse_media_playlist(playlist(), "http://localhost/source.m3u8")
        self.session, _ = HC.prepare_session(manifest, Path(self.tmp.name), metadata={}, request_key="seek")
        self.addCleanup(HC.close_session, self.session.playlist_path)

    def commit(self, index):
        stage = Path(self.tmp.name) / "stage.tmp"
        stage.write_bytes(b"G" * 188)
        HC.commit_segment(self.session, index, stage)

    def test_seek_continues_forward_then_fills_holes_without_duplicates(self):
        self.commit(0)
        self.session.request_position(1800)
        order = []
        while (index := self.session.next_download_segment()) is not None:
            order.append(index)
            self.commit(index)
        self.assertEqual(order, [3, 4, 5, 6, 7, 1, 2])
        HC.complete_session(self.session)
        self.assertEqual(self.session.status, "complete")

    def test_variable_durations_and_invalid_positions(self):
        self.session._source_descriptor["segments"][0]["duration"] = 1.25
        for pos in (-1, float("nan"), float("inf"), 999999):
            self.session.request_position(pos)
        self.assertFalse(self.session._wanted)
        self.session.request_position(601.25)
        self.assertEqual(self.session.next_download_segment(), 2)

    def test_multiple_requests_deduplicate_and_skip_resumed_segments(self):
        self.commit(0)
        self.commit(4)
        for index in (4, 6, 6, -1, 99, 2):
            self.session.request_segment(index)
        self.assertEqual(self.session.next_download_segment(), 6)
        self.commit(6)
        self.assertEqual(self.session.next_download_segment(), 2)
        self.commit(2)
        self.assertEqual(self.session.next_download_segment(), 3)
        self.session.cancel()
        self.assertIsNone(self.session.next_download_segment())
        self.assertIsNone(self.session._await_segment(99))


class DownloaderSeekTest(unittest.TestCase):
    def test_http_reader_jumps_thirty_minutes_before_prefix_is_downloaded(self):
        requests, errors, handles = [], [], []
        first_started, release_first = threading.Event(), threading.Event()
        at_four, release_four = threading.Event(), threading.Event()
        cancel = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/source.m3u8":
                    body = playlist()
                else:
                    index = int(self.path[2:-3])
                    requests.append(index)
                    if index == 1:
                        first_started.set()
                        release_first.wait(5)
                    if index == 4:
                        at_four.set()
                        release_four.wait(5)
                    body = b"G" * 188
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with tempfile.TemporaryDirectory() as tmp:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            url = f"http://127.0.0.1:{server.server_port}/source.m3u8"
            probe = {"format": {"format_name": "mpegts"}, "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"}]}

            def download():
                try:
                    QD._download_hls(url, {}, {"ext": "ts"}, Path(tmp), "test", "best",
                                     lambda _: None, cancel, handles.append)
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=download)
            try:
                with patch.object(QD, "_probe_ts", return_value=probe):
                    worker.start()
                    self.assertTrue(first_started.wait(3))
                    session = handles[0]
                    # Match the local proxy request made by a media reader seeking to 30 min.
                    output = []

                    def read():
                        try:
                            with urlopen(session.runtime_url.rsplit("/", 1)[0] + "/segment-000003.ts", timeout=5) as r:
                                output.append(r.read())
                        except Exception as exc:
                            errors.append(exc)

                    reader = threading.Thread(target=read)
                    reader.start()
                    with session._condition:
                        self.assertTrue(session._condition.wait_for(lambda: 3 in session._wanted, timeout=3))
                    release_first.set()
                    self.assertTrue(at_four.wait(3))
                    reader.join(3)
                    self.assertEqual(output, [b"G" * 188])
                    self.assertEqual(requests, [0, 1, 3, 4])
                    self.assertFalse((Path(tmp) / "hls-cache/segment-000002.ts").exists())
                    self.assertEqual(session.status, "downloading")
                    release_four.set()
                    worker.join(5)
                    self.assertFalse(worker.is_alive())
                    self.assertFalse(errors)
                    self.assertEqual(requests, [0, 1, 3, 4, 5, 6, 7, 2])
                    self.assertEqual(Counter(requests), Counter(range(8)))
                    self.assertEqual(session.status, "complete")
            finally:
                release_first.set()
                release_four.set()
                cancel.set()
                worker.join(6)
                for session in handles:
                    HC.close_session(session.playlist_path)
                server.shutdown()
                server.server_close()
                server_thread.join(2)


if __name__ == "__main__":
    unittest.main()
