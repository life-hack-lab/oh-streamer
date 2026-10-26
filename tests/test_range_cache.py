"""Range-cache tests; every upstream is an in-process loopback HTTP server."""
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import json
import os
from unittest import mock
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from oh_streamer.sources import range_cache as RC
from oh_streamer.queue.queue_worker import source_status_error

BS = 1000


def make_data(n: int) -> bytes:
    return bytes((i * 31 + (i >> 8) * 7) & 0xFF for i in range(n))


class Upstream:
    """Loopback file server with Range support, optional delay, token expiry and no-Range mode."""

    def __init__(self, data: bytes, *, delay: float = 0.0, range_supported: bool = True):
        self.data = data
        self.delay = delay
        self.range_supported = range_supported
        self.token = "a"
        self.expire_after: int | None = None     # flip token after N range requests
        self.log: list[int] = []                 # start offsets of served ranges
        self.requests = 0
        self.active = 0                          # concurrent range requests right now
        self.max_active = 0
        self.slow_offsets: dict[int, float] = {}  # start offset -> extra delay (simulates a slow upstream block)
        self.started: list[tuple[float, int]] = []   # (monotonic time, start offset) when a range request began
        self.lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_GET(self):
                outer.requests += 1
                if outer.expire_after is not None and outer.requests > outer.expire_after:
                    outer.token = "b"
                    outer.expire_after = None
                m = re.search(r"t=(\w+)", self.path)
                if (m.group(1) if m else "a") != outer.token:
                    self.send_response(403)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if outer.delay:
                    time.sleep(outer.delay)
                rng = self.headers.get("Range")
                total = len(outer.data)
                if not rng or not outer.range_supported:
                    self.send_response(200)
                    self.send_header("Content-Length", str(total))
                    self.end_headers()
                    self.wfile.write(outer.data)
                    return
                a, b = re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups()
                a, b = int(a), min(int(b), total - 1)
                outer.log.append(a)
                with outer.lock:
                    outer.started.append((time.monotonic(), a))
                    outer.active += 1
                    outer.max_active = max(outer.max_active, outer.active)
                try:
                    time.sleep(outer.slow_offsets.get(a, 0.0))
                finally:
                    with outer.lock:
                        outer.active -= 1
                body = outer.data[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{total}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/f.mp4"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def http_get(url, rng=None, method="GET"):
    req = urllib.request.Request(url, method=method, headers={"Range": rng} if rng else {})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class RangeCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "cache"
        self._closers = []

    def tearDown(self):
        for fn in reversed(self._closers):
            try:
                fn()
            except Exception:
                pass
        RC.close_session(self.dir / RC.MARKER_NAME)
        try:
            self._tmp.cleanup()
        except OSError:
            pass

    def upstream(self, data, **kw):
        up = Upstream(data, **kw)
        self._closers.append(up.close)
        return up

    def prepare(self, up, data, **kw):
        kw.setdefault("request_key", "page|fmt")
        kw.setdefault("block_size", BS)
        kw.setdefault("retry_delay", 0.01)
        return RC.prepare_session(self.dir, url=up.url, total_bytes=len(data), **kw)

    def test_sequential_download_completes_and_serves(self):
        data = make_data(4500)
        up = self.upstream(data)
        s, ready = self.prepare(up, data, metadata={"format": {"duration": "1"}})
        self.assertEqual(s.wait(10), "complete")
        self.assertEqual(up.log, [0, 1000, 2000, 3000, 4000])
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)
        self.assertTrue(Path(s.playlist_path).is_file())
        snap = s.snapshot()
        self.assertEqual((snap["segment_count"], len(snap["ready_segments"])), (5, 5))
        self.assertEqual((snap["total_bytes"], snap["block_size"], snap["downloaded_bytes"]), (4500, BS, 4500))
        status = json.loads(Path(s.status_path).read_text(encoding="utf-8"))
        self.assertIsNone(source_status_error(status, s.identity, s.generation))
        self.assertTrue(s.runtime_url.endswith("/source.mp4"))
        code, h, body = http_get(s.runtime_url)
        self.assertEqual((code, body, h["Accept-Ranges"]), (200, data, "bytes"))
        code, h, body = http_get(s.runtime_url, "bytes=900-2100")
        self.assertEqual((code, body, h["Content-Range"]), (206, data[900:2101], "bytes 900-2100/4500"))
        code, h, body = http_get(s.runtime_url, "bytes=-100")
        self.assertEqual((code, body), (206, data[-100:]))
        code, h, _ = http_get(s.runtime_url, "bytes=9999-")
        self.assertEqual((code, h["Content-Range"]), (416, "bytes */4500"))
        code, h, body = http_get(s.runtime_url, method="HEAD")
        self.assertEqual((code, h["Content-Length"], body), (200, "4500", b""))
        self.assertEqual(http_get(s.runtime_url.replace("source.mp4", "x.mp4"))[0], 404)

    def test_sidecar_gate_rejects_while_downloading(self):
        data = make_data(3000)
        up = self.upstream(data, delay=0.3)
        s, _ = self.prepare(up, data)
        status = json.loads(Path(s.status_path).read_text(encoding="utf-8"))
        self.assertIn("downloading", source_status_error(status, s.identity, s.generation))
        s.close()

    def test_priority_jump_to_tail_then_middle(self):
        data = make_data(20 * BS)
        up = self.upstream(data, delay=0.1)
        s, _ = self.prepare(up, data)
        res = {}
        t = threading.Thread(target=lambda: res.update(tail=http_get(s.runtime_url, "bytes=-200")))
        t.start()
        t.join(10)
        self.assertEqual(res["tail"][2], data[-200:])
        res2 = http_get(s.runtime_url, f"bytes={10 * BS + 5}-{10 * BS + 50}")
        self.assertEqual(res2[2], data[10 * BS + 5:10 * BS + 51])
        order = [x // BS for x in up.log]
        self.assertLessEqual(order.index(19), 2)
        self.assertLessEqual(order.index(10), 6)
        self.assertEqual(s.wait(30), "complete")
        order = [x // BS for x in up.log]
        self.assertEqual(order[order.index(10) + 1], 11)       # continues after the jump target
        self.assertEqual(sorted(order), list(range(20)))        # holes filled, nothing fetched twice
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)

    def test_priority_request_does_not_wait_for_current_sequential_block(self):
        data = make_data(20 * BS)
        up = self.upstream(data)
        up.slow_offsets = {0: 2.0}                       # sequential connection is stuck on block 0 for 2 s
        s, _ = self.prepare(up, data)
        time.sleep(0.2)
        t0 = time.monotonic()
        code, _, body = http_get(s.runtime_url, f"bytes={15 * BS + 3}-{15 * BS + 40}")
        waited = time.monotonic() - t0
        self.assertEqual((code, body), (206, data[15 * BS + 3:15 * BS + 41]))
        self.assertLess(waited, 1.0)
        self.assertNotIn(14 * BS, up.log)                # only the block the reader needed, no read-ahead
        self.assertEqual(s.wait(30), "complete")
        self.assertEqual(sorted(x // BS for x in up.log), list(range(20)))      # nothing fetched twice
        self.assertLessEqual(up.max_active, 2)
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)

    def test_single_connection_mode_has_one_connection_and_waits(self):
        data = make_data(12 * BS)
        up = self.upstream(data)
        up.slow_offsets = {0: 1.0}
        s, _ = self.prepare(up, data, connections=1)
        time.sleep(0.2)
        self.assertIsNone(s._prio_fetcher)
        t0 = time.monotonic()
        code, _, body = http_get(s.runtime_url, f"bytes={9 * BS}-{9 * BS + 9}")
        self.assertEqual((code, body), (206, data[9 * BS:9 * BS + 10]))
        self.assertGreater(time.monotonic() - t0, 0.7)   # had to wait for the in-flight block 0
        self.assertEqual(s.wait(30), "complete")
        self.assertEqual(up.max_active, 1)
        self.assertEqual(sorted(x // BS for x in up.log), list(range(12)))

    def test_many_priority_readers_never_duplicate_fetches(self):
        data = make_data(30 * BS)
        up = self.upstream(data)
        up.slow_offsets = {i * BS: 0.05 for i in range(30)}
        s, _ = self.prepare(up, data)
        results = {}

        def reader(i):
            results[i] = http_get(s.runtime_url, f"bytes={i * BS}-{i * BS + 4}")[2]

        threads = [threading.Thread(target=reader, args=(i,)) for i in (29, 20, 11, 11, 5, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
        self.assertEqual(s.wait(30), "complete")
        for i, got in results.items():
            self.assertEqual(got, data[i * BS:i * BS + 5])
        self.assertEqual(sorted(x // BS for x in up.log), list(range(30)))
        self.assertLessEqual(up.max_active, 2)

    def test_expired_url_with_two_connections_renews_once_per_expiry(self):
        data = make_data(16 * BS)
        up = self.upstream(data)
        up.slow_offsets = {i * BS: 0.03 for i in range(16)}
        up.expire_after = 4
        calls = []

        def renew(code, old_url):
            calls.append(code)
            return up.url + "?t=b", None

        s, _ = self.prepare(up, data, on_expired=renew)
        threading.Thread(target=lambda: http_get(s.runtime_url, f"bytes={14 * BS}-{14 * BS + 3}"), daemon=True).start()
        self.assertEqual(s.wait(20), "complete")
        self.assertLessEqual(len(calls), 2)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)

    def test_cancel_stops_both_connections_quickly(self):
        data = make_data(30 * BS)
        up = self.upstream(data)
        up.slow_offsets = {i * BS: 0.3 for i in range(30)}
        s, _ = self.prepare(up, data)
        threading.Thread(target=lambda: http_get(s.runtime_url, f"bytes={25 * BS}-{25 * BS + 3}"), daemon=True).start()
        time.sleep(0.4)
        t0 = time.time()
        s.cancel()
        s._fetcher.join(3)
        s._prio_fetcher.join(3)
        self.assertFalse(s._fetcher.is_alive() or s._prio_fetcher.is_alive())
        self.assertLess(time.time() - t0, 2)
        self.assertEqual(s.status, "paused")

    def test_resume_old_block_size_sidecar_keeps_its_block_size(self):
        data = make_data(4 * 4096)
        up = self.upstream(data, delay=0.02)
        s, _ = self.prepare(up, data, block_size=4096)
        deadline = time.time() + 10
        while len(s.snapshot()["ready_segments"]) < 2 and time.time() < deadline:
            time.sleep(0.01)
        s.cancel()
        RC.close_session(s.playlist_path)
        up.log.clear()
        s2, ready = RC.prepare_session(self.dir, url=up.url, total_bytes=len(data), request_key="page|fmt",
                                       retry_delay=0.01)                    # block_size not given: adopt the sidecar's
        self.assertEqual(s2.block_size, 4096)
        self.assertEqual(s2.wait(15), "complete")
        self.assertTrue(set(range(4)) >= {x // 4096 for x in up.log})
        self.assertFalse({x // 4096 for x in up.log} & set(ready))           # verified blocks were not refetched
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)
        RC.close_session(s2.playlist_path)
        # a fresh cache without an explicit size uses the default
        other = Path(self._tmp.name) / "fresh"
        s3, _ = RC.prepare_session(other, url=up.url, total_bytes=len(data), request_key="k")
        self.assertEqual(s3.block_size, RC.DEFAULT_BLOCK_SIZE)
        RC.close_session(s3.playlist_path)

    def test_resume_fetches_only_missing_blocks(self):
        data = make_data(10 * BS)
        up = self.upstream(data, delay=0.05)
        got = []
        s, _ = self.prepare(up, data, progress=lambda d, t, r, n: got.append((d, t, r, n)))
        deadline = time.time() + 10
        while len(s.snapshot()["ready_segments"]) < 4 and time.time() < deadline:
            time.sleep(0.01)
        s.cancel()
        self.assertEqual(s.status, "paused")
        s.close()
        have = {x["index"] for x in s.snapshot()["ready_segments"]}
        self.assertTrue(0 < len(have) < 10)
        self.assertEqual(got[-1][1:], (10 * BS, len(have), 10))
        up.log.clear()
        s2, ready = self.prepare(up, data)
        self.assertEqual(set(ready), have)
        self.assertEqual(s2.wait(15), "complete")
        self.assertEqual(sorted(x // BS for x in up.log), sorted(set(range(10)) - have))
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)

    def test_resume_drops_corrupt_block_and_refetches_it(self):
        data = make_data(4 * BS)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        RC.close_session(s.playlist_path)
        with (self.dir / RC.DATA_NAME).open("r+b") as f:
            f.seek(2 * BS + 3)
            f.write(b"\xff\xfe")
        self.assertIsNone(RC.open_session(self.dir / RC.MARKER_NAME))
        up.log.clear()
        s2, ready = self.prepare(up, data)
        self.assertEqual(sorted(ready), [0, 1, 3])
        self.assertEqual(s2.wait(10), "complete")
        self.assertEqual(up.log, [2 * BS])

    def test_tampered_block_is_never_served(self):
        data = make_data(3 * BS)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        with (self.dir / RC.DATA_NAME).open("r+b") as f:
            f.seek(BS + 10)
            f.write(b"EVIL")
        code, _, body = http_get(s.runtime_url, f"bytes={BS}-{BS + 99}")
        self.assertEqual(code, 503)
        self.assertNotIn(b"EVIL", body)
        status = json.loads(Path(s.status_path).read_text(encoding="utf-8"))
        self.assertIsNotNone(source_status_error(status, s.identity, s.generation))
        self.assertEqual(http_get(s.runtime_url, "bytes=0-9")[2], data[:10])   # intact blocks still fine

    def test_open_session_reopens_verified_complete_cache(self):
        data = make_data(2500)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        RC.close_session(s.playlist_path)
        again = RC.open_session(self.dir / RC.MARKER_NAME)
        self.assertIsNotNone(again)
        self.assertEqual((again.identity, again.generation), (s.identity, s.generation))
        self.assertEqual(http_get(again.runtime_url, "bytes=2000-2499")[2], data[2000:])
        self.assertIs(RC.get_session(self.dir / RC.MARKER_NAME), again)

    def test_reopen_unchanged_complete_cache_skips_hashing(self):
        data = make_data(4 * BS)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        RC.close_session(s.playlist_path)
        sidecar = json.loads((self.dir / RC.SIDE_CAR_NAME).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["data_stat"]["size"], 4 * BS)
        with mock.patch.object(RC, "_verified_ready", side_effect=AssertionError("must not hash the file")):
            again = RC.open_session(self.dir / RC.MARKER_NAME)
        self.assertIsNotNone(again)
        self.assertEqual(http_get(again.runtime_url, f"bytes={BS}-{BS + 9}")[2], data[BS:BS + 10])

    def test_lazy_reopen_still_verifies_each_block_when_served(self):
        data = make_data(3 * BS)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        RC.close_session(s.playlist_path)
        data_file = self.dir / RC.DATA_NAME
        before = data_file.stat()
        with data_file.open("r+b") as f:
            f.seek(BS + 5)
            f.write(b"EVIL")
        os.utime(data_file, ns=(before.st_atime_ns, before.st_mtime_ns))     # same size and mtime: lazily trusted
        again = RC.open_session(self.dir / RC.MARKER_NAME)
        self.assertIsNotNone(again)
        code, _, body = http_get(again.runtime_url, f"bytes={BS}-{BS + 99}")
        self.assertEqual(code, 503)
        self.assertNotIn(b"EVIL", body)
        self.assertEqual(again.status, "failed")

    def test_changed_size_or_mtime_or_missing_stat_forces_full_verification(self):
        data = make_data(3 * BS)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        RC.close_session(s.playlist_path)
        data_file = self.dir / RC.DATA_NAME
        st = data_file.stat()
        os.utime(data_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))      # touched: must hash
        calls = []
        real = RC._verified_ready
        with mock.patch.object(RC, "_verified_ready", side_effect=lambda *a: (calls.append(1), real(*a))[1]):
            again = RC.open_session(self.dir / RC.MARKER_NAME)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(again)
        RC.close_session(again.playlist_path)
        sidecar_file = self.dir / RC.SIDE_CAR_NAME
        doc = json.loads(sidecar_file.read_text(encoding="utf-8"))
        doc.pop("data_stat")                                    # a sidecar written before lazy verification existed
        sidecar_file.write_text(json.dumps(doc), encoding="utf-8")
        with mock.patch.object(RC, "_verified_ready", side_effect=lambda *a: (calls.append(2), real(*a))[1]):
            self.assertIsNotNone(RC.open_session(self.dir / RC.MARKER_NAME))
        self.assertEqual(calls, [1, 2])

    def test_expired_url_is_renewed_via_callback(self):
        data = make_data(8 * BS)
        up = self.upstream(data)
        up.expire_after = 3
        calls = []

        def renew(code, old_url):
            calls.append((code, old_url))
            return up.url + "?t=b", {"X-New": "1"}

        s, _ = self.prepare(up, data, on_expired=renew)
        self.assertEqual(s.wait(15), "complete")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 403)
        self.assertEqual((self.dir / RC.DATA_NAME).read_bytes(), data)

    def test_expired_url_without_renewal_fails(self):
        data = make_data(4 * BS)
        up = self.upstream(data)
        up.expire_after = 1
        s, _ = self.prepare(up, data, on_expired=lambda c, u: None)
        self.assertEqual(s.wait(10), "failed")
        self.assertIn("expired", s.snapshot()["error"])
        self.assertEqual(http_get(s.runtime_url, f"bytes={3 * BS}-")[0], 503)

    def test_set_upstream_replaces_url(self):
        data = make_data(3 * BS)
        up = self.upstream(data)
        up.token = "z"
        s, _ = self.prepare(up, data, on_expired=lambda c, u: (up.url + "?t=z", None))
        self.assertEqual(s.wait(10), "complete")
        s.set_upstream("http://example.invalid/x", {"A": "b"})
        self.assertEqual(s._url, "http://example.invalid/x")

    def test_probe_range_support(self):
        data = make_data(12345)
        up = self.upstream(data)
        info = RC.probe_range_support(up.url)
        self.assertEqual(info["total_bytes"], 12345)
        up2 = self.upstream(data, range_supported=False)
        self.assertIsNone(RC.probe_range_support(up2.url))
        self.assertIsNone(RC.probe_range_support("http://127.0.0.1:1/none"))

    def test_server_without_range_fails_session(self):
        data = make_data(3 * BS)
        up = self.upstream(data, range_supported=False)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "failed")
        self.assertIn("Range", s.snapshot()["error"])

    def test_identity_change_refuses_mixing(self):
        data = make_data(2 * BS)
        up = self.upstream(data)
        s, _ = self.prepare(up, data)
        self.assertEqual(s.wait(10), "complete")
        RC.close_session(s.playlist_path)
        with self.assertRaises(RC.RangeCacheError):
            self.prepare(up, data, request_key="other")

    def test_cancel_stops_fetcher_quickly(self):
        data = make_data(30 * BS)
        up = self.upstream(data, delay=0.2)
        s, _ = self.prepare(up, data)
        time.sleep(0.3)
        t0 = time.time()
        s.cancel()
        s._fetcher.join(3)
        self.assertFalse(s._fetcher.is_alive())
        self.assertLess(time.time() - t0, 2)
        self.assertEqual(s.status, "paused")
        status = json.loads(Path(s.status_path).read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "paused")


if __name__ == "__main__":
    unittest.main()
