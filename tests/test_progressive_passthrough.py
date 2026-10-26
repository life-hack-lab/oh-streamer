"""Focused tests for source passthrough and prepared progressive restore candidates."""
from pathlib import Path as _Path
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from unittest import mock

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_SOURCE_ROOT))

import oh_streamer.player.progressive_stream as P  # noqa: E402


class ProgressivePassthroughTest(unittest.TestCase):
    def setUp(self):
        self.temp_root = _STREAMER_ROOT / ".tmp-t12-feeder"
        self.temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.addCleanup(self.temp.cleanup)
        self.work = _Path(self.temp.name)
        self.seg_dir = self.work / "segments"
        self.seg_dir.mkdir()
        self.play = P.PlayInfo(1, 4, 0.0, count=5)
        self.info = {"duration": 20.0, "fps": Fraction(1, 1), "frames": 20,
                     "video_start": 0.0, "has_audio": False}
        self.logs = []

    def publish(self, keys, *, complete=False, frames=None, partial=()):
        segments = []
        for k in keys:
            name = f"seg_{k:06d}.mkv"
            (self.seg_dir / name).write_bytes(b"segment")
            item = {"k": k, "file": name, "frames": (frames or {}).get(k, 4)}
            if k in partial:
                item["partial"] = True
            segments.append(item)
        raw = {"version": 2, "segments": segments, "last_k": max(keys) if complete and keys else None,
               "complete": complete,
               "play": {"fps": "1/1", "n": 4, "video_start": 0, "rate_mult": 1,
                        "layers": ["processed"], "count": 5}}
        path = self.work / "segments.json"
        tmp = self.work / "segments.tmp"
        tmp.write_text(json.dumps(raw), encoding="utf-8")
        os.replace(tmp, path)

    def playback(self, *, path=None, source_url=None, replies=False, send=None):
        po = P.ProgressiveOpen(path or str(self.work / "source.mp4"), str(self.work), "q1", "Clip", self.info,
                               source_url=source_url)
        pb = P.ProgressivePlayback(po, send=send or self.logs.append, log=self.logs.append, replies=replies)
        self.addCleanup(pb.close)
        return pb

    def test_close_returns_while_candidate_source_read_is_still_blocked(self):
        pb = self.playback()
        candidate = P.RestoreCandidate(pb.watcher, "source.mp4", 2.0, False, str(self.work))
        entered, release = threading.Event(), threading.Event()

        def blocked_read():
            entered.set()
            release.wait(5)

        candidate._thread = threading.Thread(target=blocked_read, daemon=True)
        candidate._thread.start()
        self.assertTrue(entered.wait(1))
        pb.feeder = candidate
        try:
            started = time.perf_counter()
            pb.close()
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.5)
            self.assertTrue(candidate.alive, "the source read must remain blocked while close returns")
            self.assertIsNone(pb.feeder)
        finally:
            release.set()
            candidate._thread.join(2)
            candidate.retire()
            candidate._cleanup_thread.join(2)
        self.assertFalse(os.path.exists(candidate.url))

    def test_segment_available_requires_a_listed_file_covering_the_position(self):
        self.publish([0, 2])
        pb = self.playback()

        self.assertTrue(pb.segment_available(3.5))
        self.assertFalse(pb.segment_available(4.0))
        self.assertFalse(pb.segment_available(7.999))
        (self.seg_dir / "seg_000002.mkv").unlink()
        self.assertFalse(pb.segment_available(8.5))

    def test_restore_range_requires_contiguous_real_files_and_minimum_lead(self):
        self.publish([0, 1, 3])
        pb = self.playback()

        self.assertEqual(pb.restore_range(1.0, min_ahead=7.0), (0.0, 8.0))
        self.assertIsNone(pb.restore_range(1.0, min_ahead=8.0))
        self.assertIsNone(pb.restore_range(9.0, min_ahead=0.0))
        (self.seg_dir / "seg_000001.mkv").unlink()
        self.assertIsNone(pb.restore_range(5.0, min_ahead=0.0))

    def test_complete_tail_allows_less_than_minimum_lead_but_needs_one_frame(self):
        self.publish([0, 1, 2], complete=True, frames={2: 3})
        pb = self.playback()

        self.assertEqual(pb.restore_range(10.0, min_ahead=8.0), (8.0, 11.0))
        self.assertIsNone(pb.restore_range(10.99, min_ahead=8.0))

    def test_short_nonfinal_island_does_not_qualify_as_a_complete_tail(self):
        self.publish([0, 2, 4], complete=False)
        pb = self.playback()

        self.assertIsNone(pb.restore_range(9.0, min_ahead=8.0))

    def test_restore_range_does_not_use_partial_segments_as_handoff_coverage(self):
        self.publish([0, 1, 2], complete=True, partial=(1,))
        pb = self.playback()

        self.assertIsNone(pb.restore_range(1.0, min_ahead=8.0))

    def test_restore_range_uses_the_watcher_snapshot_without_forcing_a_manifest_read(self):
        self.publish([0, 1, 2])
        pb = self.playback()
        pb.watcher.close()

        with mock.patch.object(pb.watcher, "poll") as poll:
            self.assertEqual(pb.restore_range(1.0, min_ahead=8.0), (0.0, 12.0))

        poll.assert_not_called()

    def test_start_source_retires_candidate_updates_position_and_sends_one_foreground_want(self):
        self.publish([0, 1])
        sent = []
        pb = self.playback(send=sent.append)
        old = mock.Mock()
        pb.feeder = old

        url = pb.start_source(6.25)

        self.assertEqual(url, pb.po.path)
        old.retire.assert_called_once_with()
        self.assertEqual((pb.target, pb.last_position), (6.25, 6.25))
        self.assertEqual(sent, [{"cmd": "want", "id": "q1", "k": 1, "pos": 6.25, "foreground": True}])

    def test_start_source_rejects_rcache_marker_without_runtime_url(self):
        sent = []
        pb = self.playback(path=str(self.work / "source.rcache"), send=sent.append)
        old = mock.Mock()
        pb.feeder = old

        with self.assertRaisesRegex(ValueError, "缺少本機來源網址"):
            pb.start_source(2.0)

        old.retire.assert_not_called()
        self.assertEqual(sent, [])

    def test_prepare_restore_starts_a_candidate_without_want_or_position_change(self):
        sent, captured = [], []
        pb = self.playback(source_url="http://127.0.0.1:1234/media", send=sent.append)
        pb.target, pb.last_position = 2.0, 2.0

        class Candidate:
            def __init__(self, watcher, source, target, has_audio, work_dir, log):
                self.args = (source, target, has_audio, work_dir)
                self.waiting_k = 1
                self.error = RuntimeError("candidate failed")
                self.retired = False
                self.prepared = threading.Event()
                self.continuation = mock.Mock()
                captured.append(self)

            def start(self):
                return self

            def retire(self, keep_continuation=False):
                self.retired = True

            def close(self):
                self.retired = True

        with mock.patch.object(P, "RestoreCandidate", Candidate):
            feeder = pb.prepare_restore(5.0)
            pb.tick()

        self.assertIs(feeder, pb.feeder)
        self.assertEqual(captured[0].args, (pb.po.source_url, 5.0, False, pb.po.work_dir))
        self.assertEqual((pb.target, pb.last_position), (2.0, 2.0))
        self.assertEqual([p for p in sent if p.get("cmd") == "want"], [])
        self.assertNotEqual(pb.hold, "failed")

    def test_activate_restore_promotes_continuation_without_a_new_want(self):
        sent = []
        pb = self.playback(send=sent.append)

        class Candidate:
            continuation = mock.Mock()

        candidate = Candidate()
        pb.feeder = candidate
        pb._candidate_mode = True

        with mock.patch.object(P, "RestoreCandidate", Candidate):
            pb.activate_restore(5.0)

        self.assertIs(pb.feeder, candidate.continuation)
        self.assertFalse(pb._candidate_mode)
        self.assertEqual((pb.target, pb.last_position), (5.0, 5.0))
        self.assertEqual(sent, [])

    def test_superseded_want_reply_cannot_change_current_hold(self):
        replies, sent = [], []

        def send(payload, callback):
            sent.append(payload)
            replies.append(callback)

        pb = self.playback(replies=True, send=send)
        pb.want(0)
        old_reply = replies[-1]
        pb.want(1)
        current_reply = replies[-1]
        current_reply({"ok": True})
        old_reply({"ok": False, "error": "failed"})

        self.assertIsNone(pb.hold)

    def test_retire_is_nonblocking_and_closes_the_sink(self):
        watcher = mock.Mock()
        sink = mock.Mock()
        feeder = P.Feeder(watcher, "source.mp4", 0.0, sink=sink)
        feeder._thread = mock.Mock()
        feeder._thread.is_alive.return_value = True

        feeder.retire()

        self.assertTrue(feeder.stop_ev.is_set())
        sink.close.assert_called_once_with()
        watcher.wake.assert_called_once_with()
        feeder._thread.join.assert_not_called()

class RestoreCandidateIntegrationTest(unittest.TestCase):
    """The handoff clip is seekable at source time and contains B-frame video tracks and copied source audio."""

    @classmethod
    def setUpClass(cls):
        temp_root = _STREAMER_ROOT / ".tmp-t12-feeder"
        temp_root.mkdir(exist_ok=True)
        cls.root = tempfile.mkdtemp(prefix="restore-candidate-", dir=temp_root)
        cls.source = os.path.join(cls.root, "source.mp4")
        cls.segment_store = os.path.join(cls.root, "segments")
        os.makedirs(cls.segment_store)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-filter_complex", "[0:v]split=3[a][b][c];[b]hue=s=0[b2];[c]negate[c2]",
            "-map", "[a]", "-map", "[b2]", "-map", "[c2]", "-map", "1:a",
            "-t", "8", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-bf", "2", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-force_key_frames", "expr:gte(t,n_forced*1)", "-c:a", "aac", "-b:a", "64k",
            "-metadata:s:v:0", "title=oh:processed", "-metadata:s:v:1", "title=oh:original",
            "-metadata:s:v:2", "title=oh:restored", cls.source], check=True)
        for k, start in enumerate((0, 4)):
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", cls.source,
                "-t", "4", "-map", "0:v", "-an", "-c:v", "copy", "-avoid_negative_ts", "disabled",
                "-metadata:s:v:0", "title=oh:processed", "-metadata:s:v:1", "title=oh:original",
                "-metadata:s:v:2", "title=oh:restored",
                os.path.join(cls.segment_store, f"seg_{k:06d}.mkv")], check=True)
        cls.info = P.probe_source(cls.source)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="work-", dir=self.root)
        self.seg_dir = os.path.join(self.work, "segments")
        os.makedirs(self.seg_dir)
        for name in os.listdir(self.segment_store):
            shutil.copy2(os.path.join(self.segment_store, name), os.path.join(self.seg_dir, name))
        raw = {
            "version": 2,
            "segments": [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": 120} for k in range(2)],
            "last_k": 1,
            "complete": True,
            "total_frames": 240,
            "play": {"fps": "30/1", "n": 120, "video_start": 0, "rate_mult": 1,
                     "layers": ["processed", "original", "restored"], "count": 2, "compare": True},
        }
        with open(os.path.join(self.work, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f)
        self.logs = []
        self.watcher = P.ManifestWatcher(self.work, P.play_from_probe(self.info), interval=0.1,
                                         log=self.logs.append).start()
        self.addCleanup(self.watcher.close)

    def decoded_frame_at_or_after(self, path, position, stream_index=0):
        import av
        with av.open(path) as container:
            stream = container.streams.video[stream_index]
            container.seek(P._seek_ts(stream, position), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None and float(frame.pts * frame.time_base) >= position - 1e-6:
                    return float(frame.pts * frame.time_base), frame.to_ndarray(format="rgb24")
        self.fail(f"no decoded frame at or after {position}s in {path}")

    def test_clip_is_seekable_at_source_time_with_bframes_all_video_tracks_and_audio(self):
        import av
        target = 2.37
        candidate = P.RestoreCandidate(self.watcher, self.source, target, True, self.work, self.logs.append).start()
        self.addCleanup(candidate.retire)

        self.assertTrue(candidate.prepared.wait(15), self.logs)
        self.assertIsNone(candidate.error, self.logs)
        self.assertTrue(os.path.isfile(candidate.url))
        self.assertAlmostEqual(candidate.first_s, 2.0, delta=0.08)
        self.assertAlmostEqual(candidate.end_s, 4.0, delta=0.002)
        self.assertIsNotNone(candidate.continuation)
        self.assertAlmostEqual(candidate.continuation.target, 4.0, delta=0.002)
        with av.open(candidate.url) as clip:
            videos = list(clip.streams.video)
            self.assertEqual(len(videos), 3)
            self.assertEqual(len(clip.streams.audio), 1)
            self.assertEqual([s.metadata.get("title") for s in videos],
                             ["oh:processed", "oh:original", "oh:restored"])
            video_packets = list(clip.demux(videos[0]))
            self.assertTrue(any(p.pts is not None and p.dts is not None and p.pts != p.dts for p in video_packets),
                            "the output should retain B-frame packet ordering")
            clip.seek(P._seek_ts(videos[0], target), stream=videos[0], backward=True)
            decoded = next((f for f in clip.decode(videos[0])
                            if f.pts is not None and float(f.pts * f.time_base) >= target - 1e-6), None)
            self.assertIsNotNone(decoded, "the Matroska cue index should seek to a decodable frame near target")
            clip_time = float(decoded.pts * decoded.time_base)
        source_time, source_frame = self.decoded_frame_at_or_after(self.source, target)
        clip_time, clip_frame = self.decoded_frame_at_or_after(candidate.url, target)
        self.assertAlmostEqual(clip_time, source_time, delta=0.002)
        self.assertEqual(clip_frame.tobytes(), source_frame.tobytes(),
                         "the frame at the source timestamp should match after seeking the restored clip")

    def test_candidate_source_open_uses_a_finite_read_timeout(self):
        candidate = P.RestoreCandidate(self.watcher, self.source, 2.0, True, self.work)
        self.assertEqual(candidate._source_timeout, (P.RESTORE_SOURCE_TIMEOUT, P.RESTORE_SOURCE_TIMEOUT))
        candidate.retire()

    def test_initial_bframe_keyframe_is_not_dropped_when_demuxer_has_no_dts_yet(self):
        candidate = P.RestoreCandidate(self.watcher, self.source, 0.37, True, self.work, self.logs.append).start()
        self.addCleanup(candidate.retire)
        self.assertTrue(candidate.prepared.wait(15), self.logs)
        self.assertIsNone(candidate.error, self.logs)
        self.assertAlmostEqual(candidate.first_s, 0.0, delta=0.002)
        source_time, source_frame = self.decoded_frame_at_or_after(self.source, 0.37)
        clip_time, clip_frame = self.decoded_frame_at_or_after(candidate.url, 0.37)
        self.assertAlmostEqual(clip_time, source_time, delta=0.002)
        self.assertEqual(clip_frame.tobytes(), source_frame.tobytes())
        self.assertTrue(candidate.continuation.ready.wait(5), self.logs)
        self.assertAlmostEqual(candidate.continuation.first_s, candidate.end_s, delta=0.002)

    def test_clip_and_continuation_own_disjoint_audio_packets_without_a_gap(self):
        import av
        from tests.test_progressive_stream import StreamReader

        candidate = P.RestoreCandidate(self.watcher, self.source, 2.37, True, self.work, self.logs.append).start()
        self.addCleanup(candidate.retire)
        self.assertTrue(candidate.prepared.wait(15), self.logs)
        self.assertIsNone(candidate.error, self.logs)
        live = candidate.continuation
        self.assertIsNotNone(live)
        self.assertTrue(live.ready.wait(5), self.logs)
        output = os.path.join(self.work, "continuation.mkv")
        reader = StreamReader(live.sink.port, output)
        self.addCleanup(reader.sock.close)
        self.assertTrue(reader.join(10), self.logs)
        self.assertIsNone(live.error, self.logs)

        def packets(path):
            with av.open(path) as container:
                return [(float(packet.pts * packet.time_base), float((packet.duration or 0) * packet.time_base))
                        for packet in container.demux(container.streams.audio[0]) if packet.pts is not None]

        before, after = packets(candidate.url), packets(output)
        self.assertTrue(before and after)
        boundary = candidate.end_s
        self.assertTrue(all(pts < boundary for pts, _ in before), "audio prefetch must stop at the clip boundary")
        self.assertTrue(all(pts >= boundary - 1e-6 for pts, _ in after), "the clip owns the crossing packet")
        previous_end = before[-1][0] + before[-1][1]
        self.assertAlmostEqual(previous_end, after[0][0], delta=0.002,
                               msg="copied audio must neither repeat a packet nor leave a packet-sized hole")


if __name__ == "__main__":
    unittest.main(verbosity=2)
