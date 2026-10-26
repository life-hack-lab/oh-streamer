"""CPU-only contract tests of the lada integration points (no GPU, no model weights, no video): the hooks and markers
that oh_player re-exports and the probes that wrap a FrameRestorer's queues. Part of the characterization suite for
the engine decoupling; they need torch and the
lada library to be importable and skip otherwise."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[2]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import threading
import time
import unittest
from fractions import Fraction
from types import SimpleNamespace

try:
    import torch
    from tests.engine_contract.lada_names import namespace
    LS = namespace()
    from lada.utils import threading_utils as lada_threading_utils
    from lada.utils import video_utils as lada_video_utils
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - torch / lada not available in this environment
    LS = torch = None
    IMPORT_ERROR = e

from oh_streamer.player import buffer_progress  # noqa: E402
from oh_streamer.processing import compare_stack, detect_gate  # noqa: E402
from oh_streamer.sources import source_resolve  # noqa: E402


def canned_probe(width=1280, height=720, fps="24/1", duration="300.0", audio=True, nb_frames=None, start_pts=512):
    video = {"codec_type": "video", "codec_name": "h264", "width": width, "height": height, "r_frame_rate": fps,
             "avg_frame_rate": fps, "duration": duration, "time_base": "1/12288", "start_pts": start_pts}
    if nb_frames is not None:
        video["nb_frames"] = str(nb_frames)
    streams = [video] + ([{"codec_type": "audio", "codec_name": "aac"}] if audio else [])
    return {"streams": streams, "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": duration,
                                          "start_time": "0.000000"}}


@unittest.skipIf(IMPORT_ERROR is not None, f"torch / lada not importable: {IMPORT_ERROR!r}")
class MetadataContract(unittest.TestCase):

    def test_build_metadata_from_canned_probe(self):
        m = LS.build_metadata("clip.mp4", canned_probe())
        self.assertEqual((m.video_file, m.video_width, m.video_height), ("clip.mp4", 1280, 720))
        self.assertEqual(m.video_fps_exact, Fraction(24, 1))
        self.assertEqual(m.video_fps, 24.0)
        self.assertEqual(m.codec_name, "h264")
        self.assertEqual(m.duration, 300.0)
        self.assertEqual(m.frames_count, 7200)                 # no nb_frames: duration * fps
        self.assertEqual(m.time_base, Fraction(1, 12288))
        self.assertEqual(m.start_pts, 512)

    def test_build_metadata_prefers_nb_frames_and_fractional_fps(self):
        m = LS.build_metadata("x", canned_probe(fps="30000/1001", duration="10.0", nb_frames=300))
        self.assertEqual(m.video_fps_exact, Fraction(30000, 1001))
        self.assertAlmostEqual(m.video_fps, 29.97, places=2)
        self.assertEqual(m.frames_count, 300)

    def test_source_from_probe_without_ffprobe(self):
        s = LS.Source("clip.mp4", {"X-A": "1"}, probe=canned_probe())
        self.assertTrue(s.has_audio)
        self.assertIs(s.meta, s.native_meta)
        self.assertEqual(s.headers, {"X-A": "1"})
        self.assertEqual(s.audio_src, "clip.mp4")
        self.assertEqual(s.fmt_start, 0.0)
        self.assertIn("1280x720", s.describe())
        self.assertFalse(LS.Source("clip.mp4", {}, probe=canned_probe(audio=False)).has_audio)

    def test_source_rejects_non_video(self):
        for probe in ({"streams": [], "format": {"format_name": "mov"}},
                      {"streams": [dict(canned_probe()["streams"][0], codec_name="ansi")], "format": {"format_name": "tty"}},
                      {"streams": [dict(canned_probe()["streams"][0])], "format": {"format_name": "image2"}}):
            with self.assertRaises(LS.open_input.OpenInputError):
                LS.Source("x", {}, probe=probe)


@unittest.skipIf(IMPORT_ERROR is not None, f"torch / lada not importable: {IMPORT_ERROR!r}")
class IoHooksContract(unittest.TestCase):

    def setUp(self):
        self._sources = dict(LS._IO_SOURCES)

    def tearDown(self):
        LS._IO_SOURCES.clear()
        LS._IO_SOURCES.update(self._sources)

    def test_install_patches_video_utils_and_is_idempotent(self):
        LS.install_lada_io_hooks()
        reader, get_meta = lada_video_utils.VideoReader, lada_video_utils.get_video_meta_data
        self.assertTrue(getattr(reader, "_lada_stream_hook", False))
        self.assertTrue(issubclass(reader, LS._LADA_VIDEO_READER))
        self.assertIsNot(get_meta, LS._LADA_GET_META)
        LS.install_lada_io_hooks()
        self.assertIs(lada_video_utils.VideoReader, reader)
        self.assertIs(lada_video_utils.get_video_meta_data, get_meta)

    def test_register_source_serves_metadata_and_forget_drops_it(self):
        src = LS.Source("contract-clip.mp4", {"X-A": "1"}, probe=canned_probe())
        LS.register_source(src)
        self.assertIs(lada_video_utils.get_video_meta_data("contract-clip.mp4"), src.meta)
        meta, http = LS._IO_SOURCES["contract-clip.mp4"]
        self.assertIs(http, src.http)
        LS.forget_sources(keep=())
        self.assertNotIn("contract-clip.mp4", LS._IO_SOURCES)

    def test_forget_sources_keeps_the_given_ones(self):
        a = LS.Source("a.mp4", {}, probe=canned_probe())
        b = LS.Source("b.mp4", {}, probe=canned_probe())
        LS.register_source(a)
        LS.register_source(b)
        LS.forget_sources(keep=(a,))
        self.assertIn("a.mp4", LS._IO_SOURCES)
        self.assertNotIn("b.mp4", LS._IO_SOURCES)

    def test_install_with_metadata_registers_it(self):
        m = LS.build_metadata("legacy.mp4", canned_probe())
        LS.install_lada_io_hooks(m, source_resolve.FFmpegHttpOptions({}))
        self.assertIs(lada_video_utils.get_video_meta_data("legacy.mp4"), m)

    def test_reader_passes_http_options_only_for_urls(self):
        LS.install_lada_io_hooks()
        src = LS.Source("http://127.0.0.1:1/x.mp4", {"User-Agent": "ua-test", "X-A": "1"}, probe=canned_probe())
        LS.register_source(src)
        opened = {}

        def fake_open(file, **kw):
            opened.update(file=file, **kw)
            return object()
        real_open = LS.av.open
        LS.av.open = fake_open
        try:
            r = lada_video_utils.VideoReader.__new__(lada_video_utils.VideoReader)
            r.file = "http://127.0.0.1:1/x.mp4"
            r.__enter__()
        finally:
            LS.av.open = real_open
        self.assertEqual(opened["options"]["user_agent"], "ua-test")
        self.assertIn("X-A: 1", opened["options"]["headers"])


class _DummyModel:
    """Stands in for a restoration model wrapper: restore(video) runs a torch module that holds sub-modules."""

    def __init__(self):
        inner = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        self.net = torch.nn.Sequential(inner, torch.nn.Linear(2, 2))
        self.flag = "shared"
        self.calls = 0

    def restore(self, video, *a, **kw):
        self.calls += 1
        self.net(torch.zeros(1, 2))
        return ["restored"] * len(video)


@unittest.skipIf(IMPORT_ERROR is not None, f"torch / lada not importable: {IMPORT_ERROR!r}")
class CancelHooksContract(unittest.TestCase):

    def test_install_counts_non_leaf_modules_only(self):
        m = _DummyModel()
        self.assertEqual(LS.install_cancel_hooks(m), 2)       # net and its inner Sequential; the leaves get none
        self.assertEqual(len(m.net._forward_pre_hooks), 1)
        self.assertEqual(len(m.net[1]._forward_pre_hooks), 0)

    def test_proxy_is_an_instance_and_shares_state(self):
        m = _DummyModel()
        p = LS.cancellable_restoration_model(m)
        self.assertIsInstance(p, _DummyModel)
        self.assertIsNot(type(p), _DummyModel)
        self.assertEqual(type(p).__name__, "Cancellable_DummyModel")
        self.assertIs(p.net, m.net)
        self.assertIsInstance(p.cancel_event, threading.Event)
        self.assertIsNot(p.cancel_event, LS.cancellable_restoration_model(m).cancel_event)   # one per restorer

    def test_restore_without_cancel_runs_normally(self):
        m = _DummyModel()
        LS.install_cancel_hooks(m)
        p = LS.cancellable_restoration_model(m)
        self.assertEqual(p.restore([1, 2, 3]), ["restored"] * 3)

    def test_cancel_returns_the_frames_unchanged(self):
        m = _DummyModel()
        LS.install_cancel_hooks(m)
        p = LS.cancellable_restoration_model(m)
        p.cancel_event.set()
        frames = [object(), object()]
        out = p.restore(frames)
        self.assertEqual(out, frames)
        self.assertEqual(m.calls, 0)
        self.assertIsNone(getattr(LS._cancel_tls, "event", None))   # thread-local cleared again

    def test_other_proxies_and_threads_are_not_affected(self):
        m = _DummyModel()
        LS.install_cancel_hooks(m)
        a, b = LS.cancellable_restoration_model(m), LS.cancellable_restoration_model(m)
        a.cancel_event.set()
        self.assertEqual(b.restore([1]), ["restored"])
        res = []
        th = threading.Thread(target=lambda: res.append(m.restore([1])))      # the raw model: no event in this thread
        th.start()
        th.join()
        self.assertEqual(res, [["restored"]])


@unittest.skipIf(IMPORT_ERROR is not None, f"torch / lada not importable: {IMPORT_ERROR!r}")
class MarkerContract(unittest.TestCase):

    def test_markers_are_lada_threading_utils(self):
        self.assertIs(LS.STOP_MARKER, lada_threading_utils.STOP_MARKER)
        self.assertIs(LS.ErrorMarker, lada_threading_utils.ErrorMarker)
        self.assertIs(LS.threading_utils, lada_threading_utils)
        self.assertTrue(issubclass(LS.ErrorMarker, Exception))

    def test_stop_marker_wakes_a_blocked_consumer(self):
        q = LS.threading_utils.PipelineQueue(name="contract")
        got = []
        th = threading.Thread(target=lambda: got.append(q.get()), daemon=True)
        th.start()
        time.sleep(0.1)
        self.assertTrue(th.is_alive())
        LS.threading_utils.put_queue_stop_marker(q)
        th.join(2)
        self.assertFalse(th.is_alive())
        self.assertIs(got[0], LS.STOP_MARKER)

    def test_empty_out_queue(self):
        q = LS.threading_utils.PipelineQueue(name="contract")
        for i in range(3):
            q.put(i)
        LS.threading_utils.empty_out_queue(q)
        self.assertTrue(q.empty())


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item, *a, **kw):
        self.items.append(item)

    def get(self, *a, **kw):
        return self.items.pop(0)


class FakeRestorer:
    def __init__(self):
        self.frame_detection_queue = FakeQueue()
        self.mosaic_clip_queue = FakeQueue()
        self.restored_clip_queue = FakeQueue()
        self.frame_restoration_queue = FakeQueue()
        self.max_clip_length = 60
        self.stop_requested = False


class OriginalCaptureContract(unittest.TestCase):

    def make_cls(self):
        class Dummy:
            capture_original = False

            def _restore_frame(self, frame, frame_num, restored_clips):
                return frame, frame_num
        return Dummy

    def test_install_is_idempotent(self):
        cls = self.make_cls()
        self.assertTrue(compare_stack.install_original_capture(cls))
        patched = cls._restore_frame
        self.assertFalse(compare_stack.install_original_capture(cls))
        self.assertIs(cls._restore_frame, patched)
        self.assertTrue(cls._lada_compare_hook)

    def test_capture_only_when_flag_is_set_and_clip_starts_on_the_frame(self):
        import numpy as np
        cls = self.make_cls()
        compare_stack.install_original_capture(cls)
        r = cls()
        clip = SimpleNamespace(frame_start=5, boxes=[(1, 1, 2, 3)])
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3).view(np.ndarray)

        class Frame(np.ndarray):
            pass
        f = frame.view(Frame)
        r.capture_original = False
        r._restore_frame(f, 5, [clip])
        self.assertFalse(hasattr(f, "_lada_orig"))
        r.capture_original = True
        r._restore_frame(f, 4, [clip])                          # the clip does not start on this frame
        self.assertFalse(hasattr(f, "_lada_orig"))
        out = r._restore_frame(f, 5, [clip])
        self.assertEqual(out, (f, 5))                           # the original method still runs
        (box, patch), = f._lada_orig
        self.assertEqual(box, (1, 1, 2, 3))
        self.assertEqual(patch.shape, (2, 3, 3))
        self.assertTrue((patch == frame[1:3, 1:4]).all())
        self.assertIsNot(patch.base, frame)                     # a copy

    @unittest.skipIf(IMPORT_ERROR is not None, "torch not importable")
    def test_torch_frames_take_the_attribute(self):
        t = torch.zeros(4, 6, 3, dtype=torch.uint8)
        compare_stack.capture_boxes(t, [(0, 0, 1, 1)])
        self.assertEqual(tuple(t._lada_orig[0][1].shape), (2, 2, 3))


class ProbeContract(unittest.TestCase):

    def test_attach_is_idempotent_and_none_without_queues(self):
        self.assertIsNone(buffer_progress.PipelineProbe.attach(SimpleNamespace()))
        self.assertIsNone(detect_gate.DetectGate.attach(SimpleNamespace()))
        r = FakeRestorer()
        p = buffer_progress.PipelineProbe.attach(r)
        self.assertIs(buffer_progress.PipelineProbe.attach(r), p)
        g = detect_gate.DetectGate.attach(r)
        self.assertIs(detect_gate.DetectGate.attach(r), g)

    def test_probe_and_gate_chain_on_the_same_queues(self):
        r = FakeRestorer()
        p = buffer_progress.PipelineProbe.attach(r)
        g = detect_gate.DetectGate.attach(r)
        p.reset(0.0)
        g.reset(100)
        r.frame_detection_queue.put((10, 1))                       # (frame number, number of mosaics)
        r.frame_detection_queue.put((11, 0))
        r.frame_restoration_queue.put(("frame", 0))
        self.assertEqual(len(r.frame_detection_queue.items), 2)    # the originals still receive everything
        snap = p.snapshot()
        self.assertEqual((snap["base"], snap["det"], snap["mosaic"]), (10, 2, [0]))
        self.assertEqual((g.det_n, g.out_n), (2, 1))
        g.open("stopped")
        self.assertTrue(g.is_open)
        self.assertIn("stopped", g.summary())
        p.freeze()
        r.frame_detection_queue.put((12, 1))
        self.assertEqual(p.snapshot()["det"], 2)                    # frozen: ignored


class ReExportContract(unittest.TestCase):

    @unittest.skipIf(IMPORT_ERROR is not None, f"torch / lada not importable: {IMPORT_ERROR!r}")
    def test_names_other_modules_use_stay_importable_from_oh_player(self):
        for name in ("RestorerFactory", "Source", "install_lada_io_hooks", "install_cancel_hooks",
                     "cancellable_restoration_model", "STOP_MARKER", "ErrorMarker", "register_source",
                     "forget_sources", "build_metadata", "ffprobe", "gpu_has_fp16_acceleration", "device_backend",
                     "threading_utils", "FrameRestorer", "buffer_progress", "detect_gate", "compare_stack"):
            self.assertTrue(hasattr(LS, name), name)


if __name__ == "__main__":
    unittest.main()
