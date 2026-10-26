"""CPU-only tests of the engine interface (engines/base.py, engines/__init__.py get_engine) and of the Lada engine's
session logic against a fake FrameRestorer (no GPU, no model weights, no video). The GPU counterpart that drives real
restoration through the same API is test_engine_interface_gpu.py. Needs torch + lada importable (skipped otherwise).
"""
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

from oh_streamer import engines
from oh_streamer.engines import base

try:
    import torch
    from oh_streamer.engines.lada import engine as lada_engine   # first: it points LADA_MODEL_WEIGHTS_DIR at lada
    from lada.utils import threading_utils as TU
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - torch / lada not available in this environment
    torch = TU = lada_engine = None
    IMPORT_ERROR = e


class BaseModuleTests(unittest.TestCase):

    def test_unknown_engine_is_a_value_error(self):
        with self.assertRaises(ValueError):
            engines.get_engine("no-such-engine")

    def test_base_imports_neither_torch_nor_lada(self):
        import ast
        tree = ast.parse((_SOURCE_ROOT / "oh_streamer" / "engines" / "base.py").read_text(encoding="utf-8"))
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(names & {"torch", "lada", "numpy", "av"}, names)

    def test_restored_frame_is_slotted_and_cheap(self):
        f = base.RestoredFrame("img", 5)
        self.assertFalse(hasattr(f, "__dict__"))
        self.assertEqual((f.image, f.pts, f.original_patches), ("img", 5, None))

    def test_markers(self):
        self.assertEqual(base.EndOfStream().reason, "stopped")
        e = base.EngineError("x", ValueError("y"))
        self.assertIsInstance(e, Exception)
        self.assertIsInstance(e.cause, ValueError)

    def test_capability_constants_cover_all(self):
        self.assertEqual(len(base.ALL_CAPABILITIES), 6)


class _FakeRestorer:
    """Records the calls LadaSession makes and serves a scripted output queue."""

    def __init__(self, elems=()):
        self.calls = []
        self.eof = True
        self.stop_requested = False
        self.max_clip_length = 60
        self.out_q = TU.PipelineQueue(name="frame_restoration_queue", maxsize=8)
        self.mosaic_restoration_model = type("M", (), {"cancel_event": threading.Event()})()
        for e in elems:
            self.out_q.put(e)

    def get_frame_restoration_queue(self):
        return self.out_q

    def start(self, start_ns=0):
        self.calls.append(("start", start_ns, self.eof, self.out_q.qsize(), self.capture_original))

    def stop(self):
        self.calls.append(("stop", self.mosaic_restoration_model.cancel_event.is_set()))

    def __iter__(self):
        return self

    def __next__(self):      # FrameRestorer.__next__ in short
        elem = self.out_q.get()
        if elem is TU.EOF_MARKER:
            raise StopIteration
        return elem


def _session(restorer):
    s = lada_engine.LadaSession.__new__(lada_engine.LadaSession)
    s.source, s.max_clip_length, s._restorer, s._capture, s._running = None, 60, restorer, False, False
    s.source = type("S", (), {"src": "x.mp4", "http": None, "meta": type("Me", (), {})()})()
    return s


@unittest.skipIf(IMPORT_ERROR is not None, f"torch / lada not importable: {IMPORT_ERROR!r}")
class LadaEngineCpuTests(unittest.TestCase):

    def test_get_engine_lada_is_a_singleton_unless_new(self):
        a = engines.get_engine("lada")
        self.assertIs(engines.get_engine("lada"), a)
        self.assertIsNot(engines.get_engine("lada", new=True), a)
        self.assertIsNot(engines.get_engine("lada", new=True), engines.get_engine("lada", new=True))

    def test_info_and_catalog(self):
        e = lada_engine.LadaEngine()
        info = e.info()
        self.assertEqual(info.name, "lada")
        self.assertTrue(info.version)
        self.assertEqual(info.capabilities, base.ALL_CAPABILITIES)
        cat = e.list_models()
        self.assertIn("v4-fast", cat.detection)
        self.assertIn("basicvsrpp-v1.2", cat.restoration)

    def test_nothing_loaded(self):
        e = lada_engine.LadaEngine()
        self.assertIsNone(e.models)
        self.assertIsNone(e.device)
        e.release()      # idempotent
        with self.assertRaises(RuntimeError):
            e.open(base.SourceSpec("x.mp4", None))

    def test_load_checks_the_lada_version_without_blocking(self):
        from unittest import mock
        e = lada_engine.LadaEngine()
        with mock.patch.object(lada_engine.compat, "check") as check:
            with self.assertRaises(RuntimeError):
                e.load(torch.device("cpu"), base.ModelChoice("no-such", "basicvsrpp-v1.2", False), 60)
        check.assert_called_once_with(e.info().version)

    def test_lada_commit_prefers_the_one_in_lada_version(self):
        from unittest import mock
        with mock.patch.object(lada_engine, "_commit_cache", []), \
                mock.patch.object(lada_engine, "_LADA_VERSION", "0.11.1-dev+20cb34a"), \
                mock.patch.object(lada_engine.subprocess, "run", side_effect=AssertionError("no git needed")):
            self.assertEqual(lada_engine._lada_commit(), "20cb34a")

    def test_lada_commit_falls_back_to_git_without_a_commit_in_version(self):
        from unittest import mock
        done = mock.Mock(returncode=0, stdout="abc1234\n")
        with mock.patch.object(lada_engine, "_commit_cache", []), \
                mock.patch.object(lada_engine, "_LADA_VERSION", "0.11.1-dev"), \
                mock.patch.object(lada_engine.subprocess, "run", return_value=done) as run:
            self.assertEqual(lada_engine._lada_commit(), "abc1234")
            self.assertEqual(lada_engine._lada_commit(), "abc1234")
            run.assert_called_once()

    def test_unknown_model_name_fails_and_keeps_nothing(self):
        e = lada_engine.LadaEngine()
        with self.assertRaises(RuntimeError):
            e.load(torch.device("cpu"), base.ModelChoice("no-such", "basicvsrpp-v1.2", False), 60)
        self.assertIsNone(e.models)

    def test_load_same_models_is_a_no_op_but_applies_clip_length(self):
        e = lada_engine.LadaEngine()
        f = lada_engine.RestorerFactory(torch.device("cpu"), 60)
        f.det_model, f.res_model, f.det_name, f.res_name, f.fp16 = object(), object(), "d", "r", False
        e._factory = f
        e.load(torch.device("cpu"), base.ModelChoice("d", "r", False), 90)
        self.assertIs(e._factory, f)
        self.assertEqual((e.max_clip_length, f.max_clip_length), (90, 90))
        self.assertEqual(e.models, base.ModelChoice("d", "r", False))
        e.release()
        self.assertIsNone(e.models)

    def test_frames_maps_elements_without_copying(self):
        img = torch.zeros(2, 2, 3, dtype=torch.uint8)
        err = TU.ErrorMarker("boom", "trace")
        r = _FakeRestorer([(img, 7), TU.STOP_MARKER])
        out = list(_session(r).frames())
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], base.RestoredFrame)
        self.assertIs(out[0].image, img)
        self.assertEqual(out[0].pts, 7)
        self.assertIsNone(out[0].original_patches)
        self.assertIsInstance(out[1], base.EndOfStream)
        out = list(_session(_FakeRestorer([err])).frames())
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], base.EngineError)
        self.assertIn("boom", str(out[0]))
        self.assertIs(out[0].cause, err)

    def test_end_of_source_ends_the_iteration_silently(self):
        img = torch.zeros(1, 1, 3, dtype=torch.uint8)
        out = list(_session(_FakeRestorer([(img, 0), TU.EOF_MARKER])).frames())
        self.assertEqual([type(x) for x in out], [base.RestoredFrame])

    def test_original_patches_only_when_capturing(self):
        img = torch.zeros(1, 1, 3, dtype=torch.uint8)
        img._lada_orig = [((0, 0, 0, 0), "patch")]
        s = _session(_FakeRestorer([(img, 0), TU.EOF_MARKER]))
        self.assertIsNone(next(s.frames()).original_patches)
        s = _session(_FakeRestorer([(img, 0), TU.EOF_MARKER]))
        s._capture = True
        self.assertEqual(next(s.frames()).original_patches, [((0, 0, 0, 0), "patch")])

    def test_start_prepares_the_restorer_like_the_pipeline_did(self):
        r = _FakeRestorer([(1, 2)])      # a leftover element: start() must empty the queue and reset eof
        s = _session(r)
        s.start(12.5, capture_original=False)
        self.assertEqual(r.calls, [("start", 12_500_000_000, False, 0, False)])
        self.assertTrue(s._running)

    def test_stop_abandons_via_the_cancel_event_and_rearms_it(self):
        r = _FakeRestorer()
        s = _session(r)
        s.start(0.0)
        s.stop()
        self.assertEqual(r.calls[-1], ("stop", True))
        self.assertFalse(r.mosaic_restoration_model.cancel_event.is_set())
        s.stop(abandon=False)
        self.assertEqual(r.calls[-1], ("stop", False))
        self.assertFalse(s._running)

    def test_wake_unblocks_a_waiting_consumer_and_drain_empties(self):
        r = _FakeRestorer()
        s = _session(r)
        got = []
        done = threading.Event()

        def consumer():
            got.extend(s.frames())
            done.set()
        threading.Thread(target=consumer, daemon=True).start()
        time.sleep(0.2)
        self.assertFalse(done.is_set())
        s.wake()
        self.assertTrue(done.wait(3))
        self.assertEqual([type(x) for x in got], [base.EndOfStream])
        r.out_q.put((1, 1))
        s.drain()
        self.assertEqual(r.out_q.qsize(), 0)

    def test_close_stops_a_running_session_and_closed_calls_are_safe(self):
        r = _FakeRestorer()
        s = _session(r)
        s.start(0.0)
        s.close()
        self.assertEqual(r.calls[-1][0], "stop")
        s.close()
        s.stop()
        s.wake()
        s.drain()
        self.assertIsNone(s.attach_progress_probe())
        self.assertIsNone(s.attach_detect_gate())
        self.assertEqual(s.metrics(), base.PipelineMetrics())
        with self.assertRaises(RuntimeError):
            s.start(0.0)

    def test_metrics_without_the_lada_queues_still_reports_the_output(self):
        r = _FakeRestorer([(1, 1), (2, 2)])
        r.eof = False
        m = _session(r).metrics()      # the fake has no feeder / clip queues: no "queues" detail
        self.assertEqual((m.output_queue, m.output_capacity, m.eof, m.frames_detected, m.queues), (2, 8, False, 0, {}))


if __name__ == "__main__":
    unittest.main()
