"""Passthrough engine specifics (no GPU, no weights): the frames equal the source, the missing capabilities degrade
instead of crashing the upper layers, and the engine registry (--engine / OH_STREAMER_ENGINE). The shared contract runs in
test_engine_interface_gpu.py (ENGINE_NAMES).
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

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from oh_streamer import engines
from oh_streamer.engines import base
from tests.engine_contract.test_engine_interface_gpu import CLIP, PLAIN_S, pull, require_engine, spec_for


class RegistryTests(unittest.TestCase):

    def setUp(self):
        self._saved = list(engines._selected)
        self.addCleanup(lambda: engines._selected.__setitem__(slice(None), self._saved))
        engines._selected.clear()

    def test_default_is_lada_and_env_selects(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OH_STREAMER_ENGINE", None)
            self.assertEqual(engines.default_engine_name(), "lada")
            os.environ["OH_STREAMER_ENGINE"] = "passthrough"
            self.assertEqual(engines.default_engine_name(), "passthrough")
            os.environ["OH_STREAMER_ENGINE"] = "nope"
            with self.assertRaises(ValueError):
                engines.default_engine_name()

    def test_set_default_engine_beats_env_and_rejects_unknown(self):
        with mock.patch.dict(os.environ, {"OH_STREAMER_ENGINE": "lada"}):
            engines.set_default_engine("passthrough")
            self.assertEqual(engines.default_engine_name(), "passthrough")
        with self.assertRaises(ValueError):
            engines.set_default_engine("nope")


class PassthroughTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        require_engine("passthrough")
        cls.engine = engines.get_engine("passthrough", new=True)
        cls.engine.load(None, base.ModelChoice("passthrough", "passthrough", False), 180)
        cls.spec = spec_for(CLIP)

    def test_info_and_singleton(self):
        info = self.engine.info()
        self.assertEqual((info.name, info.capabilities), ("passthrough", frozenset({base.FAST_CANCEL, base.OWN_READER})))
        a = engines.get_engine("passthrough")
        self.assertIs(engines.get_engine("passthrough"), a)
        self.assertIsNot(engines.get_engine("passthrough", new=True), a)

    def test_missing_capabilities_degrade(self):
        s = self.engine.open(self.spec)
        self.addCleanup(s.close)
        self.assertEqual(s.capabilities, self.engine.info().capabilities)
        self.assertIsNone(s.attach_progress_probe())
        self.assertIsNone(s.attach_detect_gate())
        self.assertFalse(s.observe_detections(lambda n: None))
        s.set_lookahead_limit(100)
        s.set_lookahead_limit(None)
        m = s.metrics()
        self.assertEqual((m.frames_detected, m.clips_pending, m.clips_restoring, m.queues), (0, 0, 0, {}))
        self.assertGreater(m.output_capacity, 0)

    def test_frames_equal_the_source(self):
        import av
        import numpy as np
        s = self.engine.open(self.spec)
        self.addCleanup(s.close)
        s.start(0.0)
        try:
            frames, marker = pull(s, 5)
        finally:
            s.stop()
            s.wake()
            s.drain()
        self.assertIsNone(marker)
        with av.open(str(CLIP)) as c:
            for f, ref in zip(frames, c.decode(video=0)):
                self.assertEqual(ref.pts, f.pts)
                self.assertTrue(np.array_equal(f.image.numpy(), ref.to_ndarray(format="bgr24")))
                self.assertIsNone(f.original_patches)

    def test_end_of_source_sets_eof_and_a_restart_yields_again(self):
        s = self.engine.open(self.spec)
        self.addCleanup(s.close)
        tail = self.spec.duration - 0.5
        for _ in range(2):
            s.start(tail)
            frames, marker = pull(s, 1000, timeout=30)
            self.assertIsNone(marker)
            self.assertGreater(len(frames), 3)
            self.assertTrue(s.eof)
            s.stop()
            s.drain()

    def test_compare_is_refused_for_an_engine_without_original_capture(self):
        from oh_streamer.player import pipeline as P
        s = self.engine.open(self.spec)
        self.addCleanup(s.close)
        source = SimpleNamespace(src=self.spec.url, audio_src=self.spec.url, has_audio=False, fmt_start=0.0, http=None,
                                 meta=s.source.meta)
        deps = P.PipelineDependencies(
            log=lambda _m: None, start_muxer=None, stop_process=None, free_port=None, is_listening=None,
            torch=SimpleNamespace(Tensor=type("Tensor", (), {}), cuda=SimpleNamespace(is_available=lambda: False)),
            open_reader=lambda src: self.engine.open_reader(src))
        sess = P.Session(s, source, "libx264", None, deps=deps)
        sess.compare_mode = "on"
        with mock.patch.object(P.Session, "_pump", lambda self: None):
            sess.start(0.0)
            sess.thread.join()
        self.assertEqual(sess.layers, ("processed",))
        self.assertTrue(sess.cmp_why)

    def test_open_reader_uses_the_registered_source(self):
        s = self.engine.open(self.spec)
        self.addCleanup(s.close)
        with self.engine.open_reader(self.spec.url) as r:
            r.seek(int(PLAIN_S * 1e9))
            pts = next(iter(r.frames()))[1]
        self.assertLess(abs(float(pts * self.spec.time_base) - PLAIN_S), 1.0 / float(self.spec.fps_exact))


if __name__ == "__main__":
    unittest.main()
