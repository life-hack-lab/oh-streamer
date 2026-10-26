"""Which engine made a result: manifest / output tags, the queue's "same settings" keys, per-engine perf profiles.

Everything written before the engine fields existed counts as lada with an unknown version, and its keys must not change
(no existing queue item may turn into "needs reprocess"). No GPU, no weights, temp dirs only."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_STREAMER_ROOT, _STREAMER_ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import os
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from oh_streamer import engines
from oh_streamer.perf import profile as P
from oh_streamer.player import queue_bridge as L
from oh_streamer.processing import compare_stack as CS
from oh_streamer.queue import perf_runner
from oh_streamer.queue import queue_store
from oh_streamer.queue import queue_worker as Q
from tests.test_engine_setting import EngineStateTestCase, register_fake
from tests.test_queue_service import ServiceTestBase
from tests.test_perf_profile import fixture

# cache_key() values of the code before the engine setting existed (computed with that code).
LEGACY_SETTINGS = {"detection_model": "v4-fast", "restoration_model": "basicvsrpp-v1.2", "fp16": True,
                   "max_clip_length": 180, "encoder": "hevc", "interpolation": "full", "compare": "on",
                   "compare_original": "copy", "compare_quality": "small"}
LEGACY_KEY_FULL = "2e96db2bb9fd5614443ad1e8cc397c5f86d6d1f0"
LEGACY_KEY_MIN = "81d796fae9cbf8ea267cde167d31c9f03f9097b4"
MIN_KEYS = ("detection_model", "restoration_model", "fp16", "max_clip_length")


class CacheKeyTests(unittest.TestCase):
    def test_items_without_engine_keep_their_key(self):
        self.assertEqual(queue_store.cache_key("https://example.com/v.m3u8", "best", LEGACY_SETTINGS), LEGACY_KEY_FULL)
        self.assertEqual(queue_store.cache_key("C:/videos/a.mp4", None, {k: LEGACY_SETTINGS[k] for k in MIN_KEYS}),
                         LEGACY_KEY_MIN)

    def test_lada_engine_is_the_same_as_none(self):
        for engine in ("lada", "", None):
            self.assertEqual(queue_store.cache_key("https://example.com/v.m3u8", "best",
                                                   {**LEGACY_SETTINGS, "engine": engine}), LEGACY_KEY_FULL)

    def test_other_engine_is_a_different_item(self):
        other = queue_store.cache_key("https://example.com/v.m3u8", "best", {**LEGACY_SETTINGS, "engine": "passthrough"})
        self.assertNotEqual(other, LEGACY_KEY_FULL)

    def test_old_item_in_a_loaded_store_keeps_state_and_key(self):
        """A queue.json written before the engine setting: the item stays done and a lada re-add dedupes onto it."""
        tmp = Path(tempfile.mkdtemp(prefix="engrec-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        store = queue_store.QueueStore(tmp)
        settings = {k: LEGACY_SETTINGS[k] for k in MIN_KEYS}
        item = store.add("C:/videos/a.mp4", "file", {}, None, settings)
        self.assertEqual(item.cache_key, LEGACY_KEY_MIN)
        store.update(item.id, state="done", output_path=str(tmp / "out.mkv"))
        store.flush() if hasattr(store, "flush") else None
        again = queue_store.QueueStore(tmp)
        got = again.get(item.id)
        self.assertEqual((got.state, got.cache_key), ("done", LEGACY_KEY_MIN))
        self.assertNotIn("engine", got.settings)
        # the same video queued with the lada engine now is the same item; with another engine it is a new one
        self.assertEqual(again.add("C:/videos/a.mp4", "file", {}, None, dict(settings, engine="lada")).id, item.id)
        self.assertNotEqual(again.add("C:/videos/a.mp4", "file", {}, None, dict(settings, engine="passthrough")).id, item.id)


class WorkerRecordTests(EngineStateTestCase):
    INFO = Q.SourceInfo(Fraction(30), 300, 128, 96, 0.0, 10.0, True)

    def job(self, **extra):
        return {"source": "nonexistent.mp4", "start_k": 0,
                "settings": {"detection_model": "d", "restoration_model": "r", "fp16": None, "max_clip_length": 30,
                             "interpolation": "off", "compare": "off", **extra}}

    def key(self, **extra):
        plan = Q.SegmentPlan(self.INFO.fps, self.INFO.frames, 0, 1.0)
        return Q.job_key(self.job(**extra), self.INFO, plan, "libx264")

    def test_lada_job_key_is_the_legacy_key(self):
        legacy = self.key()
        self.assertNotIn("engine", legacy)
        self.assertEqual(self.key(engine="lada"), legacy)
        self.assertEqual(self.key(engine="passthrough")["engine"], "passthrough")

    def test_old_manifest_resumes_and_reads_as_lada(self):
        tmp = Path(tempfile.mkdtemp(prefix="engrec-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        seg = tmp / "segments"
        seg.mkdir()
        (seg / "seg_000000.mkv").write_bytes(b"x")
        key = self.key(engine="lada")
        raw = {"version": Q.MANIFEST_VERSION, "key": self.key(), "n": 30,
               "segments": [{"k": 0, "file": "seg_000000.mkv", "frames": 30}],
               "last_k": None, "complete": False, "total_frames": None, "play": None}
        (tmp / "segments.json").write_text(json.dumps(raw), encoding="utf-8")
        m = Q.Manifest.load(str(tmp / "segments.json"), key, str(seg))
        self.assertEqual(m.next_k, 1)                              # the old segments are reused, not invalidated
        self.assertEqual(Q.manifest_engine(raw), ("lada", None))
        self.assertNotIn("engine", m.doc())                        # nothing stamped until a run stamps it

    def test_other_engine_does_not_reuse_lada_segments(self):
        tmp = Path(tempfile.mkdtemp(prefix="engrec-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        seg = tmp / "segments"
        seg.mkdir()
        (seg / "seg_000000.mkv").write_bytes(b"x")
        m = Q.Manifest(str(tmp / "segments.json"), self.key())
        m.add(0, 30)
        self.assertEqual(Q.Manifest.load(str(tmp / "segments.json"), self.key(engine="passthrough"), str(seg)).next_k, 0)

    def test_manifest_stamps_engine_and_version(self):
        tmp = Path(tempfile.mkdtemp(prefix="engrec-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        m = Q.Manifest(str(tmp / "segments.json"), self.key())
        m.engine, m.engine_version = "lada", "0.11.1-dev+20cb34a"
        m.add(0, 30)
        saved = json.loads((tmp / "segments.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["engine"], saved["engine_version"]), ("lada", "0.11.1-dev+20cb34a"))
        self.assertEqual(Q.manifest_engine(saved), ("lada", "0.11.1-dev+20cb34a"))
        self.assertEqual(saved["key"], self.key())                 # the key (resume identity) is untouched

    def test_engine_stamp_uses_the_engine_info_and_survives_failures(self):
        register_fake(self)
        info = mock.Mock(version="9.9")
        engines.unregister_engine("fake")
        engines.register_engine("fake", lambda: mock.Mock(info=lambda: info), user_visible=False)
        self.assertEqual(Q.engine_stamp("fake"), ("fake", "9.9"))
        engines.unregister_engine("fake")
        engines.register_engine("fake", lambda: (_ for _ in ()).throw(RuntimeError("no")), user_visible=False)
        self.assertEqual(Q.engine_stamp("fake"), ("fake", None))
        engines.unregister_engine("fake")


class TagTests(unittest.TestCase):
    FULL = {"OH_SOURCE_FPS": "30000/1001", "OH_RATE_MULT": "2", "OH_VIDEO_START": "0.033367",
            "OH_COMPARE": "1", "OH_ORIGINAL": "copy"}

    def test_existing_tags_unchanged_and_engine_added(self):
        old = CS.lada_tags(Fraction(30000, 1001), 2, 0.0333667, True, "copy")
        self.assertEqual(set(old), {"OH_SOURCE_FPS", "OH_RATE_MULT", "OH_VIDEO_START", "OH_COMPARE", "OH_ORIGINAL"})
        new = CS.lada_tags(Fraction(30000, 1001), 2, 0.0333667, True, "copy", "lada", "0.11.1-dev+20cb34a")
        self.assertEqual({k: v for k, v in new.items() if k in old}, old)
        self.assertEqual((new["OH_ENGINE"], new["OH_ENGINE_VERSION"]), ("lada", "0.11.1-dev+20cb34a"))
        self.assertNotIn("OH_ENGINE_VERSION", CS.lada_tags(Fraction(24), 1, 0, False, None, "lada"))

    def test_finalize_tags_check_accepts_the_engine_tags(self):
        tags = CS.lada_tags(Fraction(30), 1, 0.0, False, None, "lada", "1.0")
        probe = {"format": {"tags": {k.lower(): v for k, v in tags.items()}},
                 "streams": [{"codec_type": "video", "tags": {}}]}
        self.assertIsNone(Q.check_tracks(probe, ("processed",), tags))

    def test_reader_old_file_is_lada_without_version(self):
        probe = {"streams": [], "format": {"tags": dict(self.FULL)}}
        tags = L.lada_tags_from_probe(probe)
        self.assertEqual((tags["engine"], tags["engine_version"]), ("lada", None))
        self.assertEqual(L.ProcessedOpen("a.mkv", tags=tags).engine, "lada")
        self.assertEqual(L.ProcessedOpen("a.mkv").engine, "lada")     # no tags at all (older than the tags)

    def test_reader_new_file(self):
        probe = {"streams": [], "format": {"tags": {**self.FULL, "OH_ENGINE": "passthrough",
                                                    "OH_ENGINE_VERSION": "2"}}}
        tags = L.lada_tags_from_probe(probe)
        self.assertEqual((tags["engine"], tags["engine_version"]), ("passthrough", "2"))
        self.assertEqual(L.ProcessedOpen("a.mkv", tags=tags).engine, "passthrough")

    def test_done_item_of_another_engine_still_plays(self):
        tmp = Path(tempfile.mkdtemp(prefix="engrec-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        out = tmp / "o.mkv"
        out.write_bytes(b"x")
        link = mock.Mock()
        link.request.return_value = {"ok": True, "item": {"id": "7", "state": "done", "output": str(out),
                                                        "engine": "passthrough"}}
        probe = lambda path: ("", "", {"engine": "passthrough", "engine_version": None})   # noqa: E731
        po = L.queue_lookup(link, "C:/v.mp4", None, probe_layers=probe)
        self.assertIsNotNone(po)
        self.assertEqual((po.queue_id, po.engine), ("7", "passthrough"))


class PerfPerEngineTests(EngineStateTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp(prefix="engrec-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = self.tmp / "perf_profile.json"

    def profile(self, engine=None):
        p = fixture()
        p["fingerprint"]["engine"] = engine
        return p

    def test_paths(self):
        self.assertEqual(P.path_for_engine(self.base, None), self.base)
        self.assertEqual(P.path_for_engine(self.base, "lada"), self.base)
        self.assertEqual(P.path_for_engine(self.base, "passthrough").name, "perf_profile.passthrough.json")
        self.assertEqual(P.path_for_engine(self.base, "../x").name, "perf_profile.x.json")

    def test_old_profile_without_engine_is_lada(self):
        old = fixture()
        old["fingerprint"].pop("engine", None)
        P.save(old, self.base)
        self.assertEqual(P.profile_engine(old), "lada")
        self.assertEqual(P.load(self.base, "lada")["fingerprint"].get("engine"), None)
        self.assertIsNone(P.load(self.base, "passthrough"))
        self.assertIsNotNone(P.load(self.base))                       # no engine asked: as before

    def test_engines_do_not_share_a_file(self):
        P.save(self.profile("lada"), self.base)
        P.save(self.profile("passthrough"), P.path_for_engine(self.base, "passthrough"))
        self.assertEqual(P.profile_engine(P.load(P.path_for_engine(self.base, "passthrough"), "passthrough")), "passthrough")
        self.assertIsNone(P.load(P.path_for_engine(self.base, "passthrough"), "lada"))
        self.assertIsNone(P.load(self.base, "passthrough"))

    def runner(self):
        r = perf_runner.PerfRunner(settings_provider=lambda: {}, is_player_busy=lambda: False, profile_path=self.base,
                                   fingerprint_provider=lambda: {})
        self.addCleanup(r.join, 1) if hasattr(r, "join") else None
        return r

    def test_runner_reads_the_profile_of_the_engine_in_use(self):
        register_fake(self)
        lada, fake = self.profile(None), self.profile("fake")
        fake["restore"]["720"]["d0_fps"] = 1.0
        P.save(lada, self.base)
        P.save(fake, P.path_for_engine(self.base, "fake"))
        r = self.runner()
        engines.set_configured_engine("lada")
        self.assertEqual(r.current_profile()[1]["restore"]["720"]["d0_fps"], lada["restore"]["720"]["d0_fps"])
        engines.set_configured_engine("fake")
        self.assertEqual(r.profile_path.name, "perf_profile.fake.json")
        self.assertEqual(r.current_profile()[1]["restore"]["720"]["d0_fps"], 1.0)
        os.remove(P.path_for_engine(self.base, "fake"))
        self.assertIsNone(r.current_profile()[1])                     # a missing fake profile never falls back to lada's

    def test_observation_goes_to_its_own_engine_only(self):
        register_fake(self)
        P.save(self.profile(None), self.base)
        before = self.base.read_bytes()
        r = self.runner()
        engines.set_configured_engine("lada")
        self.assertFalse(r.record_observation(mode="off", height=720, observed_fps=50.0, engine="fake"))
        self.assertEqual(self.base.read_bytes(), before)
        engines.set_configured_engine("fake")
        self.assertFalse(r.record_observation(mode="off", height=720, observed_fps=50.0, engine=None))   # lada's numbers
        self.assertEqual(self.base.read_bytes(), before)

    def test_observation_without_engine_calibrates_lada(self):
        P.save(self.profile(None), self.base)
        r = self.runner()
        engines.set_configured_engine("lada")
        r._fingerprint = lambda: {}
        self.assertTrue(r.record_observation(mode="off", height=720, observed_fps=40.0))
        self.assertIn("off:720", P.load(self.base)["calibration"])

    def test_player_hint_profile_per_engine(self):
        from oh_streamer.player import perf_hint
        register_fake(self)
        P.save(self.profile(None), self.base)
        engines.set_configured_engine("fake")
        with mock.patch.object(P, "default_path", lambda engine=None: P.path_for_engine(self.base, engine)):
            perf_hint._cache.clear()
            self.assertIsNone(perf_hint.get_profile())
            engines.set_configured_engine("lada")
            self.assertIsNotNone(perf_hint.get_profile())
            perf_hint._cache.clear()


class ServiceEngineTests(EngineStateTestCase, ServiceTestBase):
    def setUp(self):
        EngineStateTestCase.setUp(self)
        ServiceTestBase.setUp(self)

    def tearDown(self):
        ServiceTestBase.tearDown(self)

    def test_old_item_is_looked_up_as_lada_and_keeps_its_key(self):
        item = self.add("https://example.com/legacy")
        before = self.svc.store.get(item).cache_key
        self.assertNotIn("engine", self.svc.store.get(item).settings)
        self.assertEqual(self.svc._lookup("https://example.com/legacy", None)["engine"], "lada")
        self.svc._build_job(self.svc.store.get(item))
        got = self.svc.store.get(item)
        self.assertEqual((got.cache_key, got.settings.get("engine")), (before, "lada"))   # recorded, key untouched

    def test_item_of_another_engine_is_a_separate_item_but_a_done_one_still_plays(self):
        register_fake(self)
        old = self.svc.add("https://example.com/v")["id"]
        self.assertTrue(self.svc.set_settings({"engine": "fake"})["ok"])
        new = self.svc.add("https://example.com/v")["id"]
        self.assertNotEqual(old, new)                                  # not silently the same result
        out = self.tmp / "o.mkv"
        out.write_bytes(b"x")
        self.svc.store.update(old, state="done", output_path=str(out))
        found = self.svc._lookup("https://example.com/v", None)
        self.assertEqual((found["id"], found["state"], found["engine"]), (old, "done", "lada"))   # still plays


if __name__ == "__main__":
    unittest.main()
