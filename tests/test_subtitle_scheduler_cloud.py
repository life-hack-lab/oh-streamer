"""Tests for queue/subtitle_scheduler.py (unified scheduler, M2: cloud executor v2). No network, no GPU:
fake ASR / translation providers, fake clock."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr
from oh_streamer.subtitles.providers.http import ProviderError

ASR_ID = {"id": "asr-test", "provider": "openai-compatible", "remote_model": "whisper-test"}
MT_ID = {"id": "mt-test", "provider": "openai-compatible", "remote_model": "llm-test"}


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class FakeProviders:
    """Records calls; `asr_errors` / `mt_errors` are consumed one per call (None = succeed)."""

    def __init__(self, cues_per_slice=4):
        self.cues_per_slice = cues_per_slice
        self.asr_calls = []
        self.mt_calls = []          # list of batch sizes
        self.asr_errors = []
        self.mt_errors = []
        self.active = 0
        self.max_active = 0
        self.asr_delay = 0.0
        self.always_mt_error = None
        self.piece_starts = {}      # audio file -> absolute start (the executor adds the ASR pre-roll)
        self._lock = threading.Lock()

    def asr(self, piece, s):
        with self._lock:
            self.asr_calls.append(s.index)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.asr_delay:
                time.sleep(self.asr_delay)
            if self.asr_errors:
                err = self.asr_errors.pop(0)
                if err:
                    raise err
            n = self.cues_per_slice
            lead = s.start - self.piece_starts.get(str(piece), s.start)    # pre-roll in front of the slice
            return [{"start": lead + 1.2 * i, "end": lead + 1.2 * i + 1.1, "text": f"スライス{s.index}の第{i}文です。"}
                    for i in range(n)]
        finally:
            with self._lock:
                self.active -= 1

    def translator(self):
        outer = self

        class Client:
            def translate(self, numbered):
                lines = numbered.splitlines()
                outer.mt_calls.append(len(lines))
                if outer.always_mt_error:
                    raise outer.always_mt_error
                if outer.mt_errors:
                    err = outer.mt_errors.pop(0)
                    if err:
                        raise err
                return "\n".join(re.sub(r"^(\d+)\.\s*(.*)$", r"\1. 翻譯\2", ln) for ln in lines)

            def close(self):
                pass
        return Client()

    def pieces(self, source, out_path, start, duration):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-audio")
        self.piece_starts[str(out_path)] = start
        return [(out_path, 0.0)]


def every_30s_silence(_source, start, end):
    """Fake silencedetect: a 1 s pause at every multiple of 30 s (so cuts land on the nominal times)."""
    k = int(start // 30)
    return [(30.0 * i - 0.5, 30.0 * i + 0.5) for i in range(k, int(end // 30) + 2) if start <= 30.0 * i <= end]


class FakeHost(ss.CloudHost):
    def __init__(self, total=1000.0, limit=None):
        self.total, self.limit = total, limit
        self.published = []
        self.progress = []
        self.context = sp.SchedulerContext(item_state="done")
        self.cancel_after = None
        self._checks = 0
        self.srt_at_publish = []
        self.folder = None

    def source_state(self):
        return True, self.limit, self.total

    def media_id(self):
        return "hls:test-media"

    def scheduler_context(self):
        return self.context

    def check_cancel(self):
        self._checks += 1
        if self.cancel_after is not None and self._checks > self.cancel_after:
            raise RuntimeError("cancel")

    def on_progress(self, info):
        self.progress.append(info)

    def publish(self, result):
        self.published.append(result)
        if self.folder is not None:
            text = (Path(self.folder) / "partial.zh-TW.srt").read_text(encoding="utf-8")
            self.srt_at_publish.append(len(re.findall(r"^\d+$", text, re.M)))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name) / "subtitles"
        self.source = Path(self._tmp.name) / "movie.mp4"
        self.source.write_bytes(b"x" * 5000)
        self.clock = FakeClock()
        self.fake = FakeProviders()
        self.host = FakeHost()
        self.host.folder = self.folder

    def deps(self, **over):
        values = dict(asr=self.fake.asr, translator=self.fake.translator, asr_identity=ASR_ID,
                      mt_identity=MT_ID, pieces=self.fake.pieces, probe_duration=lambda s: 1000.0,
                      detect_silences=every_30s_silence, asr_params={"p": 1}, mt_params={"batch_size": 30},
                      asr_model_id="asr-test")
        values.update(over)
        return ss.CloudDeps(**values)

    def executor(self, **kw):
        kw.setdefault("deps", self.deps())
        kw.setdefault("host", self.host)
        kw.setdefault("slice_seconds", 300.0)       # fixtures are laid out for 300 s slices (cloud default is 60 s)
        kw.setdefault("clock", self.clock)
        kw.setdefault("sleep", self.clock.sleep)
        return ss.CloudSliceExecutor(self.folder, self.source, **kw)

    def run_ex(self, wait=False, **kw):
        ex = self.executor(**kw)
        return ex, ex.run(wait=wait)


class GrowingDownloadTests(Base):
    def test_slices_grow_with_the_download(self):
        self.host.limit = 40.0                      # first cut window (20-45 s, T3.6) not readable yet
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "download")
        self.assertEqual(self.fake.asr_calls, [])
        self.assertEqual(ex.plan.slices, [])

        self.host.limit = 200.0
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "download")
        self.assertEqual(self.fake.asr_calls, [0])  # only the slice whose audio is fully readable
        self.assertEqual([(s.start, s.end) for s in ex.plan.slices], [(0.0, 30.0)])
        self.assertEqual(self.fake.mt_calls, [4])
        self.assertTrue(ex.plan.slices[0].all_done())
        self.assertTrue((self.folder / "partial.zh-TW.srt").is_file())
        self.assertFalse(res.publication["complete"])

        self.host.limit = None                      # download finished
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.asr_calls, [0, 1, 2, 3, 4])      # slice 0 not recognised again
        self.assertTrue(res.publication["complete"])
        self.assertTrue((self.folder / "sakura.zh-TW.srt").is_file())
        self.assertTrue(sp.load_slice_plan(self.folder).slices[-1].all_done())

    def test_audio_past_the_readable_prefix_is_never_extracted(self):
        self.host.limit = 400.0
        ex, res = self.run_ex()
        for i in self.fake.asr_calls:
            self.assertLessEqual(ex.plan.slices[i].end, 400.5)
        self.assertEqual(res.reason, "download")


class CacheTests(Base):
    def test_cache_hit_does_not_call_providers(self):
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "complete")
        calls = (len(self.fake.asr_calls), len(self.fake.mt_calls))
        self.assertTrue(list((self.folder / "cache" / "asr").glob("*.json")))
        self.assertTrue(list((self.folder / "cache" / "mt").glob("*.json")))
        self.assertFalse(list((self.folder / "cache" / "mt").glob("*.partial.json")))

        (self.folder / sp.SLICE_PLAN_FILE).unlink()          # plan lost (e.g. new item folder state)
        ex2, res2 = self.run_ex()
        self.assertEqual(res2.reason, "complete")
        self.assertEqual((len(self.fake.asr_calls), len(self.fake.mt_calls)), calls)

    def test_new_translation_model_reuses_asr(self):
        self.run_ex()
        asr_calls = len(self.fake.asr_calls)
        (self.folder / sp.SLICE_PLAN_FILE).unlink()
        mt_calls = len(self.fake.mt_calls)
        ex, res = self.run_ex(deps=self.deps(mt_identity=dict(MT_ID, id="mt-other")))
        self.assertEqual(res.reason, "complete")
        self.assertEqual(len(self.fake.asr_calls), asr_calls)
        self.assertGreater(len(self.fake.mt_calls), mt_calls)

    def test_media_id_independent_of_mtime_and_path(self):
        a = sr.media_fingerprint(self.source)
        os.utime(self.source, (1, 1))
        self.assertEqual(sr.media_fingerprint(self.source), a)
        other = Path(self._tmp.name) / "copy.mp4"
        other.write_bytes(self.source.read_bytes())
        self.assertEqual(sr.media_fingerprint(other), a)
        other.write_bytes(b"y" * 5000)
        self.assertNotEqual(sr.media_fingerprint(other), a)

    def test_fingerprint_media_id_used_without_host_id(self):
        host = FakeHost()
        host.media_id = lambda: None
        ex, res = self.run_ex(host=host)
        self.assertTrue(ex.plan.media_id.startswith("file:"))
        self.assertEqual(res.reason, "complete")

    def test_other_media_gets_a_new_plan_but_keeps_cache(self):
        self.run_ex()
        host = FakeHost()
        host.media_id = lambda: "hls:another"
        ex, res = self.run_ex(host=host)
        self.assertEqual(ex.plan.media_id, "hls:another")
        self.assertEqual(res.reason, "complete")


class QuotaTests(Base):
    def test_rate_limit_defers_instead_of_failing(self):
        self.fake.mt_errors = [ProviderError("rate_limit", "429", retryable=True, retry_after=30.0)]
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "quota")
        self.assertAlmostEqual(res.wait_until, self.clock() + 30.0, delta=0.5)
        state = ex.plan.slices[0].get("mt")
        self.assertEqual(state.status, sp.PENDING)
        self.assertEqual(state.attempt, 0)
        self.assertFalse(res.status.stuck)
        self.assertEqual(sp.load_slice_plan(self.folder).slices[0].get("mt").retry_at, state.retry_at)

        # not yet: the deferred unit is not retried
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "quota")
        self.clock.t += 31
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "complete")

    def test_wait_true_sleeps_through_quota(self):
        self.fake.asr_errors = [ProviderError("quota_daily", "daily", retryable=True, retry_after=120.0)]
        ex, res = self.run_ex(wait=True)
        self.assertEqual(res.reason, "complete")
        self.assertGreaterEqual(self.clock.t, 1000.0 + 120.0)


class PublishTests(Base):
    def test_every_translated_batch_is_published(self):
        self.fake.cues_per_slice = 70               # 30 + 30 + 10 per slice
        self.host.total = 380.0                     # two slices (90 s first slice + 290 s)
        # a 30 s first slice (the default since T3.6) holds only 24 of the 70 cues: this test is about the
        # per-batch publication, so it keeps the first slice long enough for 70 cues
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 380.0), first_slice_seconds=90.0)
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.mt_calls, [30, 30, 10, 30, 30, 10])
        in_flight = [(p["cues"], p["slices_done"]) for p in self.host.published]
        # slice 0: 30 and 60 cues appear before the slice is done, then 70
        self.assertIn((30, 0), in_flight)
        self.assertIn((60, 0), in_flight)
        self.assertIn((70, 1), in_flight)
        # slice 1 batches come on top of slice 0's finished cues
        self.assertIn((100, 1), in_flight)
        self.assertIn((130, 1), in_flight)
        self.assertEqual(in_flight[-1], (140, 2))
        self.assertEqual(self.host.srt_at_publish[in_flight.index((30, 0))], 30)
        self.assertEqual(self.host.published[-1]["complete"], True)

    def test_batch_checkpoint_resumes_after_failure(self):
        self.fake.cues_per_slice = 70
        self.host.total = 90.0
        self.fake.mt_errors = [None, ProviderError("auth", "bad key")]     # 2nd batch fatal
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "stuck")
        self.assertTrue(list((self.folder / "cache" / "mt").glob("*.partial.json")))
        ex.plan.reset_failed()
        sp.save_slice_plan(ex.plan, self.folder)
        self.fake.mt_calls.clear()
        ex2, res2 = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res2.reason, "complete")
        self.assertEqual(self.fake.mt_calls, [30, 10])      # the first 30 were not translated again

    def test_garbled_numbering_retried_in_halves(self):
        self.fake.cues_per_slice = 4
        self.fake.mt_errors = [ProviderError("bad_response", "garbled")]
        self.host.total = 90.0
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.mt_calls, [4, 2, 2])

    def test_garbled_batch_is_not_resent_when_the_half_hits_a_429(self):
        # S1: 30 garbled (200) -> the 15-batch is rate limited -> the unit used to restart with the full 30 forever
        self.fake.cues_per_slice = 30
        self.host.total = 90.0
        self.fake.mt_errors = [ProviderError("bad_response", "garbled"),
                               ProviderError("rate_limit", "429", retryable=True, retry_after=2.0)]
        ex, res = self.run_ex(wait=True, deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.mt_calls[:2], [30, 15])
        self.assertEqual(self.fake.mt_calls[2:], [15, 15])      # never the garbled 30 again
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    def test_429_after_shrink_keeps_checkpoint_and_size(self):
        self.fake.cues_per_slice = 30
        self.host.total = 90.0
        self.fake.mt_errors = [ProviderError("bad_response", "garbled"), None,
                               ProviderError("rate_limit", "429", retryable=True, retry_after=2.0)]
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "quota")
        self.assertEqual(self.fake.mt_calls, [30, 15, 15])
        partials = list((self.folder / "cache" / "mt").glob("*.partial.json"))
        self.assertEqual(len(partials), 1)
        self.clock.t += 31
        self.fake.mt_calls.clear()
        res = ex.run(wait=True)
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.mt_calls, [15])

    def test_shrink_returns_to_full_size_after_the_bad_stretch(self):
        self.fake.cues_per_slice = 70
        self.host.total = 90.0
        self.fake.mt_errors = [ProviderError("bad_response", "garbled")]
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.mt_calls, [30, 15, 15, 30, 10])

    def test_single_cue_bad_response_is_not_shrunk_further(self):
        self.fake.cues_per_slice = 1
        self.host.total = 90.0
        self.fake.always_mt_error = ProviderError("bad_response", "garbled")
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(self.fake.mt_calls[0], 1)
        self.assertNotEqual(res.reason, "complete")


class FailureTests(Base):
    def test_fatal_error_marks_failed_and_other_slices_continue(self):
        self.fake.asr_errors = [ProviderError("auth", "bad key")]
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "stuck")
        st = ex.plan.slices[0].get("asr")
        # M4b: a single-candidate chain gives up = failed(exhausted); the candidate's own reason is in `tried`
        self.assertEqual((st.status, st.fail_kind, st.attempt), (sp.FAILED, "exhausted", 1))
        self.assertEqual(st.model_id, "asr-test")
        self.assertEqual(st.tried["asr-test"]["kind"], "auth")
        self.assertTrue(res.status.stuck)
        self.assertEqual([(c.id, c.kind) for c in res.status.failures[0].candidates], [("asr-test", "auth")])
        self.assertTrue(ex.plan.slices[1].all_done())        # failure of the head slice does not stop the rest

    def test_transient_error_retried_then_succeeds(self):
        self.fake.mt_errors = [ProviderError("server", "500", retryable=True)] * 2
        ex, res = self.run_ex(wait=True)
        self.assertEqual(res.reason, "complete")
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    def test_transient_error_gives_up_after_max_attempts(self):
        self.fake.always_mt_error = ProviderError("network", "down", retryable=True)
        ex, res = self.run_ex(wait=True)
        self.assertEqual(res.reason, "stuck")
        st = ex.plan.slices[0].get("mt")
        self.assertEqual((st.status, st.fail_kind, st.attempt), (sp.FAILED, "exhausted", ss.MAX_ATTEMPTS))
        self.assertEqual(st.tried["mt-test"]["kind"], "network")

    def test_fatal_error_hands_off_to_a_local_backup(self):
        # M4b (replaces the M2 `handoff` hook): the current frozen plan's chain decides the next candidate
        local = {"id": "gpu-asr", "kind": "asr", "provider": "local", "recipe": "__dual__", "model_path": "x"}
        frozen = {"models": {"asr": dict(ASR_ID, kind="asr"), "translation": dict(MT_ID, kind="translation")},
                  "fallback": {"asr": [local], "translation": []}, "policy": {}}
        self.fake.asr_errors = [ProviderError("auth", "bad key")]
        ex, res = self.run_ex(plan_provider=lambda: frozen)
        st = ex.plan.slices[0].get("asr")
        self.assertEqual((st.status, st.wait_executor, st.candidate), (sp.PENDING, "local", "gpu-asr"))
        self.assertEqual(st.tried["asr-test"]["kind"], "auth")
        self.assertEqual(res.status.waiting_executor, "local")
        self.assertEqual((res.reason, res.waiting_executor), ("idle", "local"))
        self.assertTrue(ex.plan.slices[1].all_done())        # the other slices stay on the cloud primary

    def test_candidates_know_their_executor(self):
        frozen = {"models": {"asr": {"id": "g", "provider": "openai-compatible"},
                             "translation": {"id": "t", "provider": "openai-compatible"}},
                  "fallback": {"translation": [{"id": "sakura", "provider": "local"}]}}
        self.assertEqual(ss.candidates_for(frozen, "translation"), [("t", "cloud"), ("sakura", "local")])
        self.assertEqual(ss.candidates_for(frozen, "asr"), [("g", "cloud")])


class RecoveryTests(Base):
    def test_running_work_is_released_on_restart(self):
        self.host.limit = 200.0
        ex, _ = self.run_ex()
        ex.plan.mark_running(0, "mt", "cloud")        # simulate a crash in the middle of work
        sp.save_slice_plan(ex.plan, self.folder)
        # the mt result is cached, but the plan said running: after restart it is released and finishes
        # from the cache without calling the translator again
        before = len(self.fake.mt_calls)
        self.host.limit = None
        ex2, res = self.run_ex()
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.fake.asr_calls.count(0), 1)
        self.assertEqual(len([c for c in self.fake.mt_calls]), before + 4)   # 4 more slices, slice 0 from cache
        for s in ex2.plan.slices:
            self.assertEqual(s.get("mt").status, sp.DONE)

    def test_done_without_cached_result_runs_again(self):
        self.host.total = 90.0
        self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        for f in (self.folder / "cache" / "mt").glob("*.json"):
            f.unlink()
        calls = len(self.fake.mt_calls)
        ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "complete")
        self.assertEqual(len(self.fake.mt_calls), calls + 1)
        self.assertEqual(len(self.fake.asr_calls), 1)

    def test_cancel_releases_running_work(self):
        self.host.cancel_after = 6
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "cancelled")
        plan = sp.load_slice_plan(self.folder)
        for s in plan.slices:
            for st in s.tasks.values():
                self.assertNotEqual(st.status, sp.RUNNING)


class ConcurrencyTests(Base):
    def test_default_is_one_request_at_a_time(self):
        self.fake.asr_delay = 0.02
        self.run_ex()
        self.assertEqual(self.fake.max_active, 1)

    def test_configurable_concurrency(self):
        self.fake.asr_delay = 0.05
        ex, res = self.run_ex(concurrency=3, poll_s=0.01)
        self.assertEqual(res.reason, "complete")
        self.assertGreaterEqual(self.fake.max_active, 2)
        self.assertLessEqual(self.fake.max_active, 3)
        self.assertEqual(sorted(self.fake.asr_calls), [0, 1, 2, 3, 4])

    def test_run_in_thread(self):
        ex = self.executor()
        thread = ex.run_in_thread(wait=False)
        thread.join(30)
        self.assertEqual(ex.result.reason, "complete")


class SelectorIntegrationTests(Base):
    def test_watching_seek_gets_the_urgent_slice_first(self):
        self.host.context = sp.SchedulerContext(play_pos=700.0, watched=True, item_state="processing")
        ex, res = self.run_ex()
        # the seek slice is most urgent; the head slice no longer jumps the queue once the viewer is past it
        # (T3.6: rule 2 is skipped when watching with play_pos >= head.end), the next slice is next
        self.assertEqual(self.fake.asr_calls[:2], [3, 4])


class HorizonIdleTests(Base):
    """sample video A: owned work the smart horizon holds back is "horizon" (the run ends, no wait_until), never
    "idle until now" (which the host restarted at once, thousands of times a night)."""

    def setUp(self):
        super().setUp()
        self.host.total = 3600.0
        self.host.context = sp.SchedulerContext(play_pos=10.0, watched=True, item_state="processing")

    def executor(self, **kw):
        kw.setdefault("deps", self.deps(probe_duration=lambda s: 3600.0))
        return super().executor(**kw)

    def covered(self, ex, play_pos):
        return sp.covered_end(ex.plan, play_pos) - play_pos

    def test_watched_run_fills_to_the_low_watermark_then_ends_with_horizon(self):
        ex, res = self.run_ex(wait=True)
        self.assertEqual((res.reason, res.wait_until, res.waiting_executor), ("horizon", None, None))
        self.assertGreaterEqual(self.covered(ex, 10.0), sp.DEFAULT_LOW_WATERMARK_SECONDS)
        self.assertFalse(ex.plan.open)
        self.assertIsNone(sp.select_next_task(ex.plan, ex._dispatch(None)[0], ex.EXECUTOR))
        self.assertTrue(any(not s.all_done() for s in ex.plan.slices))      # work left, held back
        calls = len(self.fake.asr_calls)
        # a restart with nothing new ends the same way without any request
        ex, res = self.run_ex(wait=True)
        self.assertEqual((res.reason, res.wait_until), ("horizon", None))
        self.assertEqual(len(self.fake.asr_calls), calls)

    def test_playback_below_the_low_watermark_resumes_work(self):
        ex, _res = self.run_ex(wait=True)
        end = sp.covered_end(ex.plan, 10.0)
        calls = len(self.fake.asr_calls)
        play = end - sp.DEFAULT_LOW_WATERMARK_SECONDS + 60.0                 # buffer now below 20 min
        self.host.context = sp.SchedulerContext(play_pos=play, watched=True, item_state="processing")
        self.assertEqual(ex._idle_reason(None), ("idle", self.clock(), None))  # dispatchable again: not horizon
        ex, res = self.run_ex(wait=True)
        self.assertEqual(res.reason, "horizon")
        self.assertGreater(len(self.fake.asr_calls), calls)
        self.assertGreaterEqual(self.covered(ex, play), sp.DEFAULT_LOW_WATERMARK_SECONDS)

    def test_seek_beyond_the_buffer_resumes_work_at_the_seek_slice(self):
        ex, _res = self.run_ex(wait=True)
        calls = len(self.fake.asr_calls)
        self.host.context = sp.SchedulerContext(play_pos=3500.0, watched=True, item_state="processing")
        ex, res = self.run_ex(wait=True)
        self.assertEqual(self.fake.asr_calls[calls], sp.find_slice_index_for_position(ex.plan.slices, 3500.0))
        self.assertIn(res.reason, ("horizon", "complete"))

    def test_unwatched_item_stops_at_the_horizon_cap(self):
        self.host.context = sp.SchedulerContext(item_state="processing")
        ex, res = self.run_ex(wait=True)
        self.assertEqual((res.reason, res.wait_until), ("horizon", None))
        cap = ex.plan.horizon_minutes * 60
        self.assertTrue(all(s.all_done() for s in ex.plan.slices if s.start < cap))
        self.assertFalse(any(s.all_done() for s in ex.plan.slices if s.start >= cap))
        self.host.context = sp.SchedulerContext(item_state="done")          # restoration done: everything due
        ex, res = self.run_ex(wait=True)
        self.assertEqual(res.reason, "complete")


class TimingAndSimulationTests(Base):
    def test_simulated_download_and_timing(self):
        sim = ss.SimulatedDownloadHost(380.0, rate=50.0, clock=self.clock)
        sim.media_id = lambda: "hls:test-media"
        timing = ss.TimingHost(sim, clock=self.clock)
        self.fake.mt_errors = [ProviderError("rate_limit", "429", retryable=True, retry_after=7.0)]
        ex, res = self.run_ex(wait=True, host=timing, download_poll_s=0.5,
                              deps=self.deps(probe_duration=lambda s: 380.0))
        self.assertEqual(res.reason, "complete")
        # the readable prefix grew with the (fake) clock: nothing before the first cut window (45 s / 50 = 0.9 s)
        self.assertGreaterEqual(timing.first_asr_done_s, 0.8)
        self.assertGreaterEqual(timing.first_subtitle_s, timing.first_asr_done_s)
        self.assertEqual(sorted(timing.slice_done_s), [0, 1, 2])         # 30 s first slice: 30 + 300 + 50
        self.assertEqual(timing.defers, 1)
        self.assertAlmostEqual(timing.defer_wait_s, 7.0, delta=0.1)
        self.assertEqual(timing.defer_kinds, {"rate_limit": 1})
        self.assertTrue(timing.complete_s)
        path = ss.write_timing(self.folder, timing, ex.plan)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["first_slice_seconds"], sp.DEFAULT_FIRST_SLICE_SECONDS)
        self.assertEqual(data["slices"][0], [0, 0.0, 30.0])
        self.assertTrue(data["publishes"])

    def test_first_slice_seconds_override(self):
        for first, expect in ((60.0, 60.0), (None, 30.0)):
            with self.subTest(first=first):
                self.setUp()
                self.host.total = 1000.0
                ex, res = self.run_ex(first_slice_seconds=first)
                self.assertEqual(ex.plan.slices[0].end, expect)

    def test_mt_cache_records_actual_model(self):
        self.host.total = 90.0
        fake_client = self.fake.translator

        def translator():
            c = fake_client()
            c.last_source = {"id": "mt-sibling"}
            return c
        self.run_ex(deps=self.deps(translator=translator, probe_duration=lambda s: 90.0))
        doc = json.loads(next((self.folder / "cache" / "mt").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(doc["model_id"], "mt-sibling")
        self.assertEqual(doc["models_used"], ["mt-sibling"])


if __name__ == "__main__":
    unittest.main()
