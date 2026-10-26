"""Tests for queue/subtitle_host.py (M7 / T1): QueueSubtitleHost and the cloud half of SubtitleSchedulerMixin,
against a fake service object (not wired into QueueService yet). No network, no GPU, no real executor."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import copy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from oh_streamer.queue import subtitle_host as sh
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.queue.subtitle_recovery import SubtitleRecoveryMixin
from oh_streamer.queue.queue_service import _Run
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp

CLOUD_ASR = {"id": "g-asr", "kind": "asr", "provider": "openai-compatible", "base_url": "https://example.org/v1",
             "remote_model": "whisper"}
CLOUD_MT = {"id": "g-mt", "kind": "translation", "provider": "openai-compatible",
            "base_url": "https://example.org/v1", "remote_model": "qwen"}
LOCAL_ASR = {"id": "l-asr", "kind": "asr", "provider": "local", "recipe": "__dual__", "model_path": "x"}
LOCAL_MT = {"id": "l-mt", "kind": "translation", "provider": "local", "recipe": "sakura-v1", "model_path": "x"}


def frozen(asr=CLOUD_ASR, mt=CLOUD_MT, asr_fallback=(), mt_fallback=()):
    return {"ready": True, "errors": [], "models": {"asr": asr, "translation": mt, "draft": None},
            "fallback": {"asr": list(asr_fallback), "translation": list(mt_fallback)}, "policy": {},
            "processing": "smart_horizon", "flow": "direct"}


def make_item(item_id="it1", *, plan=None, source=None, state="processing", status="queued", **sub):
    settings = {"subtitles": True, "subtitle_runtime_contract": "cloud-first-1",
                "subtitle_resolved": plan or frozen(), "subtitle_configs": [{}]}
    return SimpleNamespace(id=item_id, state=state, settings=settings, subtitle={"status": status, **sub},
                           source_path=source, output_path=None, source_status="complete")


class FakeStore:
    def __init__(self, root: Path, settings=None):
        self.root = root
        self.items_by_id = {}
        self.settings = {"subtitle_cloud_enabled": True, **(settings or {})}
        self.updates = []

    def add(self, item):
        self.items_by_id[item.id] = item
        return item

    def get(self, item_id):
        return self.items_by_id.get(item_id)

    def items(self):
        return list(self.items_by_id.values())

    def item_dir(self, item_id):
        path = self.root / item_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_settings(self):
        return copy.deepcopy(self.settings)

    def update(self, item_id, **fields):
        self.updates.append((item_id, fields))
        for key, value in fields.items():
            setattr(self.items_by_id[item_id], key, copy.deepcopy(value))
        return self.items_by_id[item_id]


class FakeService(sh.SubtitleSchedulerMixin, SubtitleRecoveryMixin):
    """Just the members the host / mixin read (see docs in m7_t1_interface.md)."""

    def __init__(self, root: Path, **store_settings):
        self.store = FakeStore(root, store_settings)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._paused_all = False
        self._stopping = threading.Event()
        self._tombstones = set()
        self._reprocessing = set()
        self._generations = {}
        self._subtitle_clocks = {}
        self._watched = set()
        self.player_calls = []
        self.play_calls = []
        self.stops = []
        self.offset_probes = 0
        self.offset_value = 0.0
        self.now = 1000.0
        self._subtitle_clock_fn = lambda: self.now
        self._subtitle_ledger_obj = None
        self.executors = []

    # members of the real QueueService / subtitle_jobs the host uses
    def _subtitle_refreshed_settings(self, item):
        """The fake global settings hold no real subtitle configuration: a refresh keeps the item's own snapshot
        (the settings gate itself is covered by test_subtitle_settings_lock.py). A legacy item still refreshes."""
        from oh_streamer.subtitles.configuration import runtime_plan
        if runtime_plan(item.settings):
            return item.settings
        return super()._subtitle_refreshed_settings(item)

    def _run_is_current(self, run):
        return run.item_id not in self._tombstones

    def _request_stop(self, run, reason):
        self.stops.append((run.item_id, reason))
        run.reason = run.reason or reason
        run.cancel.set()

    def _subtitle_update(self, item_id, **fields):
        with self._lock:
            item = self.store.get(item_id)
            if item is None:
                return
            if "status" in fields and "phase" not in fields:
                fields.update(phase="", detail="")
            item.subtitle = {**item.subtitle, **fields}

    def _watched_ids(self):
        return set(self._watched)

    def _send_to_player(self, payload):
        self.player_calls.append(payload)
        return {"ok": True}

    def _maybe_play_after_subtitles(self, item_id=None):
        self.play_calls.append(item_id)

    def _subtitle_audio_offset(self, path):
        self.offset_probes += 1
        return self.offset_value

    def _session_for(self, item, reopen_complete=True):
        return None

    # no real ledger / providers in these tests
    def _subtitle_ledger_for(self, item):
        return None, None

    def _subtitle_new_cloud_executor(self, item, host):
        executor = FakeExecutor(host)
        self.executors.append(executor)
        return executor


class FakeExecutor:
    """Stands in for CloudSliceExecutor: run() returns the next scripted RunResult (or raises)."""

    def __init__(self, host):
        self.host = host
        self.results = []
        self.stopped = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def run(self, *, wait=False):
        self.started.set()
        assert wait is True
        self.release.wait(5)
        if self.stopped:
            return ss.RunResult(reason="cancelled")
        item = self.results.pop(0) if self.results else ss.RunResult(reason="complete")
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self):
        self.stopped = True
        self.release.set()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = self.root / "video.mp4"
        self.source.write_bytes(b"x")
        self.svc = FakeService(self.root / "items")

    def add(self, item_id="it1", **kwargs):
        kwargs.setdefault("source", str(self.source))
        return self.svc.store.add(make_item(item_id, **kwargs))

    def host(self, item_id="it1", run=None):
        return sh.QueueSubtitleHost(self.svc, item_id, run)

    def wait_idle(self, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.svc._subtitle_cloud is None:
                return
            time.sleep(0.01)
        self.fail("cloud executor thread did not end")


# --------------------------------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------------------------------

class HostSourceTests(Base):
    def test_source_state_delegates_and_removed_item_is_unusable(self):
        item = self.add()
        self.assertEqual(self.host().source_state(), (True, None, None))
        item.source_status = "paused"
        self.assertEqual(self.host().source_state(), (False, 0.0, None))
        self.assertEqual(self.host("missing").source_state(), (False, 0.0, None))

    def test_media_id_is_hls_id_or_none(self):
        self.add()
        with mock.patch.object(sh.SubtitleSchedulerMixin, "_subtitle_cloud_source_key", staticmethod(lambda item: "abc")):
            self.assertEqual(self.host().media_id(), "hls:abc")
        self.assertIsNone(self.host().media_id())          # local file: no sidecar -> fingerprint the file

    def test_downloading_source_limit_is_ready_prefix_minus_one_segment(self):
        item = self.add()
        item.source_status = "downloading"
        playlist = self.root / "index.m3u8"
        playlist.write_text("#EXTM3U\n#EXTINF:10.0,\na\n#EXTINF:10.0,\nb\n#EXTINF:10.0,\nc\n#EXTINF:10.0,\nd\n",
                            encoding="utf-8")
        item.source_path = str(playlist)
        handle = SimpleNamespace(snapshot=lambda: {"status": "downloading",
                                                    "ready_segments": [{"index": 0}, {"index": 1}, {"index": 2}]})
        self.svc._session_for = lambda it, reopen_complete=True: handle
        self.assertEqual(self.host().source_state(), (True, 20.0, 40.0))


class HostContextTests(Base):
    """audio_offset convention (R3): the IPC clock is source audio time; the slice plan starts at 0."""

    def test_play_pos_subtracts_audio_offset(self):
        self.add(audio_offset=2.5)
        self.svc._subtitle_clocks["it1"] = (100.0, True)
        ctx = self.host().scheduler_context()
        self.assertEqual(ctx.play_pos, 97.5)

    def test_play_pos_never_negative(self):
        self.add(audio_offset=5.0)
        self.svc._subtitle_clocks["it1"] = (2.0, False)
        self.assertEqual(self.host().scheduler_context().play_pos, 0.0)

    def test_offset_probed_once_when_not_stored(self):
        self.add()
        self.svc.offset_value = 1.25
        self.svc._subtitle_clocks["it1"] = (11.25, True)
        host = self.host()
        self.assertEqual(host.scheduler_context().play_pos, 10.0)
        self.assertEqual(host.scheduler_context().play_pos, 10.0)
        self.assertEqual(self.svc.offset_probes, 1)

    def test_restore_pos_is_corrected_play_pos_plus_lead_of_raw_clock(self):
        self.add(audio_offset=2.0)
        self.svc._subtitle_clocks["it1"] = (50.0, True)
        seen = []
        self.svc._subtitle_restore_lead = lambda item, pos: (seen.append(pos), 30.0)[1]
        ctx = self.host().scheduler_context()
        self.assertEqual(seen, [50.0])
        self.assertEqual((ctx.play_pos, ctx.restore_pos), (48.0, 78.0))

    def test_done_item_restore_lead_is_infinite_and_state_passes_through(self):
        self.add(state="done")
        ctx = self.host().scheduler_context()
        self.assertEqual(ctx.item_state, "done")
        self.assertEqual(ctx.restore_pos, float("inf"))

    def test_watched_and_strategy_and_limit(self):
        item = self.add()
        item.settings["subtitle_cloud_strategy"] = "full"
        self.svc._watched.add("it1")
        ctx = self.host().scheduler_context()
        self.assertEqual((ctx.watched, ctx.strategy, ctx.max_available), (True, "full", None))

    def test_restore_lead_reads_segments_file(self):
        item = self.add()
        work = self.svc.store.item_dir("it1") / "work"
        work.mkdir()
        (work / "segments.json").write_text(
            '{"n": 100, "key": {"fps": "10"}, "segments": [{"k": 0, "frames": 100}, {"k": 1, "frames": 100}]}',
            encoding="utf-8")
        self.assertEqual(self.svc._subtitle_restore_lead(item, 4.0), 16.0)   # segments end at 20 s
        self.assertEqual(self.svc._subtitle_restore_lead(item, 99.0), 0)
        item.state = "done"
        self.assertEqual(self.svc._subtitle_restore_lead(item, 4.0), float("inf"))


class HostCancelTests(Base):
    def test_ok_when_nothing_is_wrong(self):
        self.add()
        self.host(run=_Run("it1")).check_cancel()

    def test_run_cancel_event(self):
        self.add()
        run = _Run("it1")
        run.cancel.set()
        with self.assertRaises(ss.Cancelled):
            self.host(run=run).check_cancel()

    def test_removed_item_and_stale_generation(self):
        self.add()
        with self.assertRaises(ss.Cancelled):
            self.host("gone", run=_Run("gone")).check_cancel()
        self.svc._tombstones.add("it1")
        with self.assertRaises(ss.Cancelled):
            self.host(run=_Run("it1")).check_cancel()

    def test_cloud_consent_off_stops_cloud_plan_but_not_local_only_plan(self):
        self.add()
        self.svc.store.settings["subtitle_cloud_enabled"] = False
        with self.assertRaises(ss.Cancelled):
            self.host(run=_Run("it1")).check_cancel()
        self.add("loc", plan=frozen(LOCAL_ASR, LOCAL_MT))
        self.host("loc", run=_Run("loc")).check_cancel()

    def test_paused_all_stopping_paused_item_and_status(self):
        item = self.add()
        host = self.host(run=_Run("it1"))
        self.svc._paused_all = True
        with self.assertRaises(ss.Cancelled):
            host.check_cancel()
        self.svc._paused_all = False
        self.svc._stopping.set()
        with self.assertRaises(ss.Cancelled):
            host.check_cancel()
        self.svc._stopping.clear()
        item.subtitle["status"] = "paused"
        with self.assertRaises(ss.Cancelled):
            host.check_cancel()
        item.subtitle["status"] = "partial"
        item.state = "paused"
        with self.assertRaises(ss.Cancelled):
            host.check_cancel()

    def test_perf_check_holding_stops(self):
        self.add()
        self.svc.perf = SimpleNamespace(holding=True)
        with self.assertRaises(ss.Cancelled):
            self.host(run=_Run("it1")).check_cancel()


class HostPublishTests(Base):
    RESULT = {"plan_key": "k1", "covered_s": 120.0, "complete": False, "done_indices": [0], "slices_done": 1,
              "slices_total": 3, "cues": 12, "srt": {"zh-TW": "/a/partial.zh-TW.srt"},
              "subtitles": [{"path": "/a/partial.zh-TW.srt", "lang": "chi", "title": "t", "default": True}]}

    def test_partial_publish_writes_item_fields_and_updates_player(self):
        item = self.add(audio_offset=1.5)
        self.host().publish(self.RESULT)
        sub = item.subtitle
        self.assertEqual((sub["status"], sub["covered_s"], sub["audio_offset"]), ("partial", 120.0, 1.5))
        self.assertEqual(sub["block_srt"], self.RESULT["srt"])
        self.assertEqual(sub["block_subtitles"], self.RESULT["subtitles"])
        self.assertEqual(len(self.svc.player_calls), 1)
        call = self.svc.player_calls[0]
        self.assertEqual((call["cmd"], call["covered_s"], call["complete"], call["subtitle_offset"]),
                         ("subtitle_update", 120.0, False, 1.5))
        self.assertTrue(self.svc._wake.is_set())

    def test_real_publish_language_map_reaches_player_parser(self):
        from oh_streamer.player.queue_bridge import subtitle_update_from_message
        from oh_streamer.subtitles import slice_results as sr
        item = self.add(audio_offset=0.0)
        plan = sp.build_slice_plan(30)
        result = sr.publish(self.root / "published", plan,
                            {0: [{"start": 1, "end": 2, "text": "test", "translation": "測試"}]})
        self.host().publish(result)
        message = self.svc.player_calls[-1]
        self.assertIsInstance(message["srt"], list)
        update = subtitle_update_from_message(message)
        self.assertIsNotNone(update)
        self.assertEqual(update.covered_ranges, result["covered_ranges"])
        self.assertEqual(item.subtitle["covered_ranges"], result["covered_ranges"])

    def test_same_publication_is_sent_to_player_once(self):
        self.add(audio_offset=0.0)
        host = self.host()
        host.publish(self.RESULT)
        host.publish(self.RESULT)
        self.assertEqual(len(self.svc.player_calls), 1)

    def test_first_publication_triggers_play_once(self):
        self.add(audio_offset=0.0)
        host = self.host()
        host.publish(dict(self.RESULT, cues=0, done_indices=[], covered_s=0.0))   # nothing real yet
        self.assertEqual(self.svc.play_calls, [])
        host.publish(self.RESULT)
        host.publish(dict(self.RESULT, covered_s=240.0, plan_key="k2"))
        self.assertEqual(self.svc.play_calls, ["it1"])

    def test_complete_publication_status_depends_on_restoration(self):
        item = self.add(audio_offset=0.0)
        done = dict(self.RESULT, complete=True, covered_s=600.0)
        self.host().publish(done)
        self.assertEqual((item.subtitle["status"], item.subtitle["percent"]), ("ready", 100))
        item.state = "done"
        self.host().publish(done)
        self.assertEqual(item.subtitle["status"], "mux_pending")
        self.assertTrue(item.subtitle["only"])

    def test_user_states_are_not_overwritten(self):
        for status in ("paused", "cancelled", "skipped", "failed", "done"):
            item = self.add(f"i-{status}", status=status, audio_offset=0.0)
            self.host(item.id).publish(self.RESULT)
            self.assertEqual(item.subtitle["status"], status)
            self.assertEqual(item.subtitle["covered_s"], 120.0)       # the data is still recorded

    def test_removed_item_is_ignored(self):
        self.host("gone").publish(self.RESULT)
        self.assertEqual(self.svc.player_calls, [])


class HostProgressTests(Base):
    def test_unit_start_sets_phase_detail_once_and_keeps_status(self):
        item = self.add()
        host = self.host()
        info = {"state": "asr", "slice": 1, "stage": "asr", "total_slices": 4}
        host.on_progress(info)
        self.assertEqual((item.subtitle["status"], item.subtitle["phase"], item.subtitle["detail"]),
                         ("queued", "asr", "雲端切片 2/4 (ASR)"))
        item.subtitle["marker"] = 1
        host.on_progress(info)                      # unchanged -> no write (would have cleared nothing, but
        self.assertEqual(item.subtitle["marker"], 1)  # also proves update was not needed)
        host.on_progress({"state": "mt", "slice": 1, "total_slices": 4})
        self.assertEqual((item.subtitle["status"], item.subtitle["detail"]), ("queued", "雲端切片 2/4 (MT)"))

    def test_published_item_keeps_partial_status(self):
        item = self.add(status="partial", covered_s=60.0)
        self.host().on_progress({"state": "mt", "slice": 2, "total_slices": 4})
        self.assertEqual(item.subtitle["status"], "partial")
        self.assertEqual(item.subtitle["detail"], "雲端切片 3/4 (MT)")

    def test_waiting_and_inactive_items(self):
        item = self.add()
        self.host().on_progress({"state": "waiting_quota", "detail": "quota"})
        self.assertEqual((item.subtitle["phase"], item.subtitle["detail"]), ("waiting", "quota"))
        paused = self.add("p", status="paused")
        self.host("p").on_progress({"state": "asr", "slice": 0, "total_slices": 2})
        self.assertEqual(paused.subtitle["status"], "paused")

    def test_event_wakes_tick_not_written_to_item(self):
        item = self.add()
        self.host().on_event({"type": "subtitle_fallback", "state": "switch"})
        self.assertFalse(hasattr(self.svc, "_subtitle_last_event"))
        self.assertNotIn("fallback", item.subtitle)


# --------------------------------------------------------------------------------------------------
# dispatchable_for (the executor's own selection rule, sample video A)
# --------------------------------------------------------------------------------------------------

def three_slice_plan():
    plan = sp.build_slice_plan(900.0, [(299.0, 301.0), (599.0, 601.0)], first_slice_seconds=None)
    assert len(plan.slices) == 3 and not plan.open, [s.end for s in plan.slices]
    return plan


def handed_to_local():
    """A unit the cloud candidate gave up (auth error) and that now waits for the local executor."""
    return sp.TaskState(status=sp.PENDING, wait_executor="local", candidate="l-asr",
                        tried={"g-asr": {"kind": "auth", "at": 900.0, "executor": "cloud", "detail": ""}})


SAMPLE_A_PLAY_POS = 1701.824        # 28:21.824 when the busy loop started; subtitles covered to 51:55.199


def sample_a_plan():
    """Slice state of the first sample video A busy-loop window (tools/repro_cloud_executor_idle.py): slices 0-10
    done (covered to 3115.199 s), the rest pending, plan closed."""
    cuts = [0, 90, 389.309, 689.309, 989.309, 1297.679, 1597.679,
            1897.679, 2211.092, 2515.199, 2815.199, 3115.199]
    cuts += list(range(3415, 7490, 300)) + [7490.016]
    done = {f"{stage}/primary": {"status": "done"} for stage in sp.STAGES}
    return sp.SlicePlan.from_dict({
        "duration": 7490.016, "media_id": "offline-repro", "tracks": ["primary"],
        "strategy": "smart_horizon", "horizon_minutes": 30, "lazy": {"cuts": cuts, "closed": True},
        "slices": [{"index": i, "start": a, "end": b, "tasks": dict(done) if i < 11 else {}}
                   for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]})


class DispatchableForTests(unittest.TestCase):
    DUE = sp.SchedulerContext(item_state="done")         # everything due: only owner / state / retry filter

    def owners(self, plan=None):
        return fb.owner_function(plan or frozen(), now=1000.0)

    def pick(self, plan, owners, executor, *, now=1000.0, ctx=None):
        task = sh.dispatchable_for(plan, ctx or self.DUE, owners, executor, now=now)
        return None if task is None else (task.slice_index, task.stage, task.track)

    def test_next_unfinished_stage_owned_by_executor(self):
        plan = three_slice_plan()
        owner = self.owners()
        self.assertEqual(self.pick(plan, owner, "cloud"), (0, "asr", "primary"))
        self.assertIsNone(self.pick(plan, owner, "local"))

    def test_done_asr_exposes_mt_and_done_everything_is_none(self):
        plan = three_slice_plan()
        owner = self.owners()
        plan.slices[0].set(sp.STAGE_ASR, plan.primary, sp.TaskState(status=sp.DONE))
        self.assertEqual(self.pick(plan, owner, "cloud"), (0, "mt", "primary"))   # slice 0 now waits at mt
        for s in plan.slices:
            for stage in sp.STAGES:
                s.set(stage, plan.primary, sp.TaskState(status=sp.DONE))
        self.assertIsNone(self.pick(plan, owner, "cloud"))

    def test_local_primary_chain_belongs_to_local(self):
        plan = three_slice_plan()
        owner = self.owners(frozen(LOCAL_ASR, LOCAL_MT, asr_fallback=[CLOUD_ASR]))
        self.assertIsNone(self.pick(plan, owner, "cloud"))
        self.assertEqual(self.pick(plan, owner, "local"), (0, "asr", "primary"))

    def test_retry_at_in_future_and_failed_units_are_skipped(self):
        plan = three_slice_plan()
        owner = self.owners()
        plan.slices[0].set(sp.STAGE_ASR, plan.primary, sp.TaskState(status=sp.PENDING, retry_at=2000.0))
        plan.slices[1].set(sp.STAGE_ASR, plan.primary, sp.TaskState(status=sp.FAILED, fail_kind="exhausted"))
        self.assertEqual(self.pick(plan, owner, "cloud", now=1000.0), (2, "asr", "primary"))
        self.assertEqual(self.pick(plan, owner, "cloud", now=2500.0), (0, "asr", "primary"))
        plan.slices[2].set(sp.STAGE_ASR, plan.primary, sp.TaskState(status=sp.FAILED, fail_kind="exhausted"))
        self.assertIsNone(self.pick(plan, owner, "cloud", now=1000.0))

    def test_handed_off_unit_is_the_other_executors(self):
        plan = three_slice_plan()
        plan.slices[0].set(sp.STAGE_ASR, plan.primary, handed_to_local())
        owner = self.owners(frozen(asr_fallback=[LOCAL_ASR]))
        self.assertEqual(self.pick(plan, owner, "cloud"), (1, "asr", "primary"))
        self.assertEqual(self.pick(plan, owner, "local"), (0, "asr", "primary"))

    def test_track_without_owner_is_unscheduled(self):
        plan = three_slice_plan()
        plan.tracks = ["primary", "cmp-1"]
        for s in plan.slices:
            for stage in sp.STAGES:
                s.set(stage, "primary", sp.TaskState(status=sp.DONE))
        owner = {"primary": self.owners()}
        self.assertIsNone(self.pick(plan, owner, "cloud"))
        owner["cmp-1"] = self.owners()
        self.assertEqual(self.pick(plan, owner, "cloud"), (0, "asr", "cmp-1"))

    def test_watched_buffer_above_the_low_watermark_is_not_dispatchable(self):
        plan = sample_a_plan()
        owner = self.owners()
        watched = sp.SchedulerContext(play_pos=SAMPLE_A_PLAY_POS, watched=True)
        self.assertIsNone(self.pick(plan, owner, "cloud", ctx=watched))
        below = sp.SchedulerContext(play_pos=2000.0, watched=True)        # 1115 s ahead < 20 min
        self.assertEqual(self.pick(plan, owner, "cloud", ctx=below), (11, "asr", "primary"))
        seek = sp.SchedulerContext(play_pos=5000.0, watched=True)
        self.assertEqual(self.pick(plan, owner, "cloud", ctx=seek)[:2],
                         (sp.find_slice_index_for_position(plan.slices, 5000.0), "asr"))

    def test_unwatched_beyond_the_horizon_cap_is_not_dispatchable(self):
        plan = sample_a_plan()
        owner = self.owners()
        self.assertIsNone(self.pick(plan, owner, "cloud", ctx=sp.SchedulerContext(restore_pos=1000.0)))
        self.assertEqual(self.pick(plan, owner, "cloud", ctx=sp.SchedulerContext(restore_pos=2400.0)),
                         (11, "asr", "primary"))                         # cap = restore_pos + 15 min > 3115


# --------------------------------------------------------------------------------------------------
# Mixin: runs, routing, tick, gates
# --------------------------------------------------------------------------------------------------

class RunsAndStopTests(Base):
    def test_runs_lists_existing_slots_only(self):
        self.assertEqual(self.svc._subtitle_runs(), [])
        a, b = _Run("a"), _Run("b")
        self.svc._subtitle_cloud, self.svc._subtitle_local = a, b
        self.assertEqual(self.svc._subtitle_runs(), [a, b])

    def test_stop_executors_cancels_matching_item_only(self):
        mine, other = _Run("it1"), _Run("it2")
        mine.executor, other.executor = mock.Mock(), mock.Mock()
        self.svc._subtitle_cloud, self.svc._subtitle_local = mine, other
        self.assertTrue(self.svc._subtitle_stop_executors("it1", "pause"))
        self.assertTrue(mine.cancel.is_set())
        self.assertFalse(other.cancel.is_set())
        mine.executor.stop.assert_called_once()
        other.executor.stop.assert_not_called()
        self.assertEqual(self.svc.stops, [("it1", "pause")])

    def test_stop_executors_can_join_the_thread(self):
        self.add()
        self.svc._subtitle_start_cloud(self.svc.store.get("it1"))
        executor = self.svc.executors[0]
        executor.release.clear()
        self.assertTrue(self.svc._subtitle_stop_executors("it1", "remove", wait=3.0))
        self.wait_idle()


class RealExecutorConstructionTests(Base):
    def test_builds_a_cloud_executor_with_live_plan_providers(self):
        item = self.add()
        deps = ss.CloudDeps(asr=None, translator=None, asr_identity={"id": "g-asr"}, mt_identity={"id": "g-mt"})
        host = self.host()
        with mock.patch.object(ss, "default_deps", return_value=deps):
            ex = sh.SubtitleSchedulerMixin._subtitle_new_cloud_executor(self.svc, item, host)
        self.assertIsInstance(ex, ss.CloudSliceExecutor)
        self.assertIs(ex.host, host)
        self.assertEqual(ex.folder, self.svc.store.item_dir("it1") / "subtitles")
        self.assertEqual(ex.plan_provider()["models"]["asr"]["id"], "g-asr")
        item.settings["subtitle_resolved"] = frozen(asr=dict(CLOUD_ASR, id="g-asr-2"))     # next pass sees the edit
        self.assertEqual(ex.plan_provider()["models"]["asr"]["id"], "g-asr-2")
        self.assertLessEqual(set(ex.track_plans_provider()), {"primary"})     # the executor drops the primary


class RoutingTests(Base):
    def wanted(self, item):
        return self.svc._subtitle_cloud_wanted(item, self.svc.now)

    def test_new_item_cloud_first_starts_cloud_local_first_does_not(self):
        cloud = self.add("c")
        local = self.add("l", plan=frozen(LOCAL_ASR, LOCAL_MT, asr_fallback=[CLOUD_ASR]))
        self.assertTrue(self.wanted(cloud))
        self.assertFalse(self.wanted(local))

    def test_plan_on_disk_decides_by_pending_units(self):
        item = self.add()
        folder = self.svc.store.item_dir("it1") / "subtitles"
        folder.mkdir()
        plan = three_slice_plan()
        sp.save_slice_plan(plan, folder)
        self.assertTrue(self.wanted(item))
        for s in plan.slices:
            for stage in sp.STAGES:
                s.set(stage, plan.primary, sp.TaskState(status=sp.DONE))
        sp.save_slice_plan(plan, folder)
        self.assertFalse(self.wanted(item))

    def test_mt_on_cloud_after_local_asr(self):
        item = self.add(plan=frozen(LOCAL_ASR, CLOUD_MT))
        folder = self.svc.store.item_dir("it1") / "subtitles"
        folder.mkdir()
        plan = three_slice_plan()
        sp.save_slice_plan(plan, folder)
        self.assertFalse(self.wanted(item))                      # asr first, and it is local
        for s in plan.slices:
            s.set(sp.STAGE_ASR, plan.primary, sp.TaskState(status=sp.DONE))
        sp.save_slice_plan(plan, folder)
        self.assertTrue(self.wanted(item))

    def test_open_plan_with_everything_done_follows_the_download_on_the_home_executor(self):
        item = self.add()
        folder = self.svc.store.item_dir("it1") / "subtitles"
        folder.mkdir()
        plan = sp.new_slice_plan(900.0, media_id="m")
        sp.save_slice_plan(plan, folder)
        self.assertTrue(plan.open)
        self.assertTrue(self.wanted(item))

    def test_not_wanted_when_off_or_excluded(self):
        item = self.add()
        item.settings["subtitles"] = False
        self.assertFalse(self.wanted(item))
        item.settings["subtitles"] = True
        for status in ("paused", "cancelled", "failed", "skipped", "done", "mux_pending"):
            item.subtitle["status"] = status
            self.assertFalse(self.wanted(item), status)
        item.subtitle["status"] = "queued"
        self.assertTrue(self.wanted(item))

    def test_not_wanted_without_source_or_unusable_source_or_consent(self):
        item = self.add()
        item.source_path = str(self.root / "nope.mp4")
        self.assertFalse(self.wanted(item))
        item.source_path = str(self.source)
        item.source_status = "paused"
        self.assertFalse(self.wanted(item))
        item.source_status = "complete"
        self.svc.store.settings["subtitle_cloud_enabled"] = False
        self.assertFalse(self.wanted(item))
        self.svc.store.settings["subtitle_cloud_enabled"] = True
        self.assertTrue(self.wanted(item))

    def test_unready_plan_fails_item_except_cloud_disabled(self):
        bad = self.add("bad", plan=dict(frozen(), ready=False, errors=[{"code": "not_configured", "role": "asr"}]))
        self.assertFalse(self.wanted(bad))
        self.assertEqual(bad.subtitle["status"], "failed")
        self.assertIn("not_configured", bad.subtitle["error"])
        soft = self.add("soft", plan=dict(frozen(), ready=False, errors=[{"code": "cloud_disabled"}]))
        self.assertFalse(self.wanted(soft))
        self.assertEqual(soft.subtitle["status"], "queued")


class LegacyUpgradeTests(Base):
    def legacy_item(self):
        item = self.add()
        item.settings = {"subtitles": True, "subtitle_configs": [{}]}        # no contract, no resolved plan
        return item

    def test_legacy_item_is_upgraded_and_saved(self):
        item = self.legacy_item()
        upgraded = {"subtitles": True, "subtitle_runtime_contract": "cloud-first-1", "subtitle_resolved": frozen(),
                    "subtitle_configs": [{}]}
        with mock.patch("oh_streamer.queue.subtitle_recovery.refresh_snapshot_settings",
                        return_value=upgraded) as refresh:
            result = self.svc._subtitle_lock_settings(item)
        self.assertIs(result, self.svc.store.get("it1"))
        self.assertEqual(result.settings["subtitle_runtime_contract"], "cloud-first-1")
        glob = refresh.call_args.args[1]
        self.assertEqual(glob["subtitle_runtime_contract"], "cloud-first-1")     # forced even if the global is legacy
        self.assertEqual(self.svc.store.updates[-1][0], "it1")

    def test_upgrade_failure_or_still_legacy_leaves_item_alone(self):
        item = self.legacy_item()
        with mock.patch("oh_streamer.queue.subtitle_recovery.refresh_snapshot_settings", side_effect=ValueError("x")):
            self.assertIsNone(self.svc._subtitle_lock_settings(item))
        with mock.patch("oh_streamer.queue.subtitle_recovery.refresh_snapshot_settings",
                        return_value={"subtitles": True}):
            self.assertIsNone(self.svc._subtitle_lock_settings(item))
        self.assertEqual(self.svc.store.updates, [])

    def test_frozen_item_is_returned_unchanged(self):
        item = self.add()
        self.assertIs(self.svc._subtitle_lock_settings(item), item)

    def test_tick_starts_upgraded_legacy_item(self):
        self.legacy_item()
        upgraded = {"subtitles": True, "subtitle_runtime_contract": "cloud-first-1", "subtitle_resolved": frozen(),
                    "subtitle_configs": [{}]}
        with mock.patch("oh_streamer.queue.subtitle_recovery.refresh_snapshot_settings", return_value=upgraded):
            self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)
        self.wait_idle()


class TickTests(Base):
    def test_tick_starts_one_thread_records_result_and_frees_slot(self):
        item = self.add()
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)
        run = None
        self.wait_idle()
        finished = self.svc._subtitle_results["it1"]["cloud"]
        self.assertEqual(finished.reason, "complete")
        self.assertTrue(self.svc._wake.is_set())
        self.assertEqual(item.subtitle["status"], "queued")          # complete leaves the host's publish status alone

    def test_run_slot_is_registered_while_running(self):
        self.add()
        self.svc._subtitle_executor_tick()
        # first executor finishes instantly; start a blocked one to look at the slot
        self.wait_idle()
        self.svc._subtitle_executor_reset("it1")
        original = self.svc._subtitle_new_cloud_executor

        def blocked(item, host):
            ex = original(item, host)
            ex.release.clear()
            return ex

        self.svc._subtitle_new_cloud_executor = blocked
        self.svc._subtitle_executor_tick()
        run = self.svc._subtitle_cloud
        self.assertIsNotNone(run)
        self.assertEqual((run.item_id, run.stage, run.subtitle_only), ("it1", "subtitle_cloud", True))
        self.assertIs(run.executor.host.run, run)
        self.assertIn(run, self.svc._subtitle_runs())
        self.svc._subtitle_executor_tick()                 # slot busy: no second executor
        self.assertEqual(len(self.svc.executors), 2)
        run.executor.release.set()
        self.wait_idle()

    def test_only_one_item_at_a_time_and_watched_first(self):
        self.add("a")
        self.add("b")
        self.svc._watched.add("b")
        self.svc._subtitle_executor_tick()
        self.assertEqual(self.svc.executors[0].host.item_id, "b")
        self.wait_idle()

    def test_complete_is_not_restarted_until_reset(self):
        self.add()
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.svc.now += 100
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)
        self.svc._subtitle_executor_reset("it1")
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 2)
        self.wait_idle()

    def test_stuck_waits_for_reset(self):
        self.add()
        self.svc._subtitle_new_cloud_executor = self._scripted(ss.RunResult(reason="stuck"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.svc.now += 3600
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)
        self.svc._subtitle_executor_reset("it1", "cloud")
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 2)
        self.wait_idle()

    def _scripted(self, *results):
        queue = list(results)

        def make(item, host):
            ex = FakeExecutor(host)
            ex.results = [queue.pop(0)] if queue else []
            self.svc.executors.append(ex)
            return ex

        return make

    def test_idle_with_wait_until_reopens_at_that_time(self):
        self.add()
        self.svc._subtitle_new_cloud_executor = self._scripted(
            ss.RunResult(reason="idle", wait_until=self.svc.now + 120, waiting_executor="local"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.svc.now += 60
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)
        self.svc.now += 61
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 2)
        self.wait_idle()

    def test_idle_without_wait_until_needs_work_back_and_5_seconds(self):
        self.add(plan=frozen(asr_fallback=[LOCAL_ASR]))
        folder = self.svc.store.item_dir("it1") / "subtitles"
        folder.mkdir()
        plan = three_slice_plan()
        for s in plan.slices:               # the work was handed to local
            s.set(sp.STAGE_ASR, plan.primary, handed_to_local())
        sp.save_slice_plan(plan, folder)
        self.svc._subtitle_new_cloud_executor = self._scripted(ss.RunResult(reason="idle", waiting_executor="local"))
        item = self.svc.store.get("it1")
        self.assertTrue(self.svc._subtitle_cloud_wanted(item, self.svc.now) is False)    # nothing for cloud
        # force one run to have happened, then let the work come back (local finished asr, mt is cloud's)
        self.svc._subtitle_results_for("it1")["cloud"] = sh.FinishedRun(
            ss.RunResult(reason="idle", waiting_executor="local"), None, self.svc.now)
        for s in plan.slices:
            s.set(sp.STAGE_ASR, plan.primary, sp.TaskState(status=sp.DONE))
        sp.save_slice_plan(plan, folder)
        self.assertFalse(self.svc._subtitle_cloud_wanted(item, self.svc.now + 2))        # < 5 s
        self.assertTrue(self.svc._subtitle_cloud_wanted(item, self.svc.now + 6))

    def test_cancelled_restarts_after_a_second_if_still_allowed(self):
        self.add()
        self.svc._subtitle_new_cloud_executor = self._scripted(ss.RunResult(reason="cancelled"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)             # same instant
        self.svc.now += 1.5
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 2)
        self.wait_idle()

    def test_cancelled_by_pause_does_not_restart(self):
        item = self.add()
        self.svc._subtitle_new_cloud_executor = self._scripted(ss.RunResult(reason="cancelled"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        item.subtitle["status"] = "paused"
        self.svc.now += 100
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)

    def test_exception_fails_item_and_blocks_restart(self):
        item = self.add()
        self.svc._subtitle_new_cloud_executor = self._scripted(RuntimeError("boom"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.assertEqual(item.subtitle["status"], "failed")
        self.assertIn("boom", item.subtitle["error"])
        self.assertEqual(self.svc._subtitle_results["it1"]["cloud"].reason, "error")
        item.subtitle["status"] = "queued"
        self.svc.now += 100
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)             # needs reset (retry)

    def test_executor_construction_failure_fails_item(self):
        item = self.add()

        def broken(item, host):
            raise ValueError("Subtitle configuration not ready")

        self.svc._subtitle_new_cloud_executor = broken
        self.svc._subtitle_executor_tick()
        self.assertIsNone(self.svc._subtitle_cloud)
        self.assertEqual(item.subtitle["status"], "failed")

    def test_end_of_run_returns_transient_status_to_resting_status(self):
        item = self.add(status="asr")
        self.svc._subtitle_new_cloud_executor = self._scripted(ss.RunResult(reason="source_unavailable"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.assertEqual(item.subtitle["status"], "queued")
        item2 = self.add("it2", status="translate", covered_s=60.0)
        self.svc._subtitle_new_cloud_executor = self._scripted(ss.RunResult(reason="idle", waiting_executor="local"))
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.assertEqual(item2.subtitle["status"], "partial")

    def test_tick_stops_running_executor_when_consent_is_revoked(self):
        self.add()
        original = self.svc._subtitle_new_cloud_executor

        def blocked(item, host):
            ex = original(item, host)
            ex.release.clear()
            return ex

        self.svc._subtitle_new_cloud_executor = blocked
        self.svc._subtitle_executor_tick()
        run = self.svc._subtitle_cloud
        self.svc._subtitle_executor_tick()
        self.assertFalse(run.cancel.is_set())
        self.svc.store.settings["subtitle_cloud_enabled"] = False
        self.svc._subtitle_executor_tick()
        self.assertTrue(run.cancel.is_set())
        self.assertEqual(self.svc.stops[-1], ("it1", "auto"))
        self.assertTrue(run.executor.stopped)
        self.wait_idle()
        self.assertEqual(self.svc._subtitle_results["it1"]["cloud"].reason, "cancelled")

    def test_removed_item_result_is_not_remembered(self):
        self.add()
        original = self.svc._subtitle_new_cloud_executor

        def blocked(item, host):
            ex = original(item, host)
            ex.release.clear()
            return ex

        self.svc._subtitle_new_cloud_executor = blocked
        self.svc._subtitle_executor_tick()
        run = self.svc._subtitle_cloud
        self.svc._tombstones.add("it1")
        self.svc._subtitle_stop_executors("it1", "remove", wait=3.0)
        self.wait_idle()
        self.assertNotIn("cloud", getattr(self.svc, "_subtitle_results", {}).get("it1", {}))



class HorizonStartTests(Base):
    """sample video A: the host starts an executor only for work the executor's selector would take now (smart
    horizon included), and an idle end without a real wait never bypasses the 5 s restart interval."""

    def setUp(self):
        super().setUp()
        self.item = self.add(audio_offset=0.0)
        self.folder = self.svc.store.item_dir("it1") / "subtitles"
        self.folder.mkdir()
        sp.save_slice_plan(sample_a_plan(), self.folder)
        self.svc._watched = {"it1"}
        self.play(SAMPLE_A_PLAY_POS)
        self.svc._subtitle_new_cloud_executor = self.scripted

    def play(self, pos):
        self.svc._subtitle_clocks["it1"] = (pos, True)

    def scripted(self, item, host):
        ex = FakeExecutor(host)
        ex.results = [ss.RunResult(reason="horizon")]
        self.svc.executors.append(ex)
        return ex

    def wanted(self, at=None):
        return self.svc._subtitle_cloud_wanted(self.svc.store.get("it1"), self.svc.now if at is None else at)

    def finished(self, reason, wait_until=None, waiting_executor=None):
        self.svc._subtitle_results_for("it1")["cloud"] = sh.FinishedRun(
            ss.RunResult(reason=reason, wait_until=wait_until, waiting_executor=waiting_executor), None, self.svc.now)

    def test_buffer_above_the_low_watermark_starts_nothing_until_playback_catches_up(self):
        self.finished("idle", wait_until=self.svc.now)                 # the looping run's end, as in the report
        for _ in range(60):                                            # one minute of ticks: no restart at all
            self.svc.now += 1
            self.svc._subtitle_executor_tick()
        self.assertEqual(self.svc.executors, [])
        self.play(2000.0)                                              # 1115 s ahead: below the 20 min watermark
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)
        self.wait_idle()

    def test_horizon_end_is_not_restarted_while_held_and_waits_5_seconds_otherwise(self):
        self.play(2000.0)
        self.svc._subtitle_executor_tick()
        self.wait_idle()
        self.assertEqual(self.svc._subtitle_results_for("it1")["cloud"].reason, "horizon")
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)                   # same instant: 5 s interval
        self.play(SAMPLE_A_PLAY_POS)
        self.svc.now += 30
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 1)                   # held by the horizon again
        self.play(2000.0)
        self.svc._subtitle_executor_tick()
        self.assertEqual(len(self.svc.executors), 2)
        self.wait_idle()

    def test_seek_outside_the_buffer_starts_and_inside_does_not(self):
        self.play(100.0)                                               # seek back: 50 min still covered
        self.assertFalse(self.wanted())
        self.play(5000.0)                                              # seek past the subtitles
        self.assertTrue(self.wanted())

    def test_unwatched_item_beyond_the_horizon_cap_waits_for_restoration(self):
        self.svc._watched = set()
        self.play(0.0)
        self.assertFalse(self.wanted())                                # cap 30 min, all done below it
        self.item.state = "done"                                       # restoration finished: everything due
        self.assertTrue(self.wanted())

    def test_unwatched_item_with_work_inside_the_cap_starts(self):
        self.svc._watched = set()
        plan = sample_a_plan()
        plan.slices[3].set(sp.STAGE_MT, plan.primary, sp.TaskState())
        sp.save_slice_plan(plan, self.folder)
        self.assertTrue(self.wanted())

    def test_idle_without_a_real_wait_does_not_bypass_the_interval(self):
        self.play(2000.0)                                              # work is there: only the gate decides
        for until in (self.svc.now, self.svc.now - 30):
            self.finished("idle", wait_until=until)
            self.assertFalse(self.wanted(self.svc.now + 1))
            self.assertTrue(self.wanted(self.svc.now + 6))

    def test_quota_wait_until_is_still_honoured(self):
        self.play(2000.0)
        self.finished("quota", wait_until=self.svc.now + 120)
        self.assertFalse(self.wanted(self.svc.now + 60))
        self.assertTrue(self.wanted(self.svc.now + 121))

    def test_handoff_reclaim_time_is_still_honoured(self):
        self.play(2000.0)
        self.finished("idle", wait_until=self.svc.now + 40, waiting_executor="local")
        self.assertFalse(self.wanted(self.svc.now + 10))               # past 5 s, before the reclaim time
        self.assertTrue(self.wanted(self.svc.now + 41))

if __name__ == "__main__":
    unittest.main()
