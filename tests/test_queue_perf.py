"""Performance check in the queue service: fake run_benchmark, temp profile dir, no GPU, no real settings files."""
import json
import time
from unittest import mock
import threading
import types
from pathlib import Path

try:                                    # run_tests.py imports tests.* ; "-s tests" imports flat names
    from tests.test_queue_service import FAKE_WORKER, ServiceTestBase, wait_for
    from tests.test_perf_profile import fixture, rtx5060_accurate, two_detectors
except ImportError:
    from test_queue_service import FAKE_WORKER, ServiceTestBase, wait_for
    from test_perf_profile import fixture, rtx5060_accurate, two_detectors

from oh_streamer.perf import profile as P
from oh_streamer.queue import perf_runner
from oh_streamer.queue import queue_service as QS
from oh_streamer.queue import queue_store


class Cancelled(Exception):
    pass


def sample_profile():
    p = fixture()
    p.update(created="2026-10-11T12:00:00+08:00", duration_s=150.2,
             conditions={"on_ac_power": False, "battery_percent": 63.0},
             settings={"detector": "v4-fast", "restoration": "basicvsrpp-v1.2", "clip_length": 180, "fp16": True})
    p["fingerprint"].update(driver="617.42", torch="2.9.1", lada="0.x")
    return p


class FakeBench:
    """Blocks on `gate`, reports progress, honours `cancel`, optionally fails."""

    def __init__(self):
        self.gate = threading.Event()
        self.started = threading.Event()
        self.calls = []
        self.fail = None
        self.remaining = None
        self.result = sample_profile()

    def __call__(self, options, progress, cancel):
        self.calls.append(dict(options))
        progress("restore 720p d=0,1 ...", 0.3)
        if self.remaining is not None:
            progress("", 0.35, self.remaining)          # bench's per-second tick: no message, time-based progress
        self.started.set()
        while not self.gate.wait(0.02):
            if cancel.is_set():
                raise Cancelled()
        if self.fail:
            raise RuntimeError(self.fail)
        progress("finished", 1.0)
        return self.result


class PerfTestBase(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.bench = FakeBench()
        self.profile_path = self.tmp / "perf" / "perf_profile.json"
        self.fp = {"gpu": "RTX 5060", "driver": "617.42", "torch": "2.9.1", "lada": "0.x"}
        self.cur = {"detection_model": "v4-fast", "restoration_model": "basicvsrpp-v1.2", "max_clip_length": 180}
        self.svc._snapshot_settings = lambda: dict(self.cur)
        perf = self.svc.perf
        perf._profile_path = self.profile_path
        perf._run_benchmark = self.bench
        perf._fingerprint = lambda: dict(self.fp)
        perf._player_settings_path = self.tmp / "player" / "oh_settings.json"

    def tearDown(self):
        self.bench.gate.set()
        super().tearDown()

    def perf_state(self):
        return self.svc.get_state()["perf"]

    def status(self):
        return self.perf_state()["status"]

    def listening_item(self):
        i = self.add("http://h/" + "x", "listen")
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.pids(i), 20))
        return i


class RunTests(PerfTestBase):
    def test_initial_state_and_json_serialisable(self):
        s = self.perf_state()
        self.assertEqual((s["status"], s["kind"], s["progress"], s["error"]), ("none", None, 0.0, None))
        self.assertTrue(s["can_start"])
        self.assertIsNone(s["blocked_reason"])
        self.assertTrue(s["prompt"])
        self.assertIsNone(s["profile"])
        json.dumps(self.svc.get_state())

    def test_start_progress_success_saves_profile(self):
        r = self.svc.perf.start("quick")
        self.assertTrue(r["ok"], r)
        self.assertTrue(self.bench.started.wait(5))
        s = self.perf_state()
        self.assertEqual((s["status"], s["kind"], s["can_start"]), ("running", "quick", False))
        self.assertAlmostEqual(s["progress"], 0.3)
        self.assertEqual(s["stage"], "量測修復速度 720p")
        self.assertEqual(self.bench.calls[0], {"mode": "quick", "detector": "v4-fast",
                                               "restoration": "basicvsrpp-v1.2", "clip_length": 180})
        self.assertFalse(self.svc.perf.start("standard")["ok"])         # one run at a time
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "done", 10))
        s = self.perf_state()
        self.assertEqual(s["progress"], 1.0)
        self.assertFalse(s["prompt"])
        self.assertTrue(s["can_start"])
        self.assertEqual(P.load(self.profile_path)["created"], "2026-10-11T12:00:00+08:00")
        prof = s["profile"]
        self.assertEqual((prof["gpu"], prof["duration_s"], prof["on_ac_power"]), ("RTX 5060", 150.2, False))
        self.assertEqual(prof["warnings"], ["on battery power"])
        self.assertEqual(sorted(prof["matrix"]), ["30", "60", "60r30"])
        self.assertEqual(len(prof["matrix"]["30"]), 9)
        self.assertEqual(set(prof["matrix"]["30"][0]), {"height", "mode", "level", "ratio_typical", "ratio_heavy",
                                                         "prebuffer_s", "interp_fraction", "bottleneck",
                                                         "calibrated", "interp_ok"})
        rows60 = {(r["height"], r["mode"]): r for r in prof["matrix"]["60"]}
        self.assertEqual(rows60[720, "full"]["ratio_typical"], rows60[720, "off"]["ratio_typical"])   # 60 fps stays 1x
        self.assertFalse(rows60[720, "full"]["interp_ok"])
        # 60 fps restored at 30 (player "最高 30fps") = a 30 fps source: same speed, interpolation back to 60
        rows30 = {(r["height"], r["mode"]): r for r in prof["matrix"]["30"]}
        half = {(r["height"], r["mode"]): r for r in prof["matrix"]["60r30"]}
        self.assertEqual(half[720, "auto"], rows30[720, "auto"])
        self.assertIn("summary", prof["recommend"])
        self.assertFalse(self.svc.perf.holding)

    def test_remaining_time_follows_the_ticks(self):
        self.bench.remaining = 125.4
        self.assertTrue(self.svc.perf.start("standard")["ok"])
        self.assertTrue(self.bench.started.wait(5))
        s = self.perf_state()
        self.assertEqual((s["remaining_s"], s["stage"]), (125, "量測修復速度 720p"))   # a tick keeps the stage
        self.assertAlmostEqual(s["progress"], 0.35)
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "done", 10))
        self.assertIsNone(self.perf_state()["remaining_s"])

    def test_standard_kind_uses_default_mode(self):
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(self.bench.started.wait(5))
        self.assertEqual((self.bench.calls[0]["mode"], self.perf_state()["kind"]), ("default", "standard"))
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "done", 10))

    def test_refuses_when_player_busy(self):
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        s = self.perf_state()
        self.assertFalse(s["can_start"])
        self.assertEqual(s["blocked_reason"], "播放器正在即時處理，請先關閉播放器再檢測")
        r = self.svc.perf.start()
        self.assertEqual((r["ok"], r["error"]), (False, s["blocked_reason"]))
        self.assertEqual(self.bench.calls, [])
        self.assertEqual(self.status(), "none")

    def test_cancel(self):
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(self.bench.started.wait(5))
        self.assertTrue(self.svc.perf.cancel()["ok"])
        self.assertTrue(wait_for(lambda: self.status() == "cancelled", 10))
        s = self.perf_state()
        self.assertIsNone(s["error"])
        self.assertFalse(self.svc.perf.holding)
        self.assertFalse(self.profile_path.exists())
        self.assertTrue(s["can_start"])
        self.assertTrue(self.svc.perf.cancel()["ok"])                     # nothing running: harmless

    def test_failure_reports_error_and_releases_hold(self):
        self.bench.fail = "boom"
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(self.bench.started.wait(5))
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "failed", 10))
        s = self.perf_state()
        self.assertEqual(s["error"], "RuntimeError: boom")
        self.assertFalse(self.svc.perf.holding)
        self.assertFalse(self.profile_path.exists())
        self.assertTrue(s["prompt"])
        self.bench.fail = None                                             # a new attempt clears the error
        self.bench.gate.clear()
        self.bench.started.clear()
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(wait_for(lambda: self.perf_state()["error"] is None and self.status() == "running", 5))

    def test_invalid_result_fails(self):
        self.bench.result = {"schema": 99}
        self.bench.gate.set()
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(wait_for(lambda: self.status() == "failed", 10))
        self.assertFalse(self.profile_path.exists())

    def test_player_becoming_busy_cancels_the_run(self):
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(self.bench.started.wait(5))
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.assertTrue(wait_for(lambda: self.status() == "cancelled", 10))
        self.assertEqual(self.perf_state()["error"], perf_runner.CANCELLED_BY_PLAYER)
        self.assertFalse(self.svc.perf.holding)


class HoldTests(PerfTestBase):
    def test_hold_stops_worker_and_resumes_it(self):
        i = self.listening_item()
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.state(i) == "queued", 10))
        state = self.svc.get_state()
        self.assertFalse(state["paused_all"])                              # the user's pause is untouched
        self.assertEqual(state["perf"]["status"], "running")
        self.assertNotEqual(self.state(i), "paused")
        n = len(self.pids(i))
        self.assertFalse(wait_for(lambda: self.svc._wk is not None, 0.5))   # and nothing restarts meanwhile
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and len(self.pids(i)) > n, 20))
        self.assertFalse(self.svc.get_state()["paused_all"])

    def test_hold_resumes_after_failure_and_cancel(self):
        i = self.listening_item()
        self.bench.fail = "x"
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.state(i) == "queued", 10))
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "failed" and self.state(i) == "processing"
                                 and self.svc._wk is not None, 20))
        self.bench.gate.clear()
        self.bench.fail = None
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.state(i) == "queued", 10))
        self.svc.perf.cancel()
        self.assertTrue(wait_for(lambda: self.status() == "cancelled" and self.state(i) == "processing"
                                 and self.svc._wk is not None, 20))

    def test_user_pause_all_survives_the_check(self):
        i = self.listening_item()
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(wait_for(lambda: self.svc._wk is None and self.state(i) == "queued", 10))
        self.svc.pause_all()                                               # the user pauses during the check
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "done", 10))
        self.assertTrue(self.svc.get_state()["paused_all"])
        self.assertFalse(wait_for(lambda: self.svc._wk is not None, 0.6))  # still paused afterwards
        self.assertEqual(self.state(i), "queued")
        self.svc.resume_all()
        self.assertTrue(wait_for(lambda: self.state(i) == "processing" and self.svc._wk is not None, 20))

    def test_hold_does_not_unpause_a_paused_all_queue(self):
        self.svc.pause_all()
        i = self.add("http://h/q", "listen")
        self.assertTrue(self.svc.perf.start()["ok"])
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "done", 10))
        self.assertFalse(wait_for(lambda: self.svc._wk is not None, 0.6))
        self.assertTrue(self.svc.get_state()["paused_all"])
        self.assertIn(self.state(i), ("queued", "waiting", "downloading"))


class SummaryTests(PerfTestBase):
    def saved(self):
        P.save(sample_profile(), self.profile_path)

    def test_existing_profile_is_done_and_not_prompted(self):
        self.saved()
        s = self.perf_state()
        self.assertEqual(s["status"], "done")
        self.assertFalse(s["prompt"])
        self.assertTrue(s["can_start"])
        self.assertEqual(s["profile"]["stale"], [])
        self.assertEqual(s["profile"]["settings_mismatch"], [])
        rows = {(r["height"], r["mode"]): r for r in s["profile"]["matrix"]["60"]}
        self.assertEqual(set(rows), {(h, m) for h in (480, 720, 1080) for m in ("off", "auto", "full")})
        self.assertEqual(s["profile"]["recommend"], json.loads(json.dumps(P.recommend(sample_profile(), 30))))

    def test_stale_and_settings_mismatch_follow_current_machine_and_settings(self):
        self.saved()
        self.fp.update(driver="999.0", gpu="RTX 9090")
        self.cur.update(detection_model="v4-accurate", max_clip_length=240)
        prof = self.perf_state()["profile"]
        self.assertEqual(sorted(prof["stale"]), ["driver: 617.42 -> 999.0", "gpu: RTX 5060 -> RTX 9090"])
        self.assertEqual(prof["settings_mismatch"], ["量測用 v4-fast，目前設定 v4-accurate",
                                                     "量測用 clip 長度 180，目前設定 240"])
        self.fp.update(driver="617.42", gpu="RTX 5060")                      # recomputed when things change back
        self.cur.update(detection_model="v4-fast", max_clip_length=180)
        prof = self.perf_state()["profile"]
        self.assertEqual((prof["stale"], prof["settings_mismatch"]), ([], []))

    def test_a_measured_alternative_detector_is_no_mismatch_and_drives_the_table(self):
        p = sample_profile()
        p["restore_alt"] = {"v4-accurate": {h: {"d0_fps": e["d0_fps"] / 2} for h, e in p["restore"].items()}}
        p["detail"] = {"measure_clip_length": 180}
        P.save(p, self.profile_path)
        base = self.perf_state()["profile"]
        self.cur.update(detection_model="v4-accurate", max_clip_length=180)
        prof = self.perf_state()["profile"]
        self.assertEqual(prof["settings_mismatch"], [])
        cell = lambda s: next(r for r in s["matrix"]["30"] if (r["height"], r["mode"]) == (720, "off"))
        self.assertLess(cell(prof)["ratio_typical"], cell(base)["ratio_typical"])   # the slower model's numbers
        # both models are listed so the panel can switch between them; the top-level table is the one in use
        main = p["settings"]["detector"]
        self.assertEqual([e["detector"] for e in prof["by_detector"]], [main, "v4-accurate"])
        self.assertEqual((prof["detector"], prof["current_detector"]), ("v4-accurate", "v4-accurate"))
        by = {e["detector"]: e for e in prof["by_detector"]}
        self.assertEqual(by["v4-accurate"]["matrix"], prof["matrix"])
        self.assertEqual(by[main]["matrix"], base["matrix"])
        self.assertIn("summary", by[main]["recommend"])

    def test_stage_labels(self):
        label = perf_runner._stage_label
        self.assertEqual(label("restore 720,480,1080p d=0,1 ..."), "量測修復速度 720,480,1080p")
        self.assertEqual(label("detector v4-fast 720,480,1080p d=0 ..."), "量測偵測模型 v4-fast")
        self.assertIsNone(label("restore done in 65 s (reference estimate 66 s)"))
        self.assertIsNone(label("child restore_720 pid=1"))

    def test_summary_is_cached_until_profile_or_settings_change(self):
        self.saved()
        first = self.perf_state()["profile"]
        self.assertIs(self.perf_state()["profile"], first)
        other = sample_profile()
        other["restore"]["720"]["d1_fps"] = 10.0
        P.save(other, self.profile_path)
        self.assertIsNot(self.perf_state()["profile"], first)

    def test_corrupt_profile_counts_as_missing(self):
        self.profile_path.parent.mkdir(parents=True)
        self.profile_path.write_text("{not json", encoding="utf-8")
        s = self.perf_state()
        self.assertEqual((s["status"], s["profile"], s["prompt"]), ("none", None, True))


class PromptTests(PerfTestBase):
    def test_dismiss_prompt_persists(self):
        self.assertTrue(self.perf_state()["prompt"])
        self.assertTrue(self.svc.perf.dismiss_prompt()["ok"])
        self.assertFalse(self.perf_state()["prompt"])
        again = perf_runner.PerfRunner(settings_provider=lambda: self.cur, is_player_busy=lambda: False,
                                       profile_path=self.profile_path, fingerprint_provider=lambda: self.fp)
        self.assertFalse(again.snapshot()["prompt"])
        other = perf_runner.PerfRunner(settings_provider=lambda: self.cur, is_player_busy=lambda: False,
                                       profile_path=self.tmp / "elsewhere" / "p.json")
        self.assertTrue(other.snapshot()["prompt"])


class ApplyTests(PerfTestBase):
    def setUp(self):
        super().setUp()
        self.settings_file = self.tmp / "player" / "oh_settings.json"

    def test_apply_without_profile(self):
        r = self.svc.perf.apply(self.settings_file)
        self.assertFalse(r["ok"])
        self.assertFalse(self.settings_file.exists())

    def test_apply_writes_interpolation_from_recommendation(self):
        P.save(sample_profile(), self.profile_path)
        rec = P.recommend(sample_profile(), 30)
        self.settings_file.parent.mkdir(parents=True)
        self.settings_file.write_text(json.dumps({"max_clip_length": 120, "volume": 55.0}), encoding="utf-8")
        want = "auto" if rec["auto_interp"] else "off"
        plan = self.perf_state()["apply_plan"]
        self.assertEqual([(c["key"], c["from"], c["to"]) for c in plan["changes"]],
                         [("interpolation", "off", want), ("restoration_fps", "original", rec["restoration_fps"])])
        r = self.svc.perf.apply(self.settings_file)
        self.assertTrue(r["ok"], r)
        saved = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["interpolation"], want)
        self.assertEqual((saved["max_clip_length"], saved["volume"]), (120, 55.0))   # nothing else touched
        self.assertIn("播放器補幀「自動」" if want == "auto" else "沒有變更", r["message"])

    def reloads(self):
        return [p for p in self.players if p.get("cmd") == "settings_reload"]

    def test_apply_tells_a_running_player_to_reload_the_file_once_it_is_written(self):
        P.save(rtx5060_accurate(), self.profile_path)
        self.cur.update(detection_model="v4-accurate", max_clip_length=300)
        self.player_reply = {"ok": True}
        seen = []
        real = self.svc._send_to_player
        self.svc._send_to_player = lambda p: seen.append((p, self.settings_file.exists())) or real(p)
        self.assertTrue(self.svc.perf.apply(self.settings_file)["ok"])
        self.assertEqual(seen, [({"cmd": "settings_reload"}, True)])

    def test_apply_without_changes_or_without_a_player_is_quiet(self):
        slow = sample_profile()
        for h in slow["restore"].values():
            h["d0_fps"] = h["d1_fps"] = 5.0
        P.save(slow, self.profile_path)
        self.assertTrue(self.svc.perf.apply(self.settings_file)["ok"])           # nothing to write: nothing to reload
        self.assertEqual(self.reloads(), [])
        P.save(rtx5060_accurate(), self.profile_path)
        self.cur.update(detection_model="v4-accurate", max_clip_length=300)
        self.player_reply = None                                                 # no player running
        self.assertTrue(self.svc.perf.apply(self.settings_file)["ok"])
        self.assertEqual(len(self.reloads()), 1)

    def test_apply_off_when_queue_only_changes_nothing(self):
        slow = sample_profile()
        for h in slow["restore"].values():
            h["d0_fps"] = h["d1_fps"] = 5.0
        P.save(slow, self.profile_path)
        self.assertTrue(P.recommend(slow, 30)["use_queue"])
        r = self.svc.perf.apply(self.settings_file)
        self.assertTrue(r["ok"])
        self.assertEqual(r["changed"], [])                      # the player default is already off
        self.assertFalse(self.settings_file.exists())
        self.assertIn("沒有變更", r["message"])

    def test_plan_switches_to_the_accurate_model_when_no_verdict_changes(self):
        P.save(rtx5060_accurate(), self.profile_path)
        self.cur.update(detection_model="v4-accurate", max_clip_length=300)
        changes = self.perf_state()["apply_plan"]["changes"]
        self.assertEqual([(c["key"], c["from"], c["to"]) for c in changes],
                         [("detection_model", "v4-fast", "v4-accurate"), ("interpolation", "off", "auto"),
                          ("restoration_fps", "original", "30")])     # 60 fps only plays at 30
        self.assertTrue(all(c["reason"] for c in changes))
        r = self.svc.perf.apply(self.settings_file)
        self.assertEqual(r["changed"], ["detection_model", "interpolation", "restoration_fps"])
        saved = json.loads(self.settings_file.read_text(encoding="utf-8"))
        self.assertEqual((saved["detection_model"], saved["interpolation"], saved["restoration_fps"]),
                         ("v4-accurate", "auto", "30"))
        self.assertIn("播放器修復幀率「最高 30fps」", r["message"])
        # applied: the plan now changes nothing
        self.assertTrue(all(c["from"] == c["to"] for c in self.perf_state()["apply_plan"]["changes"]))

    def test_plan_keeps_the_model_when_a_verdict_changes(self):
        p = two_detectors()
        p.update(fingerprint=sample_profile()["fingerprint"], settings={**p["settings"], "clip_length": 180})
        p["detail"] = {"measure_clip_length": 180}
        P.save(p, self.profile_path)
        state = self.perf_state()
        self.assertIsNone(state["profile"]["detector_advice"]["detector"])
        self.assertTrue(state["profile"]["detector_advice"]["differs"])
        self.assertEqual([c["key"] for c in state["apply_plan"]["changes"]], ["interpolation", "restoration_fps"])


class ApiTests(PerfTestBase):
    def test_api_methods_delegate(self):
        from oh_streamer.queue import queue_app
        api = queue_app.Api(types.SimpleNamespace(service=self.svc))
        self.assertTrue(api.perf_start("quick")["ok"])
        self.assertTrue(self.bench.started.wait(5))
        self.assertEqual(self.bench.calls[0]["mode"], "quick")
        self.assertTrue(api.get_state()["perf"]["status"] == "running")
        self.assertTrue(api.perf_cancel()["ok"])
        self.assertTrue(wait_for(lambda: api.get_state()["perf"]["status"] == "cancelled", 10))
        self.assertTrue(api.perf_dismiss_prompt()["ok"])
        self.assertFalse(api.get_state()["perf"]["prompt"])
        self.assertFalse(api.perf_apply()["ok"])             # no profile: nothing is written (real settings untouched)


def slow_encode_profile():
    p = sample_profile()
    p["encode"]["hevc_nvenc"]["720"]["fps"] = 60.0          # encode-bound: compare "encode" visibly costs time
    return p


class EtaTests(PerfTestBase):
    """eta_estimate_s: the profile's prior for queued / processing items without a worker ETA."""

    def setUp(self):
        super().setUp()
        self.facts = {"height": 720, "fps": 30.0, "duration": 600.0}
        self.probes = []
        self.svc._eta_probe = lambda path: (self.probes.append(path) or (dict(self.facts) if self.facts else None))
        real = QS.platform_support.capabilities()
        patcher = mock.patch.object(QS.platform_support, "capabilities", lambda: dict(real, rife=True))
        patcher.start()
        self.addCleanup(patcher.stop)

    def eta(self, item_id):
        return next(i for i in self.svc.get_state()["items"] if i["id"] == item_id)["eta_estimate_s"]

    def wait_eta(self, item_id):
        self.assertTrue(wait_for(lambda: self.eta(item_id) is not None, 5))
        return self.eta(item_id)

    def want(self, prof, **kw):
        args = dict(duration_s=600.0, height=720, src_fps=30.0, mode="full", compare=None)
        args.update(kw)
        return P.job_seconds(prof, **args)

    def test_none_without_profile(self):
        i = self.ready()
        self.assertIsNone(self.eta(i))
        # get_state() also probes the source for the duration / source_fps display, so a probe is expected here;
        # the ETA prior itself must stay None (and the probe is not repeated per state push)
        self.assertLessEqual(len(self.probes), 1)
        json.dumps(self.svc.get_state())

    def test_queued_item_gets_the_prior(self):
        P.save(sample_profile(), self.profile_path)
        i = self.ready()
        self.assertAlmostEqual(self.wait_eta(i), self.want(sample_profile()))
        self.assertEqual(len(self.probes), 1)
        self.eta(i), self.eta(i)
        self.assertEqual(len(self.probes), 1)                   # probed once per item + source

    def test_interpolation_setting_selects_the_mode(self):
        P.save(sample_profile(), self.profile_path)
        i = self.ready()
        full = self.wait_eta(i)
        self.svc.store.set_settings(interpolation="off")
        off = self.eta(i)
        self.assertAlmostEqual(off, self.want(sample_profile(), mode="off"))
        self.assertLess(off, full)

    def test_compare_encode_costs_but_copy_does_not(self):
        P.save(slow_encode_profile(), self.profile_path)
        self.svc.store.set_settings(interpolation="off")
        i = self.ready()
        plain = self.wait_eta(i)
        self.svc.store.set_settings(save_compare=True, compare_original="copy")
        self.assertAlmostEqual(self.eta(i), plain)
        self.svc.store.set_settings(save_compare=True, compare_original="encode")
        enc = self.eta(i)
        self.assertAlmostEqual(enc, self.want(slow_encode_profile(), mode="off", compare="encode"))
        self.assertGreater(enc, plain * 1.5)

    def test_missing_source_facts_give_none(self):
        P.save(sample_profile(), self.profile_path)
        self.facts = None
        i = self.ready()
        self.assertFalse(wait_for(lambda: self.eta(i) is not None, 0.3))
        self.assertTrue(self.probes)

    def test_only_queued_and_processing_without_worker_eta(self):
        P.save(sample_profile(), self.profile_path)
        base = dict(source_path="x.mp4", source_status="complete")

        def mk(id_, state, **kw):
            return queue_store.Item(id=id_, state=state, **base, **kw)
        items = [mk("q", "queued"),
                 mk("p0", "processing", progress={"frame": 0, "frames": 18000, "eta": None}),
                 mk("p1", "processing", progress={"frame": 9000, "frames": 18000, "eta": None}),
                 mk("p2", "processing", progress={"frame": 9000, "frames": 18000, "eta": 120.0}),
                 mk("d", "done"), mk("w", "waiting"), mk("e", "failed"), mk("z", "paused")]
        for it in items:
            self.svc._eta_facts[it.id] = ((it.source_path, it.source_status), dict(self.facts), time.monotonic())
        est = self.svc._eta_estimates(items, self.svc.store.get_settings())
        self.assertEqual(sorted(est), ["p0", "p1", "q"])
        full = self.want(sample_profile())
        self.assertAlmostEqual(est["q"], full)
        self.assertAlmostEqual(est["p0"], full)
        self.assertAlmostEqual(est["p1"], full / 2)                  # half the frames are left

    def test_processing_item_uses_the_settings_it_was_built_with(self):
        P.save(slow_encode_profile(), self.profile_path)
        self.svc.store.set_settings(interpolation="full", save_compare=False)
        it = queue_store.Item(id="p", state="processing", source_path="x.mp4", source_status="complete",
                              settings={"compare": "on", "compare_original": "encode", "interpolation": "off"})
        self.svc._eta_facts["p"] = (("x.mp4", "complete"), dict(self.facts), time.monotonic())
        est = self.svc._eta_estimates([it], self.svc.store.get_settings())["p"]
        self.assertAlmostEqual(est, self.want(slow_encode_profile(), mode="off", compare="encode"))

    def test_cached_until_inputs_or_profile_change(self):
        P.save(sample_profile(), self.profile_path)
        i = self.ready()
        first = self.wait_eta(i)
        entry = self.svc._eta_cache[i]
        self.eta(i)
        self.assertIs(self.svc._eta_cache[i], entry)
        self.assertTrue(self.svc.perf.record_observation(mode="full", height=720, observed_fps=20.0, density=0.7))
        self.assertNotAlmostEqual(self.eta(i), first)                # calibration moved the prior
        self.assertIsNot(self.svc._eta_cache[i], entry)

    def test_hls_probe_is_read_from_the_handle_metadata(self):
        P.save(sample_profile(), self.profile_path)
        i = self.ready()
        handle = types.SimpleNamespace(metadata={"streams": [{"codec_type": "video", "width": 1920, "height": 1080,
                                                              "r_frame_rate": "60000/1001"}],
                                                 "format": {"duration": "300.0"}})
        self.svc._sources[i] = handle
        self.assertAlmostEqual(self.eta(i), P.job_seconds(sample_profile(), duration_s=300.0, height=1080,
                                                          src_fps=60000 / 1001, mode="full"))
        self.assertEqual(self.probes, [])

    def test_video_facts_parsing(self):
        ok = {"streams": [{"codec_type": "audio"}, {"codec_type": "video", "height": 720, "r_frame_rate": "30/1",
                                                    "duration": "12.5"}], "format": {}}
        self.assertEqual(QS.video_facts(ok), {"height": 720, "fps": 30.0, "duration": 12.5})
        for bad in (None, {}, {"streams": [{"codec_type": "video", "height": 0, "r_frame_rate": "30/1"}],
                               "format": {"duration": "5"}},
                    {"streams": [{"codec_type": "video", "height": 720, "r_frame_rate": "0/0"}],
                     "format": {"duration": "5"}},
                    {"streams": [{"codec_type": "video", "height": 720, "r_frame_rate": "30/1"}], "format": {}}):
            self.assertIsNone(QS.video_facts(bad))


OBS = {"type": "perf_obs", "mode": "full", "height": 720, "src_fps": 30.0, "fps": 25.0, "density": 0.7, "frames": 3000}


class ObservationTests(PerfTestBase):
    def calibration(self):
        return (P.load(self.profile_path) or {}).get("calibration") or {}

    def test_recorded_and_saved_and_summary_refreshed(self):
        P.save(sample_profile(), self.profile_path)
        before = self.perf_state()["profile"]
        self.assertFalse(any(r["calibrated"] for r in before["matrix"]["30"]))
        self.svc._record_perf_obs(OBS)
        cal = self.calibration()
        self.assertEqual(list(cal), ["full:720"])
        self.assertEqual(cal["full:720"]["n"], 1)
        expect = P.record_observation(sample_profile(), mode="full", height=720, observed_fps=25.0, density=0.7)
        self.assertAlmostEqual(cal["full:720"]["factor"], expect["calibration"]["full:720"]["factor"])
        after = self.perf_state()["profile"]
        self.assertIsNot(after, before)
        row = next(r for r in after["matrix"]["30"] if (r["height"], r["mode"]) == (720, "full"))
        self.assertTrue(row["calibrated"])
        self.svc._record_perf_obs(dict(OBS, mode="off", fps=60.0))
        self.assertEqual(sorted(self.calibration()), ["full:720", "off:720"])

    def test_ignored_without_profile(self):
        self.svc._record_perf_obs(OBS)
        self.assertFalse(self.profile_path.exists())
        self.assertFalse(self.svc.perf.record_observation(mode="full", height=720, observed_fps=25.0))

    def test_ignored_while_the_check_holds_the_machine(self):
        P.save(sample_profile(), self.profile_path)
        self.assertTrue(self.svc.perf.start()["ok"])
        self.assertTrue(self.bench.started.wait(5))
        self.assertTrue(self.svc.perf.holding)
        self.svc._record_perf_obs(OBS)
        self.assertEqual(self.calibration(), {})
        self.bench.gate.set()
        self.assertTrue(wait_for(lambda: self.status() == "done", 10))

    def test_ignored_for_a_stale_profile(self):
        P.save(sample_profile(), self.profile_path)
        self.fp.update(gpu="RTX 9090")
        self.svc._record_perf_obs(OBS)
        self.assertEqual(self.calibration(), {})

    def test_ignored_while_the_player_is_busy(self):
        P.save(sample_profile(), self.profile_path)
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.svc._record_perf_obs(OBS)
        self.assertEqual(self.calibration(), {})

    def test_ignored_while_the_player_plays_anything(self):
        P.save(sample_profile(), self.profile_path)
        self.svc._on_message({"cmd": "player_playing", "playing": True, "pid": 1})
        self.assertFalse(self.svc.is_player_busy())          # playing a file does not hold the queue back
        self.svc._record_perf_obs(OBS)
        self.assertEqual(self.calibration(), {})

    def test_playback_inside_the_observed_window_spoils_it(self):
        P.save(sample_profile(), self.profile_path)
        now = [1000.0]
        self.svc._clock = lambda: now[0]
        window = OBS["frames"] / OBS["fps"]                    # 120 s
        for _ in range(6):                                     # playing from 1000, heartbeat every 10 s
            self.svc._on_message({"cmd": "player_playing", "playing": True, "pid": 1})
            now[0] += 10
        now[0] = 1050
        self.svc._on_message({"cmd": "player_playing", "playing": False, "pid": 1})   # stopped at 1050
        now[0] = 1050 + window - 1                             # the window began before the stop
        self.svc._record_perf_obs(OBS)
        self.assertEqual(self.calibration(), {})
        now[0] = 1050 + window + 30                            # stop + heartbeat margin before the window
        self.svc._record_perf_obs(OBS)
        self.assertEqual(list(self.calibration()), ["full:720"])

    def test_bad_messages_are_ignored(self):
        P.save(sample_profile(), self.profile_path)
        for bad in (dict(OBS, mode="auto"), dict(OBS, mode=None), dict(OBS, height=0), dict(OBS, height="x"),
                    dict(OBS, fps=0), dict(OBS, fps=float("nan")), dict(OBS, fps=None),
                    dict(OBS, fps=1000.0)):                    # implausible vs the prediction: calibration refuses it
            self.svc._record_perf_obs(bad)
        self.assertEqual(self.calibration(), {})
        self.svc._record_perf_obs(dict(OBS, density=float("nan")))     # a bad density only falls back to typical
        self.assertEqual(list(self.calibration()), ["full:720"])

    def test_worker_stdout_message_reaches_the_profile(self):
        P.save(sample_profile(), self.profile_path)
        line = json.dumps(OBS)
        self.worker_py.write_text(FAKE_WORKER.replace(
            'emit({"type": "progress"', "emit(%s)\nemit({\"type\": \"progress\"" % line, 1), encoding="utf-8")
        i = self.add("http://h/obs")
        self.assertTrue(wait_for(lambda: self.state(i) == "done", 20))
        self.assertTrue(wait_for(lambda: "full:720" in self.calibration(), 5))

    def test_concurrent_observations_all_count(self):
        P.save(sample_profile(), self.profile_path)
        threads = [threading.Thread(target=self.svc._record_perf_obs, args=(OBS,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.calibration()["full:720"]["n"], 8)
