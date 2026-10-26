"""Offline reproduction of the cloud executor's horizon/idle restart loop (sample video A) - now FIXED.

Uses synthetic slice state matching the first sample video A busy-loop window.
No media, credentials, network requests, GPU work, or live queue writes.
Run: python tools/repro_cloud_executor_idle.py

Before the fix: the host counted 15 pending cloud units (ignoring the horizon) and started the executor, which
found nothing the selector would take, ended "idle" with wait_until=now, and the restart gate let it start again
at once. After the fix (asserted below): the host asks the executor's own selector (dispatchable_for) and starts
nothing, the executor classifies the held-back work as "horizon" (no wait_until, no sleeps), the restart gate
keeps the 5 s interval, and work resumes once playback advances below the low watermark.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.queue.subtitle_host import FinishedRun, SubtitleSchedulerMixin, dispatchable_for
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp


def main():
    now = [1000.0]
    sleeps = []
    publications = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    class Host(ss.CloudHost):
        context = sp.SchedulerContext(watched=True, play_pos=1701.824, restore_pos=3024.487)

        def scheduler_context(self):
            return self.context

        def source_state(self):
            return True, None, 7490.016

    cuts = [0, 90, 389.309, 689.309, 989.309, 1297.679, 1597.679,
            1897.679, 2211.092, 2515.199, 2815.199, 3115.199]
    cuts += list(range(3415, 7490, 300)) + [7490.016]
    plan = sp.SlicePlan.from_dict({
        "duration": 7490.016, "media_id": "offline-repro", "tracks": ["primary"],
        "strategy": "smart_horizon", "horizon_minutes": 30,
        "lazy": {"cuts": cuts, "closed": True},
        "slices": [{"index": i, "start": a, "end": b,
                    "tasks": {f"{stage}/primary": {"status": "done"}
                              for stage in sp.STAGES} if i < 11 else {}}
                   for i, (a, b) in enumerate(zip(cuts, cuts[1:]))],
    })
    deps = ss.CloudDeps(asr=None, translator=None,
                        asr_identity={"id": "offline-asr"}, mt_identity={"id": "offline-mt"})
    host = Host()
    scratch = Path(__file__).resolve().parents[1] / ".test-tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cloud-idle-repro-", dir=scratch) as tmp:
        ex = ss.CloudSliceExecutor(Path(tmp) / "subtitles", Path(tmp) / "unused.mp4",
                                   host=host, deps=deps, clock=lambda: now[0], sleep=sleep)
        ex.plan = plan
        ctx, plans, _ = ex._dispatch(None)
        selected = sp.select_next_task(plan, ctx, ex.EXECUTOR)
        idle = ex._idle_reason(None)
        owners = {track: fb.owner_function(frozen, now=now[0]) for track, frozen in plans.items()}
        host_pick = dispatchable_for(plan, host.context, owners, "cloud", now=now[0])
        # what the old host start check counted: owned pending units, horizon ignored
        pending = sum(1 for s in plan.slices if not s.is_done(sp.STAGE_ASR))
        # Only I/O boundaries are replaced. Dispatch, idle classification, launch,
        # spin limit and restart gate all execute the production implementation.
        with patch.object(ex, "_prepare", return_value=True), \
             patch.object(ex, "_publish", side_effect=lambda: publications.append("publish")), \
             patch.object(ex, "_publish_tracks"), \
             patch.object(ex, "_usage_run_start"), patch.object(ex, "_usage_run_end"):
            result = ex.run(wait=True)
        service = SubtitleSchedulerMixin()
        service._subtitle_results_for("offline")["cloud"] = FinishedRun(result, None, now[0])
        gate = service._subtitle_gate_open("offline", "cloud", now[0])
        gate_after_5s = service._subtitle_gate_open("offline", "cloud", now[0] + 5.0)
        # Control: advancing playback below the low watermark selects work again.
        host.context = dataclasses.replace(host.context, play_pos=2000.0)
        ready_ctx, _, _ = ex._dispatch(None)
        ready = sp.select_next_task(plan, ready_ctx, ex.EXECUTOR)
        host_ready = dispatchable_for(plan, host.context, owners, "cloud", now=now[0])
        report = {
            "buffer_ahead_s": round(3115.199 - 1701.824, 3),
            "pending_cloud_units_ignoring_horizon": pending,
            "host_dispatchable": dataclasses.asdict(host_pick) if host_pick else None,
            "selected_task": dataclasses.asdict(selected) if selected else None,
            "idle_reason": list(idle),
            "run_result": {"reason": result.reason, "wait_until": result.wait_until,
                           "waiting_executor": result.waiting_executor},
            "sleep_calls": sleeps, "simulated_wait_s": round(sum(sleeps), 6),
            "startup_publications": len(publications),
            "restart_gate_open_immediately": gate,
            "restart_gate_open_after_5s": gate_after_5s,
            "control_after_playback_advance": dataclasses.asdict(ready) if ready else None,
        }
        # fixed behaviour: work exists, but nothing is dispatchable now, and nobody pretends otherwise
        assert selected is None and host_pick is None and pending > 0
        assert tuple(idle) == ("horizon", None, None)
        assert result.reason == "horizon" and result.wait_until is None and not sleeps
        assert not gate and gate_after_5s
        assert ready is not None and host_ready is not None
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
