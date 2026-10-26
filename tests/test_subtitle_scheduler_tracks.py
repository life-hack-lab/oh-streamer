"""M6a: comparison subtitles as extra tracks of the same slice plan, scheduled while downloading (design 4.8,
6.1 M1 / M6, 8 D4). No network, no GPU: the M4b fake cloud provider / fake local models, one fake clock."""

from __future__ import annotations

import contextlib
import json
import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import unittest

from oh_streamer.player import subtitle_compare as pc
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.queue.subtitle_view import subtitle_view
from oh_streamer.subtitles import configuration
from oh_streamer.subtitles import slice_plan as sp
from tests.test_subtitle_scheduler_fallback import Base, cloud_row, err, frozen, local_row


def track_plan(track, asr, mt, *, ready=True, reasons=()):
    plan = frozen(asr, mt)
    plan.update(track=track, combo_id=track, label="比較 " + track, combo=None, ready=ready,
                reasons=list(reasons), errors=[{"code": "x"}] if not ready else [])
    return plan


class TrackBase(Base):
    TOTAL = 2400.0

    def setUp(self):
        super().setUp()
        self.chost.total = self.TOTAL
        self.lhost.source_state = lambda: (True, self.chost.limit, self.TOTAL)
        self.GROQ2 = cloud_row("groq2-asr")
        self.LLM2 = cloud_row("llm2-mt", "translation")
        self.LLM3 = cloud_row("llm3-mt", "translation")
        self.plan = frozen(self.GROQ, self.LLM)
        self.tracks = {}

    def cloud_ex(self, **kw):
        kw.setdefault("track_plans_provider", lambda: self.tracks)
        return super().cloud_ex(**kw)

    def local_ex(self, **kw):
        kw.setdefault("track_plans_provider", lambda: self.tracks)
        return super().local_ex(**kw)

    def unit(self, plan, i, stage, track):
        return plan.slices[i].get(stage, track)

    def download(self, step=300.0, check=None):
        """Grow the readable prefix by `step` seconds per round and run the cloud executor each round."""
        limit = 0.0
        while True:
            limit += step
            self.chost.limit = None if limit >= self.TOTAL else limit
            res = self.cloud_ex().run()
            if check:
                check(self.disk(), self.chost.limit)
            if self.chost.limit is None:
                return res


class DownloadOrderTests(TrackBase):
    def setUp(self):
        super().setUp()
        self.tracks = {"same": track_plan("same", self.GROQ, self.LLM2),
                       "diff": track_plan("diff", self.GROQ2, self.LLM3)}

    def test_primary_first_comparison_after_600s_lead_while_downloading(self):
        started_before_end = []

        def check(plan, limit):
            self.assertEqual(plan.tracks, ["primary", "same", "diff"])
            lead = sp.covered_end(plan, 0.0, "primary")
            primary_complete = not plan.open and all(s.all_done("primary") for s in plan.slices)
            for s in plan.slices:
                for track in ("same", "diff"):
                    if any(self.unit(plan, s.index, st, track).status != sp.PENDING for st in sp.STAGES):
                        # a comparison unit only starts on a slice the primary finished, after a 600 s lead
                        self.assertTrue(s.all_done("primary"))
                        self.assertTrue(primary_complete or lead >= 600.0, (lead, s.index, track))
                        if limit is not None:
                            started_before_end.append((limit, track))

        res = self.download(check=check)
        self.assertEqual(res.reason, "complete")
        plan = self.disk()
        for track in ("primary", "same", "diff"):
            self.assertTrue(all(s.all_done(track) for s in plan.slices), track)
        # D4: the comparison tracks started while the file was still downloading
        self.assertEqual({t for _l, t in started_before_end}, {"same", "diff"})

        n = len(plan.slices)
        self.assertGreater(n, 2)
        # same ASR as the primary: one model call per slice for both tracks (shared cache entry)
        self.assertEqual(self.cloud.count("asr", "groq-asr"), n)
        for s in plan.slices:
            self.assertEqual(s.get("asr", "same").result_key, s.get("asr", "primary").result_key)
            self.assertNotEqual(s.get("asr", "diff").result_key, s.get("asr", "primary").result_key)
            self.assertEqual(s.get("asr", "diff").candidate, "groq2-asr")
            self.assertEqual(s.get("mt", "same").candidate, "llm2-mt")
        # different ASR: its own run per slice; every track translated with its own model
        self.assertEqual(self.cloud.count("asr", "groq2-asr"), n)
        for cid in ("llm-mt", "llm2-mt", "llm3-mt"):
            self.assertGreaterEqual(self.cloud.count("mt", cid), n)

    def test_results_kept_per_track_in_plan_store(self):
        ex = self.cloud_ex()
        store = ss.PlanStore.acquire(self.folder)      # keep the store alive after run() releases it
        try:
            res = ex.run()
            self.assertEqual(res.reason, "complete")
            n = len(store.plan.slices)
            for track, prefix in (("primary", "譯"), ("same", "譯"), ("diff", "譯")):
                self.assertEqual(sorted(store.rows(track)), list(range(n)), track)
                self.assertTrue(all(r["zh"].startswith(prefix) for rows in store.rows(track).values() for r in rows))
            self.assertIs(store.rows("primary"), store.mt_rows)
            self.assertEqual(store.track_labels, {"same": "比較 same", "diff": "比較 diff"})
            self.assertEqual(store.track_errors, {})
            # the primary publication is unchanged: only the primary rows are published
            published = res.publication
            self.assertIsNotNone(published)
        finally:
            store.release()

    def test_restart_reconciles_every_track_from_cache(self):
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        calls = len(self.cloud.calls)
        ss.PlanStore._registry.clear()
        store = ss.PlanStore.acquire(self.folder)
        try:
            res = self.cloud_ex().run()
            self.assertEqual(res.reason, "complete")
            self.assertEqual(len(self.cloud.calls), calls)            # nothing redone
            n = len(store.plan.slices)
            self.assertEqual(sorted(store.rows("diff")), list(range(n)))
        finally:
            store.release()


class SharedAsrTests(TrackBase):
    def test_second_track_on_same_asr_hits_cache_without_model_call(self):
        self.chost.limit = None
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        n = len(self.disk().slices)
        asr_calls = self.cloud.count("asr")
        self.assertEqual(asr_calls, n)
        self.tracks = {"same": track_plan("same", self.GROQ, self.LLM2)}       # row added afterwards
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        self.assertEqual(self.cloud.count("asr"), asr_calls)                   # ASR shared: no new call
        self.assertEqual(self.cloud.count("mt", "llm2-mt") >= n, True)
        self.assertTrue(all(s.all_done("same") for s in self.disk().slices))

    def test_removed_track_drops_its_units_keeps_results(self):
        self.tracks = {"diff": track_plan("diff", self.GROQ2, self.LLM3)}
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        plan = self.disk()
        keys = [s.get("mt", "diff").result_key for s in plan.slices]
        self.tracks = {}
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        plan = self.disk()
        self.assertEqual(plan.tracks, ["primary"])
        self.assertTrue(all(not k.endswith("/diff") for s in plan.slices for k in s.tasks))
        self.assertTrue(all((self.folder / "cache" / "mt" / f"{k}.json").is_file() for k in keys))


class OwnerByTrackTests(TrackBase):
    def test_primary_cloud_comparison_local_sakura(self):
        self.chost.limit = None
        self.SAKURA2 = local_row("sakura-cmp", "translation", root=self.root)
        self.tracks = {"sak": track_plan("sak", self.GROQ, self.SAKURA2)}
        c = self.cloud_ex().run()
        plan = self.disk()
        self.assertTrue(all(s.all_done("primary") for s in plan.slices))
        # the comparison ASR (same Groq model) is cloud work: cache hit; its MT belongs to the local executor
        self.assertTrue(all(s.is_done("asr", "sak") for s in plan.slices))
        self.assertTrue(all(s.get("mt", "sak").status == sp.PENDING for s in plan.slices))
        self.assertEqual(c.reason, "idle")
        self.assertEqual(c.waiting_executor, "local")
        self.assertEqual(self.cloud.count("mt", "sakura-cmp"), 0)
        self.drive()
        plan = self.disk()
        self.assertTrue(all(s.get("mt", "sak").candidate == "sakura-cmp" and s.is_done("mt", "sak")
                            for s in plan.slices))
        self.assertTrue(all(s.get("mt", "primary").candidate == "llm-mt" for s in plan.slices))
        self.assertFalse(any(call[0] == "asr" for call in self.models.calls))      # no local ASR at all


class IsolationTests(TrackBase):
    def test_not_ready_track_gets_no_work_and_primary_completes(self):
        self.chost.limit = None
        self.tracks = {"bad": track_plan("bad", None, None, ready=False, reasons=["比較字幕暫不支援 ASR 校正"])}
        store = ss.PlanStore.acquire(self.folder)
        try:
            res = self.cloud_ex().run()
            self.assertEqual(res.reason, "complete")
            self.assertEqual(store.track_errors, {"bad": ["比較字幕暫不支援 ASR 校正"]})
        finally:
            store.release()
        plan = self.disk()
        self.assertEqual(plan.tracks, ["primary", "bad"])
        self.assertTrue(all(s.all_done("primary") for s in plan.slices))
        self.assertTrue(all(not k.endswith("/bad") for s in plan.slices for k in s.tasks))
        self.assertEqual(subtitle_view(plan, now=self.clock())["state"], "complete")

    def test_comparison_failure_only_affects_its_track(self):
        self.chost.limit = None
        self.tracks = {"cmp": track_plan("cmp", self.GROQ, self.LLM2)}
        self.cloud.errors["llm2-mt"] = [err("server")] * 200
        res = self.cloud_ex().run(wait=True)
        self.assertEqual(res.reason, "complete")
        plan = self.disk()
        self.assertTrue(all(s.all_done("primary") for s in plan.slices))
        self.assertTrue(all(s.get("mt", "cmp").status == sp.FAILED for s in plan.slices))
        view = subtitle_view(plan, now=self.clock())
        self.assertEqual(view["state"], "complete")
        self.assertIsNone(view["stuck"])
        self.assertEqual(view["slices_done"], len(plan.slices))
        self.assertEqual(view["failures_elsewhere"], len(plan.slices))
        status = sp.derive_status(plan, 0.0, self.clock())
        self.assertTrue(status.complete)
        self.assertFalse(status.stuck)


def combos_manifest(folder):
    return json.loads((folder / "combos" / "manifest.json").read_text(encoding="utf-8"))


def combo_entry(folder, track):
    return next((e for e in combos_manifest(folder)["combos"] if e["id"] == track), None)


def srt_cues(folder, entry, key="zh"):
    return pc.load_srt(folder / "combos" / entry["files"][key])


class ComboPublishTests(TrackBase):
    """M6b: every comparison track publishes <subtitles>/combos/<track>.zh-TW.srt + manifest.json as it grows."""

    def setUp(self):
        super().setUp()
        self.tracks = {"a": track_plan("a", self.GROQ, self.LLM2), "b": track_plan("b", self.GROQ2, self.LLM3)}
        self.tracks["b"]["combo"] = {"japanese_track": True}

    def test_manifest_and_srt_grow_while_downloading_then_done(self):
        seen = {"a": [], "b": []}

        def check(_plan, limit):
            if not (self.folder / "combos" / "manifest.json").is_file():
                return
            for track in seen:
                entry = combo_entry(self.folder, track)
                if entry is None:
                    continue
                cues = srt_cues(self.folder, entry)
                seen[track].append((limit, entry["status"], entry["cue_count"], len(cues)))
                self.assertEqual(entry["cue_count"], len(cues))
                self.assertEqual(entry["label"], "比較 " + track)

        res = self.download(check=check)
        self.assertEqual(res.reason, "complete")
        for track in ("a", "b"):
            history = seen[track]
            self.assertTrue(history, track)
            # first publication happened while the file was still downloading, as running
            self.assertIsNotNone(history[0][0])
            self.assertEqual(history[0][1], "running")
            counts = [c for _l, _s, c, _n in history]
            self.assertEqual(counts, sorted(counts))                  # only grows
            self.assertGreater(counts[-1], counts[0])
            self.assertEqual(history[-1][1], "done")
        manifest = combos_manifest(self.folder)
        self.assertEqual((manifest["format"], manifest["schema_version"]), ("oh_streamer.subtitle_combos", 1))
        self.assertEqual([e["id"] for e in manifest["combos"]], ["a", "b"])        # primary is not listed
        a, b = (combo_entry(self.folder, t) for t in ("a", "b"))
        self.assertEqual((a["status"], b["status"]), ("done", "done"))
        self.assertIsNone(a["files"]["ja"])
        self.assertEqual(b["files"]["ja"], "b.ja.srt")                              # japanese_track
        self.assertTrue((self.folder / "combos" / "b.ja.srt").is_file())
        self.assertEqual(a["stages"]["translation"]["status"], "done")
        self.assertEqual(a["stages"]["asr"]["slices_total"], len(self.disk().slices))
        self.assertIsNone(a["failed_stage"])
        # same combination as the primary: equal rows give the same cues
        primary_cues = pc.load_srt(self.folder / "sakura.zh-TW.srt")
        self.assertEqual(len(srt_cues(self.folder, a)), len(primary_cues))
        # the player reads it
        found = pc.load(self.root / "x.mkv", folder=self.folder / "combos")
        self.assertEqual([c.id for c in found.combos], ["a", "b"])

    def test_failed_track_and_not_ready_track_manifest(self):
        self.chost.limit = None
        self.tracks = {"cmp": track_plan("cmp", self.GROQ, self.LLM2),
                       "bad": track_plan("bad", None, None, ready=False, reasons=["比較字幕暫不支援 ASR 校正"])}
        self.cloud.errors["llm2-mt"] = [err("server")] * 200
        self.assertEqual(self.cloud_ex().run(wait=True).reason, "complete")
        cmp_, bad = combo_entry(self.folder, "cmp"), combo_entry(self.folder, "bad")
        self.assertEqual(cmp_["status"], "failed")
        self.assertEqual(cmp_["failed_stage"], "translation")
        self.assertIn("翻譯失敗", cmp_["error"])
        self.assertEqual(cmp_["stages"]["translation"]["status"], "failed")
        self.assertEqual(bad["status"], "failed")
        self.assertEqual(bad["failed_stage"], "config")
        self.assertEqual(bad["error"], "比較字幕暫不支援 ASR 校正")
        self.assertIsNone(bad["files"]["zh"])
        self.assertIsNone(pc.load(self.root / "x.mkv", folder=self.folder / "combos"))     # nothing usable
        self.assertEqual([e["id"] for e in combos_manifest(self.folder)["combos"]], ["cmp", "bad"])

    def test_two_executors_publish_without_corrupting_manifest(self):
        self.chost.limit = None
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        (self.folder / "combos" / "manifest.json").unlink()
        one, two = self.cloud_ex(), self.local_ex()
        self.assertTrue(one._prepare() and two._prepare())
        try:
            self.assertIs(one._publish_lock, two._publish_lock)
            errors, stop = [], threading.Event()

            def writer(ex, track):
                try:
                    for _ in range(40):
                        ex._publish_track(track)
                except Exception as exc:       # pragma: no cover
                    errors.append(exc)

            def reader():
                path = self.folder / "combos" / "manifest.json"
                while not stop.is_set():
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        pass
                    except PermissionError:
                        pass                    # Windows: replaced at this very moment
                    except (OSError, ValueError) as exc:
                        errors.append(exc)

            watcher = threading.Thread(target=reader)
            workers = [threading.Thread(target=writer, args=(one, "a")), threading.Thread(target=writer, args=(two, "b"))]
            watcher.start()
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            stop.set()
            watcher.join()
            self.assertEqual(errors, [])
            self.assertEqual([e["id"] for e in combos_manifest(self.folder)["combos"]], ["a", "b"])
        finally:
            one._release_store()
            two._release_store()


class LocalComparisonSentenceTests(TrackBase):
    SENTENCE = {"sentence_mode": False, "short_threshold": 2.5, "short_policy": "drop", "show_interjections": True}

    def setUp(self):
        super().setUp()
        self.sentences = []
        self.SAKURA2 = local_row("sakura-cmp", "translation", root=self.root)
        self.SAKURA3 = local_row("sakura-cmp2", "translation", root=self.root)
        self.tracks = {"sak": track_plan("sak", self.GROQ, self.SAKURA2),
                       "sak2": track_plan("sak2", self.GROQ, self.SAKURA3)}
        self.tracks["sak"]["combo"] = {"sentence": dict(self.SENTENCE)}
        outer = self

        @contextlib.contextmanager
        def session():
            def translate(ja_doc, workdir, on_batch=None, sentence=None):
                outer.sentences.append(sentence)
                return [{"start": r["start"], "end": r["end"], "text": "這是句子", "text_ja": r["text"]}
                        for r in ja_doc["segments"]]
            yield translate

        self.models.mt_session = session

    def local_ex(self, **kw):
        ex = super().local_ex(**kw)
        ex.deps.mt_sentence_option = True
        return ex

    def test_comparison_row_translates_with_its_own_sentence_options(self):
        self.chost.limit = None
        self.cloud_ex().run()
        self.drive()
        plan = self.disk()
        self.assertTrue(all(s.is_done("mt", "sak") for s in plan.slices))
        self.assertEqual(self.sentences.count(self.SENTENCE), len(plan.slices))
        # the manifest entry of a local comparison row is published too
        self.assertEqual(combo_entry(self.folder, "sak")["status"], "done")
        self.assertGreater(combo_entry(self.folder, "sak")["cue_count"], 0)

    def test_local_comparison_srt_has_content_and_survives_reopen(self):
        self.chost.limit = None
        self.cloud_ex().run()
        self.drive()
        entry = combo_entry(self.folder, "sak")
        self.assertEqual(entry["status"], "done")
        self.assertGreater(entry["cue_count"], 0)
        cues = len(srt_cues(self.folder, entry))
        self.assertGreater(cues, 0)
        self.assertTrue(all("這是句子" in c.text for c in srt_cues(self.folder, entry)))
        again = self.local_ex()                                   # a new executor reconciles from disk
        self.assertTrue(again._prepare())
        try:
            again._publish_tracks()
        finally:
            again._release_store()
        entry = combo_entry(self.folder, "sak")
        self.assertEqual(entry["cue_count"], cues)
        self.assertEqual(len(srt_cues(self.folder, entry)), cues)

    def test_cache_key_has_the_row_sentence_options_and_model(self):
        ex = self.local_ex()
        self.assertTrue(ex._prepare())
        try:
            cand = {t: ex._home_candidate(sp.STAGE_MT, t) for t in ("sak", "sak2")}
            primary_cand = ex._home_candidate(sp.STAGE_MT, "primary")
            self.assertEqual(ex._mt_params(primary_cand, "primary"), ex.deps.mt_params)     # primary unchanged
            keys = {"primary": ex._mt_key("d", primary_cand, "primary"), "sak": ex._mt_key("d", cand["sak"], "sak"),
                    "sak2": ex._mt_key("d", cand["sak2"], "sak2")}
            self.assertEqual(len(set(keys.values())), 3)
            params = ex._mt_params(cand["sak"], "sak")
            self.assertEqual(params["track_sentence"]["short_threshold"], 2.5)
            self.assertEqual(params["track_sentence"]["sentence_mode"], False)
            self.assertIn("track_mt", params)
            # same sentence options but another translation model: still another key
            self.tracks["sak2"]["combo"] = {"sentence": dict(self.SENTENCE)}
            self.assertNotEqual(ex._mt_key("d", cand["sak"], "sak"), ex._mt_key("d", cand["sak2"], "sak2"))
        finally:
            ex._release_store()


class TrackRuntimePlansTests(unittest.TestCase):
    PRIMARY_PLAN = {"models": {"asr": {"id": "a"}, "translation": {"id": "t"}}, "ready": True,
                    "processing": "smart_horizon"}

    def settings(self, *rows, contract=True):
        out = {"subtitle_configs": [{"id": "primary"}, *rows]}
        if contract:
            out.update(subtitle_runtime_contract=configuration.RUNTIME_CONTRACT, subtitle_resolved=self.PRIMARY_PLAN)
        return out

    def test_primary_is_runtime_plan_and_rows_follow_in_order(self):
        rows = [{"id": "kot", "label": "Kotoba", "asr": {"mode": "single", "model": "kotoba"},
                 "translation": {"profile": "sakura-v1"}},
                {"id": "dual", "label": "雙模型", "asr": {"mode": "dual"}, "translation": {"profile": "qwen3"}}]
        plans = configuration.track_runtime_plans(self.settings(*rows))
        self.assertEqual(list(plans), ["primary", "kot", "dual"])
        self.assertIs(plans["primary"], self.PRIMARY_PLAN)
        kot, dual = plans["kot"], plans["dual"]
        self.assertTrue(kot["ready"], kot["reasons"])
        self.assertEqual((kot["models"]["asr"]["id"], kot["models"]["translation"]["id"]), ("kotoba", "sakura-v1"))
        self.assertEqual(dual["models"]["asr"]["id"], "__dual__")
        self.assertEqual(kot["fallback"], {"asr": [], "translation": []})
        self.assertEqual((kot["label"], kot["combo_id"], kot["track"]), ("Kotoba", "kot", "kot"))
        self.assertEqual(kot["combo"]["id"], "kot")

    def test_unsupported_fields_make_only_that_row_not_ready(self):
        rows = [{"id": "corr", "label": "校正", "asr": {"mode": "single", "model": "kotoba"},
                 "correction": {"mode": "llm", "backend": "qwen3"}, "translation": {"profile": "sakura-v1"}},
                {"id": "dec", "label": "解碼", "asr": {"mode": "single", "model": "kotoba", "decode": {"beam_size": 9}},
                 "translation": {"profile": "sakura-v1"}},
                {"id": "grp", "label": "分段", "asr": {"mode": "diagnostic", "model": "kotoba", "segmentation": "grouped"},
                 "translation": {"profile": "sakura-v1"}},
                {"id": "unk", "label": "未知", "asr": {"mode": "single", "model": "nope"},
                 "translation": {"profile": "sakura-v1"}},
                {"id": "ok", "label": "好", "asr": {"mode": "single", "model": "turbo"},
                 "translation": {"profile": "sakura-v1"}}]
        plans = configuration.track_runtime_plans(self.settings(*rows))
        self.assertEqual(plans["primary"], self.PRIMARY_PLAN)
        self.assertIn("校正", " ".join(plans["corr"]["reasons"]))
        self.assertIn("解碼", " ".join(plans["dec"]["reasons"]))
        self.assertIn("分段", " ".join(plans["grp"]["reasons"]))
        self.assertFalse(plans["unk"]["ready"])
        self.assertTrue(plans["unk"]["reasons"])
        for tid in ("corr", "dec", "grp", "unk"):
            self.assertFalse(plans[tid]["ready"], tid)
        self.assertTrue(plans["ok"]["ready"], plans["ok"]["reasons"])     # turbo's default decode is fine

    def test_legacy_snapshot_and_disabled_comparison(self):
        self.assertEqual(configuration.track_runtime_plans({}), {"primary": None})
        settings = {"subtitle_comparison_enabled": False,
                    "subtitle_comparison_rows": [{"id": "x", "label": "X", "asr": {"mode": "dual"},
                                                  "translation": {"profile": "qwen3"}}]}
        self.assertEqual(list(configuration.track_runtime_plans(settings)), ["primary"])
        settings["subtitle_comparison_enabled"] = True
        self.assertEqual(list(configuration.track_runtime_plans(settings)), ["primary", "x"])


    def test_frozen_track_plans_win_over_the_rows_and_the_runtime_plan(self):
        """M7: a queued job keeps the plans it was queued with, even when its rows or plan change later."""
        rows = [{"id": "kot", "label": "Kotoba", "asr": {"mode": "single", "model": "kotoba"},
                 "translation": {"profile": "sakura-v1"}}]
        settings = self.settings(*rows)
        settings["subtitle_track_plans"] = {"primary": self.PRIMARY_PLAN, "old": {"track": "old", "ready": True}}
        plans = configuration.track_runtime_plans(settings)
        self.assertEqual(list(plans), ["primary", "old"], "not recomputed from subtitle_configs")
        self.assertIsNot(plans["old"], settings["subtitle_track_plans"]["old"], "a copy: callers cannot edit the freeze")
        plans["old"]["ready"] = False
        self.assertTrue(settings["subtitle_track_plans"]["old"]["ready"])
        # nothing frozen (or a frozen value without the primary entry): derived live as before
        for broken in (None, {}, {"x": 1}, "junk"):
            settings["subtitle_track_plans"] = broken
            self.assertEqual(list(configuration.track_runtime_plans(settings)), ["primary", "kot"])


if __name__ == "__main__":
    unittest.main()
