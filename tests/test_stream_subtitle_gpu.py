"""Tests for queue/stream_subtitle_gpu.py (T4.3): live-with-subtitles verdict, VRAM check, learned speed."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import math
import tempfile
import unittest
from pathlib import Path

from oh_streamer.perf import profile
from oh_streamer.queue import gpu_turns as gt
from oh_streamer.queue import stream_subtitle_gpu as ssg


class FormulaTests(unittest.TestCase):
    def test_effective_ratio(self):
        self.assertAlmostEqual(ssg.effective_ratio(2.0, 0.5, 0.0), 1.0)
        self.assertAlmostEqual(ssg.effective_ratio(2.0, 0.0, 0.0), 2.0)
        self.assertAlmostEqual(ssg.effective_ratio(1.0, 0.65, 0.35), 0.5)

    def test_effective_ratio_junk(self):
        for bad in (None, 0, 0.0, math.inf, math.nan, -1, "x"):
            self.assertIsNone(ssg.effective_ratio(bad, 0.65, 0.1))
        self.assertAlmostEqual(ssg.effective_ratio(2.0, None, math.nan), 2.0)

    def test_switch_cost(self):
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(gt.TurnConfig()), 90.0 / 600.0)
        cfg = gt.TurnConfig(batch_slices=1, subtitle_load_s=30, restore_restart_s=15)
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(cfg), 0.15)

    def test_switch_cost_of_media_turns_uses_the_turn_cap(self):
        cfg = gt.TurnConfig(media_turns=True)                  # (60 + 30) s per steady-state 300 s turn
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(cfg), 0.3)
        cfg = gt.TurnConfig(media_turns=True, turn_max_s=600.0, batch_slices=1)
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(cfg), 0.15)


class VerdictTests(unittest.TestCase):
    def base(self, rt, rh):
        level, pre = profile.level_for(rt, rh)
        return {"level": level, "ratio_typical": rt, "ratio_heavy": rh, "prebuffer_s": pre, "bottleneck": "restore"}

    def test_level_for_rules(self):
        self.assertEqual(profile.level_for(2.0, 1.1), ("live", 0.0))
        self.assertEqual(profile.level_for(1.0, 0.5)[0], "prebuffer")
        self.assertEqual(profile.level_for(0.99, 0.5), ("queue", None))
        self.assertEqual(profile.level_for(None, None), ("queue", None))
        self.assertEqual(profile.level_for(1.05, 1.05)[1], profile.MIN_PREBUFFER_S)
        self.assertEqual(profile.level_for(1.0, 0.01)[1], profile.MAX_PREBUFFER_S)

    def test_live_becomes_queue(self):
        v = ssg.verdict_with_subtitles(self.base(3.0, 2.0), 0.65, 0.15)
        self.assertEqual(v["level"], "queue")
        self.assertEqual(v["level_no_subtitles"], "live")
        self.assertIsNone(v["prebuffer_s"])
        self.assertAlmostEqual(v["ratio_heavy"], 1 / (1 / 2.0 + 0.8))

    def test_live_stays_live_for_a_fast_card(self):
        v = ssg.verdict_with_subtitles(self.base(20.0, 12.0), 0.65, 0.15)
        self.assertEqual(v["level"], "live")       # heavy 1/(1/12+0.8) = 1.12
        self.assertEqual(v["prebuffer_s"], 0.0)

    def test_prebuffer_with_seconds(self):
        v = ssg.verdict_with_subtitles(self.base(10.0, 3.0), 0.5, 0.1)
        rh = 1 / (1 / 3.0 + 0.6)                   # 1.07: heavy < 1.1, typical 1/(0.1+0.6) = 1.43 >= 1
        self.assertEqual(v["level"], "prebuffer")
        self.assertAlmostEqual(v["prebuffer_s"], max(profile.MIN_PREBUFFER_S,
                                                      profile.HEAVY_SEGMENT_S * (1 - rh) / rh))
        self.assertEqual(v["ratio_typical_no_subtitles"], 10.0)

    def test_unknown_ratios_queue(self):
        self.assertEqual(ssg.verdict_with_subtitles(self.base(None, None), 0.65, 0.1)["level"], "queue")

    def test_no_cost_keeps_level(self):
        self.assertEqual(ssg.verdict_with_subtitles(self.base(2.0, 1.2), 0, 0)["level"], "live")


class VramTests(unittest.TestCase):
    def test_need_and_fit(self):
        ok, need = ssg.vram_fits(8151)
        self.assertEqual(need, 1232.0 + 4801.0 + 1536.0)
        self.assertTrue(ok)
        self.assertFalse(ssg.vram_fits(6144)[0])

    def test_unknown_total_does_not_block(self):
        for total in (None, 0, math.nan):
            ok, need = ssg.vram_fits(total)
            self.assertTrue(ok)
            self.assertGreater(need, 0)

    def test_requirements_override(self):
        ok, need = ssg.vram_fits(6144, requirements={"asr_dual": 2000.0, "mt_sakura_7b": 2000.0})
        self.assertTrue(ok)
        self.assertEqual(need, 1232.0 + 2000.0 + 1536.0)


class RestoreVramMemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mem = ssg.RestoreVramMemory(Path(self._tmp.name) / "restore-vram.json")

    def test_path_sits_next_to_the_speed_file(self):
        self.assertEqual(ssg.default_restore_vram_path().parent, ssg.default_speed_path().parent)

    def test_unknown_card_has_no_value_and_falls_back_to_the_fixed_figure(self):
        self.assertIsNone(self.mem.get("gpu|8192"))
        self.assertEqual(ssg.held_restore_mib("gpu|8192", self.mem), (ssg.PAUSED_RESTORE_MIB, False))
        self.assertEqual(ssg.held_restore_mib("gpu|8192", None), (1232.0, False))

    def test_recent_larger_value_per_card(self):
        for mib in (1100.0, 1505.0, 1300.0):
            self.mem.record("a|8192", mib)
        self.mem.record("b|4096", 900.0)
        self.assertEqual(self.mem.get("a|8192"), 1505.0)
        self.assertEqual(self.mem.get("b|4096"), 900.0)
        self.assertEqual(self.mem.count("a|8192"), 3)
        self.assertEqual(ssg.held_restore_mib("a|8192", self.mem), (1505.0, True))

    def test_only_the_last_samples_count(self):
        self.mem.record("a|8192", 3000.0)
        for _ in range(ssg.RESTORE_VRAM_KEEP):
            self.mem.record("a|8192", 1000.0)
        self.assertEqual(self.mem.get("a|8192"), 1000.0)

    def test_junk_is_ignored_and_a_broken_file_is_empty(self):
        for bad in (None, 0, -5, math.nan, math.inf, "x", 10 ** 9):
            self.assertIsNone(self.mem.record("a|8192", bad))
        self.assertIsNone(self.mem.record("", 1000.0))
        self.assertIsNone(self.mem.get("a|8192"))
        Path(self.mem.path).write_text("{broken", encoding="utf-8")
        self.assertIsNone(self.mem.get("a|8192"))
        self.assertEqual(self.mem.record("a|8192", 1200.0), 1200.0)

    def test_learned_value_drives_vram_fits(self):
        self.mem.record("a|8192", 3000.0)
        held, learned = ssg.held_restore_mib("a|8192", self.mem)
        ok, need = ssg.vram_fits(8151, held)
        self.assertTrue(learned)
        self.assertEqual(need, 3000.0 + 4801.0 + 1536.0)
        self.assertFalse(ok)

    def test_a_failing_memory_falls_back(self):
        class Boom:
            def get(self, key):
                raise OSError("disk")
        self.assertEqual(ssg.held_restore_mib("a|8192", Boom()), (ssg.PAUSED_RESTORE_MIB, False))


class SwitchCostMemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mem = ssg.SwitchCostMemory(Path(self._tmp.name) / "costs.json")
        self.cfg = gh_cfg()

    def test_path_sits_next_to_the_speed_file(self):
        self.assertEqual(ssg.default_cost_path().parent, ssg.default_speed_path().parent)

    def test_first_sample_then_ewma_per_field_and_card(self):
        self.assertIsNone(self.mem.get("a|8192", ssg.COST_ASR_LOAD))
        self.assertEqual(self.mem.record("a|8192", ssg.COST_ASR_LOAD, 12.0), 12.0)
        self.assertAlmostEqual(self.mem.record("a|8192", ssg.COST_ASR_LOAD, 22.0), 0.3 * 22.0 + 0.7 * 12.0)
        self.mem.record("a|8192", ssg.COST_RESTART, 4.0)
        self.mem.record("b|4096", ssg.COST_ASR_LOAD, 30.0)
        self.assertEqual(self.mem.count("a|8192", ssg.COST_ASR_LOAD), 2)
        self.assertEqual(self.mem.get("a|8192", ssg.COST_RESTART), 4.0)
        self.assertEqual(self.mem.get("b|4096", ssg.COST_ASR_LOAD), 30.0)
        self.assertEqual(ssg.SwitchCostMemory(self.mem.path).get("b|4096", ssg.COST_ASR_LOAD), 30.0)   # persisted

    def test_junk_and_stuck_values_are_ignored(self):
        for bad in (0, -1, None, "x", float("nan"), float("inf"), True * 0, ssg.MAX_COST_S + 1):
            self.assertIsNone(self.mem.record("a|1", ssg.COST_MT_LOAD, bad))
        self.assertIsNone(self.mem.record("", ssg.COST_MT_LOAD, 5.0))
        self.assertIsNone(self.mem.get("a|1", ssg.COST_MT_LOAD))
        self.mem.path.write_text("not json", encoding="utf-8")
        self.assertIsNone(self.mem.get("a|1", ssg.COST_MT_LOAD))

    def test_learned_costs_fall_back_to_the_config_split(self):
        c = ssg.learned_costs("a|8192", self.mem, self.cfg)             # stream defaults 20 / 10, mt share 10
        self.assertEqual((c["asr_load_s"], c["mt_load_s"], c["restart_s"], c["learned"]), (10.0, 10.0, 10.0, False))
        self.assertEqual(ssg.switch_costs("a|8192", self.mem, self.cfg), (20.0, 10.0, False))
        self.assertEqual(ssg.switch_costs("a|8192", None, self.cfg), (20.0, 10.0, False))

    def test_learned_costs_replace_them_field_by_field(self):
        self.mem.record("a|8192", ssg.COST_ASR_LOAD, 14.0)
        self.mem.record("a|8192", ssg.COST_RESTART, 3.0)
        self.assertEqual(ssg.switch_costs("a|8192", self.mem, self.cfg), (14.0 + 10.0, 3.0, True))
        self.mem.record("a|8192", ssg.COST_MT_LOAD, 5.0)
        self.assertEqual(ssg.switch_costs("a|8192", self.mem, self.cfg), (19.0, 3.0, True))

    def test_the_mt_share_never_exceeds_the_load(self):
        cfg = gt.TurnConfig(subtitle_load_s=4.0)
        self.assertEqual(ssg.default_load_split(cfg), (0.0, 4.0))

    def test_a_failing_memory_falls_back(self):
        class Boom:
            def get(self, *a):
                raise OSError("disk")
        self.assertEqual(ssg.switch_costs("a|1", Boom(), self.cfg), (20.0, 10.0, False))

    def test_switch_cost_per_media_s_takes_the_learned_values(self):
        cfg = gt.TurnConfig(media_turns=True, subtitle_load_s=20.0, restore_restart_s=10.0)
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(cfg), 30.0 / 300.0)
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(cfg, 14.0, 3.0), 17.0 / 300.0)
        self.assertAlmostEqual(ssg.switch_cost_per_media_s(cfg, None, 3.0), 23.0 / 300.0)

    def test_cheaper_switching_raises_the_effective_ratio(self):
        cfg = gt.TurnConfig(media_turns=True, subtitle_load_s=60.0, restore_restart_s=30.0)
        dear = ssg.effective_ratio(2.0, 0.6, ssg.switch_cost_per_media_s(cfg))
        cheap = ssg.effective_ratio(2.0, 0.6, ssg.switch_cost_per_media_s(cfg, 14.0, 3.0))
        self.assertGreater(cheap, dear)


def gh_cfg():
    from oh_streamer.queue import subtitle_host_gpu as gh
    return gh.stream_turn_config(None)


class SpeedMemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sub" / "speed.json"
        self.mem = ssg.SubtitleSpeedMemory(self.path)

    def test_first_sample_then_ewma(self):
        self.assertIsNone(self.mem.get("k"))
        self.mem.record("k", 600.0, 600.0)
        self.assertAlmostEqual(self.mem.get("k"), 1.0)
        self.mem.record("k", 300.0, 600.0)
        self.assertAlmostEqual(self.mem.get("k"), 0.3 * 0.5 + 0.7 * 1.0)
        self.assertEqual(self.mem.count("k"), 2)

    def test_persists_and_keys_are_separate(self):
        self.mem.record("a", 100.0, 100.0)
        self.mem.record("b", 200.0, 100.0)
        again = ssg.SubtitleSpeedMemory(self.path)
        self.assertAlmostEqual(again.get("a"), 1.0)
        self.assertAlmostEqual(again.get("b"), 2.0)

    def test_ignores_junk(self):
        for wall, media in ((10, 29.9), (math.nan, 100), (100, math.inf), (0, 100), (-5, 100), (None, 100),
                            (5, "x")):
            self.assertIsNone(self.mem.record("k", wall, media))
        self.assertIsNone(self.mem.get("k"))
        self.assertFalse(self.path.exists())

    def test_corrupt_file_is_empty(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.mem.get("k"))
        self.mem.record("k", 100.0, 100.0)
        self.assertAlmostEqual(self.mem.get("k"), 1.0)
        self.assertIsInstance(json.loads(self.path.read_text(encoding="utf-8")), dict)

    def test_speed_lookup_prefers_learned(self):
        cfg = gt.TurnConfig()
        self.assertEqual(ssg.subtitle_seconds_per_media_second("k", self.mem, cfg), 0.65)
        self.assertEqual(ssg.subtitle_seconds_per_media_second("k", None, cfg), 0.65)
        self.mem.record("k", 120.0, 100.0)
        self.assertAlmostEqual(ssg.subtitle_seconds_per_media_second("k", self.mem, cfg), 1.2)


if __name__ == "__main__":
    unittest.main()
