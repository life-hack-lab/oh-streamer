"""Offline unit tests for player_settings.py and info_panel.py (run: python streamer/tests/test_settings_info.py)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import json
import os
import sys
import tempfile
import unittest
import oh_streamer.player.info_panel as I  # noqa: E402
import oh_streamer.player.player_settings as PS  # noqa: E402


def cli(**kw):
    ns = argparse.Namespace(**{k: None for k in PS.KEYS})
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


DET = ["v2", "v4-fast", "v4-accurate"]
RES = ["basicvsrpp-v1.2"]


class SettingsFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "oh_settings.json")

    def tearDown(self):
        self.dir.cleanup()

    def write(self, text):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_missing_file_gives_defaults(self):
        saved, warn = PS.load_file(self.path)
        self.assertEqual(saved, {})
        self.assertIsNone(warn)
        self.assertEqual(PS.effective(saved, {}), PS.DEFAULTS)
        self.assertEqual(PS.DEFAULTS["max_clip_length"], 240)

    def test_corrupt_file_gives_defaults(self):
        for text in ["{not json", "", "[1, 2]", "null", "\x00\x01"]:
            self.write(text)
            saved, warn = PS.load_file(self.path)
            self.assertEqual(saved, {}, text)
            self.assertIsNotNone(warn, text)

    def test_bad_values_dropped_good_kept(self):
        self.write(json.dumps({"max_clip_length": "big", "seek_prebuffer": True, "encoder": "h265",
                               "fp16": "yes", "detection_model": "v4-accurate", "unknown": 1}))
        saved, warn = PS.load_file(self.path)
        self.assertEqual(saved, {"detection_model": "v4-accurate"})
        self.assertIn("max_clip_length", warn)

    def test_save_load_roundtrip(self):
        data = {"max_clip_length": 90, "detection_model": "v4-accurate", "fp16": False, "seek_prebuffer": 0.5,
                "encoder": "libx264", "restoration_model": "basicvsrpp-v1.2"}
        PS.save_file(self.path, data)
        self.assertEqual(PS.load_file(self.path), (data, None))
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_cli_overrides_saved(self):
        PS.save_file(self.path, {"max_clip_length": 60, "detection_model": "v4-accurate", "fp16": False})
        saved, _ = PS.load_file(self.path)
        over = PS.cli_overrides(cli(max_clip_length=300, fp16=True))
        cfg = PS.effective(saved, over)
        self.assertEqual(cfg["max_clip_length"], 300)          # CLI wins
        self.assertTrue(cfg["fp16"])                           # CLI --fp16 wins over saved false
        self.assertEqual(cfg["detection_model"], "v4-accurate")  # saved wins over default
        self.assertEqual(cfg["encoder"], "auto")               # default
        self.assertEqual(PS.cli_overrides(cli()), {})          # nothing given explicitly

    def test_no_fp16_flag_is_explicit_false(self):
        self.assertEqual(PS.cli_overrides(cli(fp16=False)), {"fp16": False})

    def test_menu_change_persists_only_changed_keys(self):
        saved = {"detection_model": "v4-accurate"}
        # a CLI one-off override is in cfg but must not leak into the file
        out = PS.updated_file(saved, {"max_clip_length": 120})
        self.assertEqual(out, {"detection_model": "v4-accurate", "max_clip_length": 120})

    def test_fix_models_unavailable(self):
        cfg, notes = PS.fix_models(PS.effective({"detection_model": "v3"}, {}), {}, DET, RES)
        self.assertEqual(cfg["detection_model"], "v4-fast")
        self.assertEqual(len(notes), 1)
        # an explicit CLI name is left alone (the caller reports it)
        cfg, notes = PS.fix_models(PS.effective({}, {"detection_model": "v3"}), {"detection_model": "v3"}, DET, RES)
        self.assertEqual(cfg["detection_model"], "v3")
        self.assertEqual(notes, [])

    def test_volume_mute_persisted_but_not_settings(self):
        PS.save_file(self.path, {"volume": 35, "mute": True, "max_clip_length": 90})
        saved, warn = PS.load_file(self.path)
        self.assertIsNone(warn)
        self.assertEqual(PS.ui_state(saved), {"volume": 35.0, "mute": True, "cast_vsr": "off", "cast_vsr_size": "2x"})
        cfg = PS.effective(saved, {})
        self.assertNotIn("volume", cfg)
        self.assertEqual(cfg["max_clip_length"], 90)
        self.assertEqual(PS.ui_state({}), PS.UI_DEFAULTS)
        self.write(json.dumps({"volume": 250, "mute": "yes"}))
        saved, warn = PS.load_file(self.path)
        self.assertEqual(PS.ui_state(saved), PS.UI_DEFAULTS)
        self.assertIsNotNone(warn)

    def test_cli_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            PS.cli_overrides(cli(max_clip_length=0))


class SettingsMenuTest(unittest.TestCase):
    def menu(self, **over):
        cfg = {**PS.DEFAULTS, **over}
        return PS.build_menu(cfg, True, DET, RES, True, "h264_nvenc")

    def test_menu_items_and_values(self):
        m = self.menu()
        self.assertEqual([it["key"] for it in m], ["max_clip_length", "detection_model", "restoration_model", "fp16",
                                                   "seek_prebuffer", "encoder", "vsr", "smooth_playback",
                                                   "restoration_fps", "interpolation", "compare"])
        clip = m[0]
        self.assertEqual([o["v"] for o in clip["options"]], ["30", "60", "90", "120", "180", "240", "300", "360"])
        self.assertEqual(clip["value"], "240")
        self.assertEqual(clip["value_label"], "240 幀")
        self.assertEqual(clip["kind"], "restart")
        self.assertEqual(m[1]["kind"], "reload")
        self.assertEqual([o["v"] for o in m[1]["options"]], DET)
        self.assertEqual(m[3]["value"], "on")
        self.assertEqual(m[4]["kind"], "live")
        self.assertEqual(m[4]["value"], "1")
        self.assertEqual([o["v"] for o in m[5]["options"]], ["auto", "h264_nvenc", "libx264"])
        self.assertEqual([o["v"] for o in m[8]["options"]], ["original", "30"])
        json.dumps(m, ensure_ascii=False)   # publishable

    def test_current_value_outside_list_is_offered(self):
        m = self.menu(max_clip_length=250, seek_prebuffer=1.5)
        self.assertIn("250", [o["v"] for o in m[0]["options"]])
        self.assertIn("1.5", [o["v"] for o in m[4]["options"]])

    def test_no_nvenc_option_without_nvenc(self):
        m = PS.build_menu(dict(PS.DEFAULTS), False, DET, RES, False, "libx264")
        self.assertEqual([o["v"] for o in m[5]["options"]], ["auto", "libx264"])

    def test_parse_request(self):
        m = self.menu()
        req = "7:" + json.dumps({"max_clip_length": "60", "fp16": "off", "detection_model": "v4-accurate",
                                 "seek_prebuffer": "0.5", "encoder": "libx264"})
        self.assertEqual(PS.parse_request(req, m), {"max_clip_length": 60, "fp16": False,
                                                     "detection_model": "v4-accurate", "seek_prebuffer": 0.5,
                                                     "encoder": "libx264"})

    def test_parse_request_rejects_unoffered(self):
        m = self.menu()
        req = "8:" + json.dumps({"max_clip_length": "61", "detection_model": "../../evil", "x": "1"})
        self.assertEqual(PS.parse_request(req, m), {})
        self.assertEqual(PS.parse_request("garbage", m), {})
        self.assertEqual(PS.parse_request("9:{bad", m), {})
        self.assertEqual(PS.parse_request(None, m), {})

    def test_diff(self):
        cfg = dict(PS.DEFAULTS)
        self.assertEqual(PS.diff(cfg, True, {"fp16": True, "max_clip_length": 240}), {})
        self.assertEqual(PS.diff(cfg, True, {"fp16": False, "max_clip_length": 60}),
                         {"fp16": False, "max_clip_length": 60})

    def test_key_kinds_partition(self):
        self.assertEqual(PS.RELOAD_KEYS | PS.LIVE_KEYS | PS.RESTART_KEYS, set(PS.KEYS))
        self.assertEqual(PS.RESTART_KEYS, {"max_clip_length", "encoder", "restoration_fps", "interpolation", "compare", "engine"})
        self.assertEqual(PS.LIVE_KEYS, {"seek_prebuffer", "vsr", "smooth_playback"})
        for it in self.menu():   # the menu's kinds agree with the key sets (oh_seek.lua sends "live" ones live)
            want = "live" if it["key"] in PS.LIVE_KEYS else "reload" if it["key"] in PS.RELOAD_KEYS else "restart"
            self.assertEqual(it["kind"], want, it["key"])


class PlayerOptsTest(unittest.TestCase):
    """vsr / smooth_playback: on/off settings that act in mpv (no restart)."""

    def test_defaults_on(self):
        self.assertEqual(PS.DEFAULTS["vsr"], "on")
        self.assertEqual(PS.DEFAULTS["smooth_playback"], "on")

    def test_validate(self):
        for key in ("vsr", "smooth_playback"):
            self.assertEqual(PS.validate(key, "on"), "on")
            self.assertEqual(PS.validate(key, "off"), "off")
            for bad in (True, False, None, "yes", "ON", 1, ""):
                with self.assertRaises(ValueError, msg=f"{key}={bad!r}"):
                    PS.validate(key, bad)

    def test_persist_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            PS.save_file(path, PS.updated_file({}, {"vsr": "off", "smooth_playback": "off"}))
            saved, warn = PS.load_file(path)
            self.assertIsNone(warn)
            self.assertEqual(saved, {"vsr": "off", "smooth_playback": "off"})
            cfg = PS.effective(saved, PS.cli_overrides(cli(vsr="on")))
            self.assertEqual((cfg["vsr"], cfg["smooth_playback"]), ("on", "off"))   # CLI wins, saved kept
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"vsr": "maybe", "smooth_playback": True}, f)
            saved, warn = PS.load_file(path)
            self.assertEqual(saved, {})
            self.assertIn("vsr", warn)
        with self.assertRaises(ValueError):
            PS.cli_overrides(cli(smooth_playback="yes"))

    def test_menu_items(self):
        m = {it["key"]: it for it in PS.build_menu({**PS.DEFAULTS, "smooth_playback": "off"}, True, DET, RES, True,
                                                  "h264_nvenc")}
        for key in ("vsr", "smooth_playback"):
            self.assertEqual(m[key]["kind"], "live")
            self.assertEqual([(o["v"], o["label"]) for o in m[key]["options"]], [("on", "開"), ("off", "關")])
            self.assertTrue(m[key]["hint"])
        self.assertEqual(m["vsr"]["label"], "RTX 超解析度")
        self.assertEqual((m["vsr"]["value"], m["vsr"]["value_label"]), ("on", "開"))
        self.assertEqual((m["smooth_playback"]["value"], m["smooth_playback"]["value_label"]), ("off", "關"))

    def test_live_request(self):
        m = PS.build_menu(dict(PS.DEFAULTS), True, DET, RES, True, "h264_nvenc")
        wanted = PS.parse_request("3:" + json.dumps({"vsr": "off", "smooth_playback": "on"}), m)
        self.assertEqual(wanted, {"vsr": "off", "smooth_playback": "on"})
        changes = PS.diff(dict(PS.DEFAULTS), True, wanted)
        self.assertEqual(changes, {"vsr": "off"})
        self.assertTrue(changes.keys() <= PS.LIVE_KEYS)
        self.assertEqual(PS.parse_request("4:" + json.dumps({"vsr": "auto"}), m), {})

    def test_smooth_props(self):
        self.assertEqual(PS.smooth_props("on"),
                         {"video-sync": "display-resample", "interpolation": "yes", "tscale": "oversample"})
        self.assertEqual(PS.smooth_props("off"), {"video-sync": "audio", "interpolation": "no"})


class InterpolationSettingTest(unittest.TestCase):
    """interpolation: off / auto (adaptive RIFE 2x), restarts the segment (the output frame rate changes)."""

    def test_default_off_and_validate(self):
        self.assertEqual(PS.DEFAULTS["interpolation"], "off")
        self.assertEqual(PS.validate("interpolation", "auto"), "auto")
        for bad in ("on", True, None, "AUTO", "", 2):
            with self.assertRaises(ValueError, msg=repr(bad)):
                PS.validate("interpolation", bad)

    def test_persist_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            PS.save_file(path, PS.updated_file({}, {"interpolation": "auto"}))
            saved, warn = PS.load_file(path)
            self.assertIsNone(warn)
            self.assertEqual(saved, {"interpolation": "auto"})
            self.assertEqual(PS.effective(saved, {})["interpolation"], "auto")
            self.assertEqual(PS.effective(saved, PS.cli_overrides(cli(interpolation="off")))["interpolation"], "off")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"interpolation": "rife"}, f)
            saved, warn = PS.load_file(path)
            self.assertEqual(saved, {})
            self.assertIn("interpolation", warn)
        with self.assertRaises(ValueError):
            PS.cli_overrides(cli(interpolation="on"))

    def test_menu_item_and_request(self):
        m = PS.build_menu(dict(PS.DEFAULTS), True, DET, RES, True, "h264_nvenc")
        it = next(i for i in m if i["key"] == "interpolation")
        self.assertEqual(it["label"], "插幀（RIFE）")
        self.assertEqual(it["kind"], "restart")
        self.assertIn("重複幀", it["hint"])
        self.assertEqual([(o["v"], o["label"]) for o in it["options"]], [("off", "關"), ("auto", "自動")])
        self.assertEqual((it["value"], it["value_label"]), ("off", "關"))
        wanted = PS.parse_request("5:" + json.dumps({"interpolation": "auto"}), m)
        self.assertEqual(PS.diff(dict(PS.DEFAULTS), True, wanted), {"interpolation": "auto"})
        self.assertEqual(PS.parse_request("6:" + json.dumps({"interpolation": "on"}), m), {})


def snapshot(**over):
    s = {"title": "影片標題 test", "source": "example.com", "width": 1920, "height": 1080, "fps": 30.0,
         "codec": "h264", "duration": 600.0, "quality": "1080p", "pos": 65.25, "cache_time": 80.0,
         "proc_fps": 45.0, "proc_done": False, "frames_total": 12345, "segment_start": 60.0,
         "last_latency": 1.234, "last_kind": "seek", "state": "播放中", "detection_model": "v4-fast",
         "restoration_model": "basicvsrpp-v1.2", "max_clip_length": 240, "fp16": True, "encoder": "h264_nvenc",
         "prebuffer": 5.0, "torch_alloc_gib": 1.2,
         "gpu": {"name": "NVIDIA GeForce RTX 5060 Laptop GPU", "util": 87.0, "mem_used_gib": 3.5,
                 "mem_total_gib": 8.0, "temp": 66.0, "sm_clock": 2400.0, "power_w": 55.25}}
    s.update(over)
    return s


def rows(info, title):
    sec = next(s for s in info["sections"] if s["title"] == title)
    return {r["k"]: r for r in sec["rows"]}


class DiagnoseTest(unittest.TestCase):
    P = dict(feeder=(4, 8), clips=(0, 10), out=(50, 200))

    def d(self, ratio=0.6, ahead=0.5, util=40.0, **pipe):
        return I.diagnose({**self.P, **pipe}, ratio, ahead, util)[0]

    def test_healthy_or_unknown(self):
        self.assertEqual(I.diagnose(self.P, 1.5, 9.0, 50.0), ("正常", "good"))
        self.assertEqual(I.diagnose(None, 0.5, 0.2, 50.0)[1], "idle")
        self.assertEqual(I.diagnose(self.P, 0.5, 0.2, 50.0, done=True)[1], "idle")

    def test_stage(self):
        self.assertIn("輸入端", self.d(feeder=(0, 8)))
        self.assertIn("GPU 修復", self.d(clips=(3, 10)))
        self.assertIn("輸出", self.d(out=(190, 200)))
        self.assertIn("GPU 偵測", self.d(feeder=(8, 8)))
        self.assertIn("GPU 偵測", self.d(feeder=(1, 8), util=95.0))   # busy GPU is not an input problem

    def test_rows(self):
        p = rows(I.build_info(snapshot(proc_fps=10.0, pipe={**self.P, "feeder": (0, 8)}, drops=(3, 1),
                                         gpu={"util": 30.0})), "處理狀態")
        self.assertIn("輸入端", p["卡頓原因"]["v"])
        self.assertEqual(p["解碼佇列"]["v"], "0 / 8")
        self.assertEqual(p["播放端掉幀"]["v"], "3 掉 · 1 延遲")


class InfoBuilderTest(unittest.TestCase):
    def test_sections_and_values(self):
        info = I.build_info(snapshot())
        self.assertEqual([s["title"] for s in info["sections"]], ["影片", "處理狀態", "設定", "GPU"])
        v = rows(info, "影片")
        self.assertEqual(v["解析度"]["v"], "1920x1080")
        self.assertEqual(v["影格率"]["v"], "30 fps")
        self.assertEqual(v["長度"]["v"], "10:00")
        p = rows(info, "處理狀態")
        self.assertEqual(p["播放位置"]["v"], "01:05.2")
        self.assertEqual(p["預處理緩衝"]["v"], "14.8 秒")
        self.assertEqual(p["預處理緩衝"]["c"], "good")
        self.assertEqual(p["處理速度"]["v"], "45.0 fps · 1.50x 即時")
        self.assertEqual(p["處理速度"]["c"], "good")
        self.assertEqual(p["已處理幀數"]["v"], "12,345")
        self.assertEqual(p["上次延遲"]["v"], "1.23 秒（seek）")
        st = rows(info, "設定")
        self.assertEqual(st["FP16"]["v"], "開")
        self.assertEqual(st["修復時間窗"]["v"], "240 幀")
        self.assertEqual(st["編碼器"]["v"], "h264_nvenc")
        gp = rows(info, "GPU")
        self.assertEqual(gp["名稱"]["v"], "RTX 5060 Laptop GPU")
        self.assertEqual(gp["VRAM"]["v"], "3.50 / 8.0 GB")
        self.assertEqual(gp["功耗"]["v"], "55.2 W")
        json.dumps(info, ensure_ascii=False)

    def test_speed_colors(self):
        for fps, level in ((36.0, "good"), (35.9, "warn"), (30.0, "warn"), (29.9, "bad")):
            p = rows(I.build_info(snapshot(proc_fps=fps)), "處理狀態")
            self.assertEqual(p["處理速度"]["c"], level, fps)

    def test_buffer_colors(self):
        for cache, level in ((70.25, "good"), (68.0, "warn"), (66.0, "bad")):
            p = rows(I.build_info(snapshot(cache_time=cache)), "處理狀態")
            self.assertEqual(p["預處理緩衝"]["c"], level, cache)

    def test_missing_values_and_no_gpu(self):
        info = I.build_info({"state": "處理中"})
        p = rows(info, "處理狀態")
        self.assertEqual(p["播放位置"]["v"], "—")
        self.assertEqual(p["處理速度"]["c"], "idle")
        self.assertEqual(rows(info, "GPU")["狀態"]["v"], "不可用")
        self.assertEqual(rows(info, "設定")["FP16"]["v"], "—")
        self.assertEqual(rows(info, "設定")["RTX VSR"]["v"], "—")
        self.assertEqual(rows(info, "設定")["平滑播放"]["v"], "—")

    def test_vsr_and_smooth_rows(self):
        def st(**over):
            return rows(I.build_info(snapshot(**over)), "設定")
        r = st(vsr="開 1.48x（1080→1600）", smooth_playback="on")
        self.assertEqual(r["RTX VSR"]["v"], "開 1.48x（1080→1600）")
        self.assertEqual(r["RTX VSR"]["c"], "good")
        self.assertEqual(r["平滑播放"]["v"], "開")
        self.assertEqual(st(smooth_playback="off")["平滑播放"]["v"], "關")
        for status, level in (("關", "idle"), ("不可用（非 d3d11va 硬解）", "idle"), ("失敗（d3d11vpp 無法啟用）", "bad"),
                              ("開（等待影片）", None), ("開（不需放大：1080→1080）", None)):
            self.assertEqual(st(vsr=status)["RTX VSR"].get("c"), level, status)

    def test_done(self):
        p = rows(I.build_info(snapshot(proc_done=True, proc_fps=0.0)), "處理狀態")
        self.assertEqual(p["處理速度"]["v"], "已處理到結尾")
        self.assertEqual(p["處理速度"]["c"], "idle")

    def test_truncate_and_width(self):
        self.assertEqual(I.display_width("ab中文"), 6)
        t = I.truncate("很長的標題" * 20, 20)
        self.assertLessEqual(I.display_width(t), 20)
        self.assertTrue(t.endswith("…"))
        self.assertEqual(I.truncate("short", 20), "short")

    def test_source_label(self):
        self.assertEqual(I.source_label("https://www.example.com/v/1?x=y"), "www.example.com")
        self.assertEqual(I.source_label(r"C:\videos\a.mp4"), "本地檔案")
        self.assertEqual(I.source_label(None), "本地檔案")

    def test_fmt_clock(self):
        self.assertEqual(I.fmt_clock(3725.5, tenths=True), "1:02:05.5")
        self.assertEqual(I.fmt_clock(59.99), "00:59")
        self.assertEqual(I.fmt_clock(None), "—")

    def test_parse_smi_line(self):
        s = I.parse_smi_line("NVIDIA GeForce RTX 5060 Laptop GPU, 34, 2048, 8151, 55, 2100, 23.45, 115.00")
        self.assertEqual(s["util"], 34.0)
        self.assertAlmostEqual(s["mem_used_gib"], 2.0)
        self.assertEqual(s["power_w"], 23.45)
        self.assertEqual(s["power_limit_w"], 115.0)
        s = I.parse_smi_line("NVIDIA X, [N/A], 100, 1000, 40, [N/A], [N/A], [N/A]")
        self.assertIsNone(s["util"])
        self.assertIsNone(s["power_w"])
        self.assertIsNone(s["power_limit_w"])
        # older 7-field line (no power.limit)
        self.assertIsNone(I.parse_smi_line("NVIDIA X, 1, 100, 1000, 40, 2100, 9.5")["power_limit_w"])
        self.assertIsNone(I.parse_smi_line(""))
        self.assertIsNone(I.parse_smi_line("garbage"))

    def test_cast_section(self):
        self.assertEqual([s["title"] for s in I.build_info(snapshot(cast=None))["sections"]],
                         ["影片", "處理狀態", "設定", "GPU"])
        cast = {"mode": "airplay", "device": "客廳 Apple TV", "phase": "playing",
                "url": "http://192.168.1.20:50123/AbCdEfGhIjKlMnOpQrStUvWx/master.m3u8", "prod_pos": 125.5,
                "producing": True, "paused": False, "cached_segments": 40, "cached_bytes": 300 * 2**20,
                "cached_ranges": [(0.0, 64.0), (100.1, 160.16), (300.0, 304.0)], "requests": 55, "restarts": 2,
                "last_request": {"k": 30, "pos": 120.12, "how": "快取", "age": 1.4}, "jump_latency": 3.21,
                "error": None, "tv_pos": 118.0, "duration": 600.0}
        info = I.build_info(snapshot(cast=cast))
        self.assertEqual([s["title"] for s in info["sections"]], ["影片", "投放", "處理狀態", "設定", "GPU"])
        c = rows(info, "投放")
        self.assertEqual(c["狀態"]["v"], "投放中")
        self.assertEqual(c["網址"]["v"], "http://192.168.1.20:50123/…/master.m3u8")   # token not shown
        self.assertEqual(c["已快取"]["v"], "40 段 · 300 MB")
        self.assertEqual(c["快取範圍"]["v"], "00:00–01:04、01:40–02:40…")
        self.assertEqual(c["最後請求"]["v"], "#30 02:00 · 快取 · 1 秒前")
        self.assertEqual(c["跳轉延遲"]["c"], "good")
        self.assertEqual(c["電視位置"]["v"], "01:58.0")
        c = rows(I.build_info(snapshot(cast={**cast, "prod_pos": None, "error": "boom", "phase": "hls",
                                              "device": None, "last_request": None})), "投放")
        self.assertEqual(c["製作位置"]["v"], "閒置")
        self.assertEqual(c["錯誤"]["c"], "bad")
        self.assertNotIn("裝置", c)
        json.dumps(info, ensure_ascii=False)

    def test_history_charts(self):
        h = I.History(n=5)
        for i in range(7):
            h.add(ratio=1.0 + i / 10, buffer=None if i == 3 else 2.0, util=50.0 + i, vram=3.5, power=40.0)
        info = I.build_info(snapshot(history=h.snapshot()))
        p, gp = rows(info, "處理狀態"), rows(info, "GPU")
        self.assertEqual(p["處理速度"]["h"], [1.2, 1.3, 1.4, 1.5, 1.6])      # last n samples, oldest first
        self.assertEqual(p["處理速度"]["href"], 1.0)
        self.assertEqual(p["處理速度"]["hmax"], 2.0)                          # floor
        self.assertEqual(p["預處理緩衝"]["h"], [2.0, -1, 2.0, 2.0, 2.0])      # gap = -1 (JSON null breaks Lua #)
        self.assertEqual(p["預處理緩衝"]["hmax"], 5.0)                        # at least the prebuffer
        self.assertEqual(gp["使用率"]["hmax"], 100)
        self.assertEqual(gp["VRAM"]["hmax"], 8.0)                            # total VRAM
        self.assertNotIn("h", rows(info, "影片")["標題"])
        h.clear("ratio", "buffer")
        s = h.snapshot()
        self.assertEqual(s["ratio"], [None] * 5)                             # time axis stays aligned
        self.assertEqual(len(s["util"]), 5)
        self.assertNotIn("h", rows(I.build_info(snapshot(history=s)), "處理狀態")["處理速度"])
        json.dumps(info, ensure_ascii=False)

    def test_compare_rows(self):
        def info(c):
            return I.build_info(snapshot(compare=c))
        self.assertNotIn("對照", rows(info({"mode": "off"}), "處理狀態"))
        self.assertNotIn("對照", rows(I.build_info(snapshot()), "處理狀態"))
        r = rows(info({"mode": "on", "layers": ["processed", "original"], "why": ""}), "處理狀態")["對照"]
        self.assertEqual((r["v"], r["c"]), ("處理後、原始", "good"))
        r = rows(info({"mode": "on", "layers": ["processed"], "why": "video too large"}), "處理狀態")["對照"]
        self.assertIn("video too large", r["v"])

    def test_interpolation_rows(self):
        def info(it):
            return I.build_info(snapshot(interp=it))
        # off: only the settings row, no output note
        i = info({"mode": "off"})
        self.assertEqual(rows(i, "設定")["插幀（RIFE）"]["v"], "關")
        self.assertNotIn("插幀", rows(i, "處理狀態"))
        self.assertEqual(rows(i, "影片")["影格率"]["v"], "30 fps")
        self.assertEqual(rows(I.build_info(snapshot()), "設定")["插幀（RIFE）"]["v"], "—")
        # auto, interpolating
        base = {"mode": "auto", "eligible": True, "out_fps": 60.0, "ratio": 0.62, "rife_ms": 15.24, "state": "on",
                "lead": 7.5, "start_lead": 6.0}
        i = info(base)
        p = rows(i, "處理狀態")
        self.assertEqual(p["插幀"]["v"], "自動 62%（近 10 秒補幀比例）")
        self.assertEqual(p["插幀"]["c"], "good")
        self.assertEqual(p["RIFE"]["v"], "15.2 ms · 補幀中")
        self.assertEqual(rows(i, "設定")["插幀（RIFE）"]["v"], "自動")
        self.assertEqual(rows(i, "影片")["影格率"]["v"], "30 fps（輸出 60）")
        # copying for lack of lead / GPU / while loading / waiting
        p = rows(info({**base, "state": "lead", "lead": 2.5, "ratio": 0.0}), "處理狀態")
        self.assertEqual(p["插幀"]["v"], "自動 0%（近 10 秒補幀比例）")
        self.assertEqual(p["RIFE"]["v"], "15.2 ms · 重複幀（緩衝 2.5／6 秒）")
        self.assertEqual(rows(info({**base, "state": "budget"}), "處理狀態")["RIFE"]["c"], "warn")
        self.assertEqual(rows(info({**base, "state": "lead", "lead": -4.2}), "處理狀態")["RIFE"]["v"],
                         "15.2 ms · 重複幀（緩衝 0.0／6 秒）")
        p = rows(info({**base, "state": "loading", "rife_ms": None, "ratio": None}), "處理狀態")
        self.assertEqual((p["插幀"]["v"], p["RIFE"]["v"]), ("自動 —（近 10 秒補幀比例）", "載入模型中"))
        self.assertEqual(rows(info({**base, "state": "waiting", "lead": None}), "處理狀態")["RIFE"]["v"],
                         "15.2 ms · 等待播放")
        self.assertEqual(rows(info({**base, "state": "error", "error": "RuntimeError('x')"}), "處理狀態")["RIFE"]["c"],
                         "bad")
        # not eligible (60 fps source): 1x, says why
        i = info({"mode": "auto", "eligible": False, "why": "來源已是 60 fps"})
        self.assertEqual(rows(i, "處理狀態")["插幀"]["v"], "未啟用（來源已是 60 fps）")
        self.assertNotIn("RIFE", rows(i, "處理狀態"))
        self.assertEqual(rows(i, "影片")["影格率"]["v"], "30 fps")
        json.dumps(info(base), ensure_ascii=False)

    def test_gpu_monitor_inactive_is_cheap_and_safe(self):
        m = I.GpuMonitor()
        self.assertIsNone(m.snapshot())    # never started: no thread, no stats
        m.close()


if __name__ == "__main__":
    unittest.main(verbosity=1)
