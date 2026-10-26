"""Player performance hints from the saved profile (pure logic: no mpv, torch or GPU)."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from oh_streamer.perf import profile as P  # noqa: E402
from oh_streamer.player import info_panel  # noqa: E402
from oh_streamer.player import perf_hint as H  # noqa: E402


def prof(d0, d1, h="720"):
    return {"schema": 1, "fingerprint": {}, "restore": {h: {"d0_fps": d0, "d1_fps": d1}}, "rife": {}, "encode": {}}


FAST, SLOW = prof(900.0, 600.0), prof(12.0, 6.0)


def find_prebuffer():
    for d0 in range(20, 200, 2):
        for d1 in range(10, d0, 2):
            p = prof(float(d0), float(d1))
            if P.verdict(p, height=720, src_fps=30)["level"] == "prebuffer":
                return p
    raise AssertionError("no prebuffer profile found")


class HintForTests(unittest.TestCase):
    def test_live_has_no_hint(self):
        self.assertEqual(P.verdict(FAST, height=720, src_fps=30)["level"], "live")
        self.assertIsNone(H.hint_for(FAST, 720, 30, "off"))

    def test_queue_text(self):
        v = P.verdict(SLOW, height=720, src_fps=30)
        self.assertEqual(v["level"], "queue")
        level, text = H.hint_for(SLOW, 720, 30, "off")
        self.assertEqual(level, "queue")
        self.assertIn(f"{v['ratio_typical']:.1f}x 即時", text)
        self.assertIn("按 r 降畫質", text)
        self.assertIn("按 a 交給背景佇列", text)

    def test_prebuffer_text(self):
        p = find_prebuffer()
        v = P.verdict(p, height=720, src_fps=30)
        level, text = H.hint_for(p, 720, 30, "off")
        self.assertEqual(level, "prebuffer")
        self.assertIn(f"{v['ratio_typical']:.1f}x 即時", text)
        self.assertIn(f"{v['prebuffer_s']:.0f} 秒", text)

    def test_no_profile_or_bad_input(self):
        self.assertIsNone(H.hint_for(None, 720, 30, "off"))
        self.assertIsNone(H.hint_for({}, 720, 30, "off"))
        self.assertIsNone(H.hint_for(SLOW, None, 30, "off"))
        self.assertIsNone(H.hint_for("garbage", 720, 30, "off"))
        self.assertEqual(H.mark_for(None, 720, 30, "off"), "")

    def test_mode_follows_interpolation(self):
        self.assertEqual(H.mode_for("auto"), "auto")
        self.assertEqual(H.mode_for("off"), "off")
        self.assertEqual(H.mode_for(None), "off")

    def test_unknown_fps_assumes_30(self):
        self.assertEqual(H.mark_for(SLOW, 720, None, "off"), H.mark_for(SLOW, 720, 30, "off"))


class MarkTests(unittest.TestCase):
    def test_marks(self):
        self.assertEqual(H.mark_for(FAST, 720, 30, "off"), "live")
        self.assertEqual(H.mark_for(SLOW, 720, 30, "off"), "queue")
        self.assertEqual(H.mark_for(find_prebuffer(), 720, 30, "off"), "prebuffer")

    def test_annotate_public_list(self):
        quals = [{"id": "a", "label": "720p", "height": 720, "fps": None},
                 {"id": "b", "label": "1080p", "height": 1080, "fps": 60.0}]
        pub = json.dumps([{"id": q["id"], "label": q["label"], "height": q["height"]} for q in quals])
        out = json.loads(H.annotate_public_list(pub, quals, SLOW, "off"))
        self.assertEqual([o["id"] for o in out], ["a", "b"])
        self.assertEqual(out[0]["mark"], "queue")
        self.assertEqual(H.annotate_public_list(pub, quals, None, "off"), pub)
        self.assertEqual(H.annotate_public_list("not json", quals, SLOW, "off"), "not json")

    def test_annotate_judges_a_60_fps_rendition_at_its_restored_rate(self):
        p = find_prebuffer()    # 720p: 30 fps prebuffer, 60 fps queue
        quals = [{"id": "b", "label": "720p60", "height": 720, "fps": 60.0}]
        pub = json.dumps([{"id": "b", "label": "720p60", "height": 720}])
        mark = lambda rfps: json.loads(H.annotate_public_list(pub, quals, p, "off", rfps))[0]["mark"]
        self.assertEqual((mark("original"), mark("30")), ("queue", "prebuffer"))
        self.assertEqual(H.restored_fps(59.94, "30"), 29.97)


class InfoTests(unittest.TestCase):
    def test_info_for_and_panel_row(self):
        info = H.info_for(SLOW, 720, 30, "off")
        self.assertEqual(info["label"], "建議佇列")
        self.assertEqual(H.info_for(FAST, 720, 30, "off")["label"], "可即時")
        self.assertEqual(H.info_for(find_prebuffer(), 720, 30, "off")["label"], "需預緩衝")
        self.assertIsNone(H.info_for(None, 720, 30, "off"))
        panel = info_panel.build_info({"perf": info})
        rows = {r["k"]: r["v"] for s in panel["sections"] for r in s["rows"]}
        self.assertEqual(rows["效能預估"], f"{info['ratio']:.1f}x（建議佇列）")
        panel = info_panel.build_info({})
        self.assertNotIn("效能預估", [r["k"] for s in panel["sections"] for r in s["rows"]])


class TrackerTests(unittest.TestCase):
    def test_once_per_source_and_on_level_change(self):
        t = H.HintTracker()
        first = t.next(1, SLOW, 720, 30, "off")
        self.assertEqual(first[0], "queue")
        self.assertEqual(first[2], 6000)
        self.assertIsNone(t.next(1, SLOW, 720, 30, "off"))       # seek / segment restart
        self.assertIsNone(t.next(1, FAST, 720, 30, "off"))       # quality switch to live: nothing to say
        again = t.next(1, SLOW, 720, 30, "off")                  # level changed back
        self.assertEqual(again[0], "queue")
        self.assertEqual(t.next(2, SLOW, 720, 30, "off")[0], "queue")   # new source
        self.assertIsNone(t.next(2, None, 720, 30, "off"))

    def test_prebuffer_duration(self):
        t = H.HintTracker()
        self.assertEqual(t.next(1, find_prebuffer(), 720, 30, "off")[2], 5000)


class LoaderTests(unittest.TestCase):
    def test_lazy_load_and_reload_on_mtime_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "perf_profile.json")
            self.assertIsNone(H.get_profile(path))
            P.save(FAST, path)
            first = H.get_profile(path)
            self.assertEqual(first["restore"]["720"]["d0_fps"], 900.0)
            self.assertIs(H.get_profile(path), first)            # cached
            P.save(SLOW, path)
            os.utime(path, ns=(1, 2_000_000_000_000_000_000))
            self.assertEqual(H.get_profile(path)["restore"]["720"]["d0_fps"], 12.0)
            with open(path, "w", encoding="utf-8") as f:
                f.write("{broken")
            os.utime(path, ns=(1, 3_000_000_000_000_000_000))
            self.assertIsNone(H.get_profile(path))


if __name__ == "__main__":
    unittest.main()
