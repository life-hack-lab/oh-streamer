"""Offline unit tests for compare_stack.py: layer naming, stacked frames, ffmpeg arguments, original-frame capture and
rebuild (run: python streamer/tests/test_compare_stack.py). No ffmpeg, mpv, PyAV or torch."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import os
import sys
import types
import unittest
from fractions import Fraction

import numpy as np
import oh_streamer.processing.compare_stack as CS  # noqa: E402

W, H = 8, 4


def img(v):
    return np.full((H, W, 3), v, dtype=np.uint8)


class Frame(np.ndarray):
    """ndarray that accepts attributes, like the torch frame."""


def make_frame(v):
    return img(v).view(Frame)


class FakeRestorer:
    capture_original = False

    def _restore_frame(self, frame, frame_num, restored_clips):
        for c in restored_clips:
            if c.frame_start == frame_num:
                t, l, b, r = c.boxes.pop(0)
                frame[t:b + 1, l:r + 1] = 255
        return frame


def clip(start, box):
    return types.SimpleNamespace(frame_start=start, boxes=[box])


class TestLayers(unittest.TestCase):
    def test_names(self):
        self.assertEqual(CS.layer_names(1), ("processed", "original"))
        self.assertEqual(CS.layer_names(2), ("processed", "original", "restored"))

    def test_stack_order_and_validation(self):
        self.assertEqual(CS.stack_order(("processed", "original", "restored")), ("processed", "restored", "original"))
        with self.assertRaises(ValueError):
            CS.stack_order(("processed", "bogus"))
        with self.assertRaises(ValueError):
            CS.stack_order(("processed", "processed"))

    def test_stacked_shape(self):
        self.assertEqual(CS.stacked_shape(W, H, ("processed", "original")), (W, 2 * H))

    def test_layers_for_write_order(self):
        p, r, o = img(1), img(2), img(3)
        out = CS.layers_for_write(p, r, o, CS.layer_names(2))
        self.assertEqual([x[0, 0, 0] for x in out], [1, 2, 3])
        out = CS.layers_for_write(p, None, o, CS.layer_names(1))
        self.assertEqual([x[0, 0, 0] for x in out], [1, 3])
        with self.assertRaises(ValueError):
            CS.layers_for_write(p, None, o, CS.layer_names(2))

    def test_build_and_split(self):
        p, r, o = img(1), img(2), img(3)
        s = CS.build_stacked(p, r, o, CS.layer_names(2))
        self.assertEqual(s.shape, (3 * H, W, 3))
        self.assertTrue(np.array_equal(s, CS.stack_frames([p, r, o])))
        parts = CS.split_stacked(s, CS.layer_names(2))
        self.assertTrue(np.array_equal(parts["original"], o))
        self.assertTrue(np.array_equal(parts["restored"], r))
        buf = np.empty_like(s)
        self.assertIs(CS.build_stacked(p, r, o, CS.layer_names(2), out=buf), buf)
        self.assertTrue(np.array_equal(buf, s))
        # the pump's byte stream (layers back to back) equals the stacked frame
        raw = b"".join(bytes(memoryview(x.reshape(-1))) for x in CS.layers_for_write(p, r, o, CS.layer_names(2)))
        self.assertEqual(raw, s.tobytes())

    def test_stack_frames_errors(self):
        with self.assertRaises(ValueError):
            CS.stack_frames([])
        with self.assertRaises(ValueError):
            CS.stack_frames([img(0), np.zeros((H + 1, W, 3), np.uint8)])


class TestFfmpegArgs(unittest.TestCase):
    def test_single_layer_matches_today(self):
        f, m, c, meta = CS.ffmpeg_layer_args(W, H, ("processed",), 1, "h264_nvenc")
        self.assertEqual((f, m, meta), ([], ["-map", "0:v:0"], []))
        self.assertEqual(c, ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "17", "-b:v", "0",
                             "-pix_fmt", "yuv420p"])
        f, m, c, meta = CS.ffmpeg_layer_args(W, H, ("processed",), 2, "libx264")
        self.assertEqual(f, ["-vf", "format=yuv420p,mpdecimate=hi=0:lo=0:frac=0", "-fps_mode", "vfr"])
        self.assertEqual(c, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])

    def test_two_tracks(self):
        f, m, c, meta = CS.ffmpeg_layer_args(1920, 1080, CS.layer_names(1), 1, "h264_nvenc")
        self.assertEqual(f[0], "-filter_complex")
        fc = f[1]
        self.assertIn("[0:v]split=2[s0][s1]", fc)
        self.assertIn("[s0]crop=1920:1080:0:0,format=yuv420p[v_processed]", fc)
        self.assertIn("[s1]crop=1920:1080:0:1080,format=yuv420p[v_original]", fc)
        self.assertNotIn("mpdecimate", fc)
        self.assertEqual(len(f), 2)
        self.assertEqual(m, ["-map", "[v_processed]", "-map", "[v_original]"])
        self.assertIn("-g", c)
        self.assertEqual(c[c.index("-g") + 1], str(CS.GOP))
        self.assertIn("-forced-idr", c)
        self.assertEqual(meta[:4], ["-metadata:s:v:0", "title=oh:processed", "-metadata:s:v:1", "title=oh:original"])
        self.assertEqual(meta[4:], ["-disposition:v:0", "default", "-disposition:v:1", "0"])

    def test_three_tracks_rife(self):
        f, m, c, meta = CS.ffmpeg_layer_args(W, H, CS.layer_names(2), 2, "h264_nvenc")
        fc = f[1]
        self.assertIn("split=3[s0][s1][s2]", fc)
        # stack order processed, restored, original; tracks processed, original, restored
        self.assertIn(f"[s1]crop={W}:{H}:0:{H},format=yuv420p,mpdecimate=hi=0:lo=0:frac=0[v_restored]", fc)
        self.assertIn(f"[s2]crop={W}:{H}:0:{2 * H},format=yuv420p,mpdecimate=hi=0:lo=0:frac=0[v_original]", fc)
        self.assertEqual(f[2:], ["-fps_mode", "vfr"])
        self.assertEqual(m, ["-map", "[v_processed]", "-map", "[v_original]", "-map", "[v_restored]"])
        self.assertIn("title=oh:restored", meta)
        self.assertEqual(meta[meta.index("title=oh:restored") - 1], "-metadata:s:v:2")

    def test_libx264_gop(self):
        c = CS.ffmpeg_layer_args(W, H, CS.layer_names(1), 1, "libx264")[2]
        self.assertEqual(c[-2:], ["-g", str(CS.GOP)])
        self.assertNotIn("-forced-idr", c)

    def test_bad_layers(self):
        with self.assertRaises(ValueError):
            CS.ffmpeg_layer_args(W, H, (), 1, "h264_nvenc")
        with self.assertRaises(ValueError):
            CS.ffmpeg_layer_args(W, H, ("original", "processed"), 1, "h264_nvenc")


class TestGuard(unittest.TestCase):
    @staticmethod
    def meta(w, h, fps):
        return types.SimpleNamespace(video_width=w, video_height=h, video_fps_exact=fps)

    def test_guard(self):
        two, three = CS.layer_names(1), CS.layer_names(2)
        self.assertEqual(CS.compare_ok(self.meta(1920, 1080, 30), two, "h264_nvenc"), (True, ""))
        self.assertEqual(CS.compare_ok(self.meta(1920, 1080, 60), two, "h264_nvenc")[0], True)
        self.assertEqual(CS.compare_ok(self.meta(1920, 1080, 60), three, "h264_nvenc")[0], False)
        self.assertFalse(CS.compare_ok(self.meta(3840, 2160, 30), three, "h264_nvenc")[0])
        self.assertFalse(CS.compare_ok(self.meta(3840, 2160, 60), two, "h264_nvenc")[0])
        self.assertFalse(CS.compare_ok(self.meta(3840, 2160, 30), two, "h264_nvenc")[0])   # formula of the design: W*H*fps*n
        ok, why = CS.compare_ok(self.meta(1920, 1080, 30), two, "libx264")
        self.assertFalse(ok)
        self.assertTrue(why)
        self.assertTrue(CS.compare_ok(self.meta(1280, 720, 30), two, "libx264")[0])
        self.assertFalse(CS.compare_ok(self.meta(1920, 1080, 30), ("processed",), "h264_nvenc")[0])


class TestOriginal(unittest.TestCase):
    def test_rebuild_no_mosaic_is_same_object(self):
        arr = img(5)
        self.assertIs(CS.rebuild_original(arr, make_frame(5)), arr)
        f = make_frame(5)
        f._lada_orig = []
        self.assertIs(CS.rebuild_original(arr, f), arr)

    def test_capture_and_rebuild(self):
        f = make_frame(7)
        f[1:3, 2:5] = 9                       # mosaic content
        CS.capture_boxes(f, [(1, 2, 2, 4)])
        f[1:3, 2:5] = 200                     # Lada blends in place
        orig = CS.rebuild_original(np.asarray(f), f)
        self.assertEqual(int(orig[1, 2, 0]), 9)
        self.assertEqual(int(orig[0, 0, 0]), 7)
        self.assertEqual(int(f[1, 2, 0]), 200)           # restored frame untouched
        self.assertIsNot(orig, f)

    def test_overlapping_boxes_restore_true_original(self):
        f = make_frame(1)
        f[0:3, 0:3] = 10
        CS.capture_boxes(f, [(0, 0, 2, 2), (1, 1, 3, 3)])   # clones taken before any blend
        f[:] = 99
        orig = CS.rebuild_original(np.asarray(f), f)
        self.assertEqual(int(orig[0, 0, 0]), 10)
        self.assertEqual(int(orig[1, 1, 0]), 10)
        self.assertEqual(int(orig[3, 3, 0]), 1)

    def test_install_hook(self):
        class R(FakeRestorer):
            pass
        self.assertTrue(CS.install_original_capture(R))
        self.assertFalse(CS.install_original_capture(R))
        r = R()
        f = make_frame(3)
        f[0:2, 0:2] = 4
        r._restore_frame(f, 0, [clip(0, (0, 0, 1, 1))])
        self.assertFalse(hasattr(f, "_lada_orig"))                 # capture off: nothing recorded
        self.assertEqual(int(f[0, 0, 0]), 255)

        r.capture_original = True
        f = make_frame(3)
        f[0:2, 0:2] = 4
        c_now, c_later = clip(0, (0, 0, 1, 1)), clip(5, (2, 2, 3, 3))
        out = r._restore_frame(f, 0, [c_now, c_later])             # only clips starting at this frame count
        self.assertIs(out, f)
        self.assertEqual(len(f._lada_orig), 1)
        self.assertEqual(int(CS.rebuild_original(np.asarray(f), f)[0, 0, 0]), 4)
        self.assertEqual(int(f[0, 0, 0]), 255)
        f2 = make_frame(3)                                         # no mosaic here: no attribute, no copy
        r._restore_frame(f2, 1, [c_later])
        self.assertFalse(hasattr(f2, "_lada_orig"))
        self.assertEqual(len(c_later.boxes), 1)


class TestQueueLayout(unittest.TestCase):
    """Helpers of the queue worker's storage layout v2. Live behaviour is covered above."""

    def test_live_semantics_unchanged(self):
        self.assertEqual(CS.layer_names(2), ("processed", "original", "restored"))
        self.assertEqual(CS.GOP, 15)
        self.assertIn("-g", CS.ffmpeg_layer_args(8, 4, ("processed", "original"), 1, "libx264")[2])

    def test_quality_offset(self):
        self.assertEqual([CS.quality_offset(q) for q in ("same", "small", "tiny")], [0, 4, 8])
        with self.assertRaises(ValueError):
            CS.quality_offset("big")

    def test_queue_layers(self):
        self.assertEqual(CS.queue_layers("copy"), ("processed",))
        self.assertEqual(CS.queue_layers(None), ("processed",))
        self.assertEqual(CS.queue_layers("encode"), ("processed", "original"))

    def test_queue_layer_args_1x(self):
        filt, maps, meta = CS.queue_layer_args(8, 4, ("processed", "original"), 1, Fraction(30))
        self.assertEqual(filt[0], "-filter_complex")
        self.assertEqual(filt[1], "[0:v]split=2[s0][s1];[s0]crop=8:4:0:0,format=yuv420p[v_processed];"
                                  "[s1]crop=8:4:0:4,format=yuv420p[v_original]")
        self.assertEqual(maps, ["-map", "[v_processed]", "-map", "[v_original]"])
        self.assertEqual(meta, CS.layer_meta_args(("processed", "original")))

    def test_queue_layer_args_2x_original_real_slots_only(self):
        filt, maps, meta = CS.queue_layer_args(8, 4, ("processed", "original"), 2, Fraction(30000, 1001))
        self.assertIn("[s0]crop=8:4:0:0,format=yuv420p[v_processed]", filt[1])           # processed untouched
        self.assertIn("[s1]crop=8:4:0:4,select='not(mod(n,2))',fps=30000/1001,format=yuv420p[v_original]", filt[1])
        self.assertNotIn("mpdecimate", filt[1])
        for bad in (("original", "processed"), ("processed",)):
            with self.assertRaises(ValueError):
                CS.queue_layer_args(8, 4, bad, 1, Fraction(30))

    def test_tags(self):
        t = CS.lada_tags(Fraction(30000, 1001), 2, 0.0333667, True, "copy")
        self.assertEqual(t, {"OH_SOURCE_FPS": "30000/1001", "OH_RATE_MULT": "2", "OH_VIDEO_START": "0.033367",
                             "OH_COMPARE": "1", "OH_ORIGINAL": "copy"})
        self.assertEqual(CS.lada_tags(Fraction(24), 1, 0, False, "copy")["OH_ORIGINAL"], "none")   # compare off wins
        self.assertEqual(CS.lada_tags(Fraction(24), 1, 0, True, None)["OH_ORIGINAL"], "none")
        self.assertEqual(CS.tag_args({"A": "1", "B": "2"}), ["-metadata", "A=1", "-metadata", "B=2"])

    def test_copy_eligibility(self):
        good = {"codec_name": "h264", "pix_fmt": "yuv420p", "r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001",
                "field_order": "progressive"}
        self.assertIsNone(CS.copy_eligibility(good))
        self.assertIsNone(CS.copy_eligibility(dict(good, codec_name="hevc", pix_fmt="yuvj420p", field_order="unknown")))
        self.assertIsNone(CS.copy_eligibility({k: v for k, v in good.items() if k != "field_order"}))
        self.assertIsNone(CS.copy_eligibility(dict(good, avg_frame_rate="29970/1000")))                 # within 0.1 %
        self.assertIsNone(CS.copy_eligibility(dict(good, tags={"rotate": "0"}, side_data_list=[{"rotation": 0}])))
        for bad, word in ((dict(good, codec_name="vp9"), "codec"), (dict(good, codec_name="av1"), "codec"),
                          (dict(good, pix_fmt="yuv420p10le"), "pixel"), (dict(good, pix_fmt="yuv422p"), "pixel"),
                          (dict(good, pix_fmt="yuv444p"), "pixel"), (dict(good, field_order="tt"), "interlaced"),
                          (dict(good, field_order="bb"), "interlaced"), (dict(good, tags={"rotate": "90"}), "rotation"),
                          (dict(good, side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}]), "rotation"),
                          (dict(good, avg_frame_rate="25/1"), "variable"), (dict(good, avg_frame_rate="0/0"), "frame rate"),
                          (dict(good, r_frame_rate="0/0"), "frame rate")):
            why = CS.copy_eligibility(bad)
            self.assertTrue(why and word in why, (bad, why))


class TestLuaContract(unittest.TestCase):
    def test_source_reset_is_not_overridden_by_compare_reset(self):
        with open(_STREAMER_ROOT / "src" / "oh_streamer" / "resources" / "oh_seek.lua", encoding="utf-8") as f:
            lua = f.read()
        self.assertEqual(lua.count('mp.register_script_message("oh-reset",'), 1)
        self.assertIn('if proc.reset_compare then step(proc.reset_compare) end', lua)
        self.assertIn('proc.reset_compare = function() cmp.reset() end', lua)

    def test_lua_accepts_published_compare_layer_titles(self):
        """Regression: Python publishes user-data/oh/compare_layers as track TITLES ("oh:original"); oh_seek.lua
        looks tracks up as "oh:" .. name, so it must strip the prefix or `o` never finds the original track."""
        import re
        with open(_STREAMER_ROOT / "src" / "oh_streamer" / "resources" / "oh_seek.lua", encoding="utf-8") as f:
            lua = f.read()
        m = re.search(r'for name in v:gmatch\("\[\^,%s\]\+"\) do (.*?) end', lua)
        self.assertTrue(m, "compare_layers observer not found")
        self.assertIn('gsub("^oh:"', m.group(1))
        self.assertTrue(all(t.startswith("oh:") for t in CS.LAYER_TITLES.values()))


if __name__ == "__main__":
    unittest.main()

