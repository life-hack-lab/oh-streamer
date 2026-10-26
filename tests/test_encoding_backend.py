"""Encoder decisions/commands on any host; Mac hardware remains unverified."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

from fractions import Fraction
import shutil
import subprocess
import types
import unittest
from unittest.mock import Mock, patch

import oh_streamer.processing.compare_stack as CS
import oh_streamer.processing.encoding_backend as EB
import oh_streamer.platform.hls_cast as H
import oh_streamer.queue.queue_worker as Q


class SelectionTest(unittest.TestCase):
    def test_native_hardware_and_software_fallback(self):
        for platform, suffix in (("win32", "nvenc"), ("linux", "nvenc"), ("darwin", "videotoolbox")):
            for name, family, software in (("auto", "h264", "libx264"), ("h264", "h264", "libx264"),
                                           ("h264_nvenc", "h264", "libx264"), ("hevc", "hevc", "libx265"),
                                           ("h265", "hevc", "libx265"), ("hevc_nvenc", "hevc", "libx265")):
                with self.subTest(platform=platform, name=name):
                    probe = Mock(return_value=True)
                    self.assertEqual(EB.resolve_encoder(name, probe, platform=platform), family + "_" + suffix)
                    probe.assert_called_once_with(family + "_" + suffix)
                    self.assertEqual(EB.resolve_encoder(name, lambda _: False, platform=platform), software)

    def test_windows_automatic_backend_priority_and_software_fallback(self):
        probe = Mock(side_effect=lambda name: name == "h264_amf")
        self.assertEqual(EB.resolve_encoder("auto", probe, platform="win32"), "h264_amf")
        self.assertEqual([call.args[0] for call in probe.call_args_list], ["h264_nvenc", "h264_amf"])

        probe = Mock(return_value=False)
        self.assertEqual(EB.resolve_encoder("hevc", probe, platform="win32"), "libx265")
        self.assertEqual([call.args[0] for call in probe.call_args_list], ["hevc_nvenc", "hevc_amf"])

    def test_explicit_backend_intent_is_preserved(self):
        for encoder, fallback in (("h264_amf", "libx264"), ("hevc_amf", "libx265"),
                                  ("h264_nvenc", "libx264")):
            probe = Mock(return_value=False)
            self.assertEqual(EB.resolve_encoder(encoder, probe, platform="win32"), fallback)
            probe.assert_called_once_with(encoder)
        probe = Mock(return_value=True)
        self.assertEqual(EB.resolve_encoder("h264_amf", probe, platform="win32"), "h264_amf")
        probe.assert_called_once_with("h264_amf")
        probe = Mock(side_effect=AssertionError("AMF is only selected on Windows"))
        self.assertEqual(EB.resolve_encoder("h264_amf", probe, platform="linux"), "libx264")

    def test_explicit_apple_backend_and_software(self):
        for encoder, fallback in (("h264_videotoolbox", "libx264"), ("hevc_videotoolbox", "libx265")):
            self.assertEqual(EB.resolve_encoder(encoder, lambda _: True, platform="darwin"), encoder)
            self.assertEqual(EB.resolve_encoder(encoder, lambda _: False, platform="darwin"), fallback)
            probe = Mock(side_effect=AssertionError("must not probe unavailable Apple backend"))
            self.assertEqual(EB.resolve_encoder(encoder, probe, platform="win32"), fallback)
        for platform in ("win32", "darwin"):
            for encoder in ("libx264", "libx265"):
                probe = Mock(side_effect=AssertionError("explicit software must stay selected"))
                self.assertEqual(EB.resolve_encoder(encoder, probe, platform=platform), encoder)

    def test_names_and_queue_job_normalization(self):
        self.assertEqual(EB.normalize_encoder_name(" HEVC_VIDEOTOOLBOX "), "hevc_videotoolbox")
        self.assertEqual(EB.normalize_encoder_name("h265_videotoolbox"), "hevc_videotoolbox")
        self.assertEqual(EB.normalize_encoder_name(" H265_AMF "), "hevc_amf")
        self.assertEqual(EB.normalize_encoder_name("h264_amf"), "h264_amf")
        for bad in (None, 4, "", "av1"):
            self.assertIsNone(EB.normalize_encoder_name(bad))
            with self.assertRaises(ValueError):
                EB.resolve_encoder(bad)
        for encoder in ("h264_videotoolbox", "hevc_videotoolbox", "h264_amf", "hevc_amf"):
            job, _ = Q.normalize_job({"source": "s", "work_dir": "w", "output": "o", "settings": {"encoder": encoder}})
            self.assertEqual(job["settings"]["encoder"], encoder)
        self.assertEqual(Q.resolve_encoder("hevc_nvenc", lambda _: True, platform="darwin"), "hevc_videotoolbox")
        with self.assertRaises(Q.WorkerError):
            Q.resolve_encoder("bogus", lambda _: True)


class ProbeTest(unittest.TestCase):
    def test_probe_really_encodes_with_apple_options(self):
        with patch.object(EB.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
            self.assertTrue(EB.encoder_usable("hevc_videotoolbox", ffmpeg="/bin/ffmpeg", timeout=3.5))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "/bin/ffmpeg")
        self.assertIn("-i color=size=256x256:rate=30", " ".join(cmd))
        self.assertIn("-frames:v 3", " ".join(cmd))
        self.assertIn("-c:v hevc_videotoolbox", " ".join(cmd))
        self.assertIn("-allow_sw 0", " ".join(cmd))
        self.assertIn("-q:v 55", " ".join(cmd))
        self.assertIn("-flags +cgop", " ".join(cmd))
        self.assertIn("-force_key_frames expr:eq(mod(n,2),0)", " ".join(cmd))
        self.assertEqual(cmd[-3:], ["-f", "null", "-"])
        self.assertEqual(run.call_args.kwargs["timeout"], 3.5)

    def test_probe_failure_is_unavailable(self):
        for error in (FileNotFoundError("ffmpeg missing"), subprocess.TimeoutExpired("ffmpeg", 10),
                      OSError("cannot launch")):
            with patch.object(EB.subprocess, "run", side_effect=error):
                self.assertFalse(EB.encoder_usable("h264_videotoolbox"))
        with patch.object(EB.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)):
            self.assertFalse(EB.encoder_usable("h264_nvenc"))
        with patch.object(EB.subprocess, "run") as run:
            self.assertFalse(EB.encoder_usable("unknown"))
            run.assert_not_called()

    def test_probe_uses_supported_amf_cqp_and_idr_options(self):
        with patch.object(EB.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
            self.assertTrue(EB.encoder_usable("hevc_amf", ffmpeg="ffmpeg.exe", timeout=4))
        cmd = run.call_args.args[0]
        text = " ".join(cmd)
        self.assertIn("-c:v hevc_amf", text)
        self.assertIn("-usage transcoding", text)
        self.assertIn("-rc cqp -qp_i 26 -qp_p 26", text)
        self.assertIn("-header_insertion_mode idr", text)
        self.assertIn("-forced_idr 1", text)
        self.assertIn("-force_key_frames expr:eq(mod(n,2),0)", text)
        self.assertNotIn("-no-scenecut", cmd)
        self.assertEqual(run.call_args.kwargs["timeout"], 4)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_real_software_encode_probe(self):
        self.assertTrue(EB.encoder_usable("libx264"))


class ArgumentsTest(unittest.TestCase):
    def test_existing_live_arguments_are_exact(self):
        expected = {
            "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-pix_fmt", "yuv420p"],
            "h264_amf": ["-c:v", "h264_amf", "-usage", "lowlatency", "-latency", "1", "-quality", "speed",
                         "-rc", "cqp", "-qp_i", "20", "-qp_p", "20", "-bf", "0", "-pix_fmt", "yuv420p"],
            "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
        }
        for encoder, args in expected.items():
            self.assertEqual(EB.encoder_args(encoder), args)
            extra = ["-g", "15"] + (["-forced-idr", "1"] if encoder == "h264_nvenc" else
                                      ["-forced_idr", "1"] if encoder == "h264_amf" else [])
            self.assertEqual(CS.encoder_args(encoder, 15), args + extra)

    def test_existing_hls_arguments_are_exact(self):
        self.assertEqual(EB.encoder_args("h264_nvenc", purpose="hls"),
                         ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
                          "-no-scenecut", "1", "-forced-idr", "1"])
        self.assertEqual(EB.encoder_args("libx264", purpose="hls"),
                         ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-sc_threshold", "0", "-forced-idr", "1"])
        self.assertEqual(EB.encoder_args("h264_amf", purpose="hls"),
                         ["-c:v", "h264_amf", "-usage", "lowlatency", "-latency", "1", "-quality", "speed",
                          "-rc", "vbr_peak", "-forced_idr", "1"])

    def test_existing_segmented_arguments_are_exact(self):
        for encoder in ("h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "libx264", "libx265"):
            args = ["-c:v", encoder]
            if encoder.endswith("_nvenc"):
                args += ["-preset", "p5", "-rc", "vbr", "-b:v", "0", "-no-scenecut", "1", "-forced-idr", "1"]
            elif encoder.endswith("_amf"):
                args += ["-usage", "transcoding", "-quality", "balanced", "-forced_idr", "1"]
                if encoder == "hevc_amf":
                    args += ["-header_insertion_mode", "idr"]
            else:
                args += ["-preset", "veryfast"]
                if encoder == "libx264":
                    args += ["-sc_threshold", "0", "-forced-idr", "1"]
            args += ["-bf", "0", "-pix_fmt", "yuv420p"]
            self.assertEqual(EB.encoder_args(encoder, purpose="segmented"), args)

    def test_videotoolbox_options_do_not_leak_other_backends(self):
        forbidden = {"-rc", "-cq", "-crf", "-preset", "-forced-idr", "-no-scenecut", "-sc_threshold", "-x265-params"}
        for encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
            for purpose in ("live", "compare", "segmented"):
                args = EB.encoder_args(encoder, purpose=purpose, gop=15)
                self.assertFalse(forbidden.intersection(args))
                self.assertIn("-flags +cgop", " ".join(args))
                self.assertIn("-allow_sw 0", " ".join(args))
                self.assertIn("-bf 0 -pix_fmt yuv420p", " ".join(args))
                self.assertIn("-g 15", " ".join(args))
        with self.assertRaises(ValueError):
            EB.encoder_args("hevc_videotoolbox", purpose="hls")

    def test_quality_offsets_reduce_original_quality(self):
        for encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
            self.assertEqual(Q.quality_args(encoder), ["-q:v", "55"])
            self.assertEqual(Q.quality_args(encoder, ":v:1", 8), ["-q:v:1", "47"])
            self.assertEqual(Q.quality_args(encoder, "", 100), ["-q:v", "1"])
            self.assertEqual(Q.quality_args(encoder, "", -100), ["-q:v", "100"])
        self.assertEqual(Q.quality_args("hevc_nvenc", ":v:1", 8), ["-cq:v:1", "31"])
        self.assertEqual(Q.quality_args("libx265", ":v:1", 8), ["-crf:v:1", "32"])
        self.assertEqual(Q.quality_args("h264_amf", ":v:1", 8),
                         ["-rc:v:1", "cqp", "-qp_i:v:1", "30", "-qp_p:v:1", "30"])
        self.assertEqual(Q.quality_args("hevc_amf", "", 100),
                         ["-rc", "cqp", "-qp_i", "51", "-qp_p", "51"])
        self.assertEqual(Q.quality_args("hevc_amf", "", -100),
                         ["-rc", "cqp", "-qp_i", "0", "-qp_p", "0"])

    def test_amf_segment_gop_and_hls_commands_use_forced_idrs(self):
        plan = Q.SegmentPlan(Fraction(30), 300, 0.0, 4)
        args = Q.build_encode_cmd(plan, 1, 1280, 720, "h264_amf", "segments", 1,
                                  ("processed",), cq_offset=0)
        text = " ".join(args)
        self.assertIn("-c:v h264_amf", text)
        self.assertIn("-rc cqp -qp_i 22 -qp_p 22", text)
        self.assertIn("-g 120 -force_key_frames expr:eq(mod(n,120),0)", text)
        self.assertIn("-forced_idr 1", text)
        self.assertNotIn("-no-scenecut", args)

        hls_plan = H.SegmentPlan(Fraction(30), 300, 1280, 720)
        args = H.build_ffmpeg_cmd(hls_plan, 0, 1280, 720, "h264_amf", "segments", None)
        text = " ".join(args)
        self.assertIn("-c:v h264_amf", text)
        self.assertIn("-usage lowlatency -latency 1", text)
        self.assertIn("-rc vbr_peak", text)
        self.assertNotIn("-qp_i", args)
        self.assertIn("-forced_idr 1", text)
        self.assertIn("-g 120 -force_key_frames expr:eq(mod(n,120),0)", text)
        self.assertIn("-maxrate 20M -bufsize 40M", text)

    def test_queue_multiple_tracks_gop_and_quality(self):
        plan = Q.SegmentPlan(Fraction(30), 300, 0.0, 4)
        for encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
            args = Q.build_encode_cmd(plan, 2, 1280, 1440, encoder, "segments", 2,
                                      ("processed", "original"), cq_offset=8)
            text = " ".join(args)
            self.assertIn("-c:v " + encoder, text)
            self.assertIn("-g:v:0 240 -force_key_frames:v:0 expr:eq(mod(n,240),0)", text)
            self.assertIn("-g:v:1 120 -force_key_frames:v:1 expr:eq(mod(n,120),0)", text)
            self.assertIn("-q:v:0 55", text)
            self.assertIn("-q:v:1 47", text)
            self.assertNotIn("-forced-idr", args)
            self.assertIn("-segment_start_number 2", text)
            self.assertLess(float(args[args.index("-segment_time") + 1]), 4.0)

    def test_queue_multiple_tracks_amf_quality_stream_specifiers(self):
        plan = Q.SegmentPlan(Fraction(30), 300, 0.0, 4)
        args = Q.build_encode_cmd(plan, 2, 1280, 1440, "hevc_amf", "segments", 2,
                                  ("processed", "original"), cq_offset=8)
        text = " ".join(args)
        self.assertIn("-rc:v:0 cqp -qp_i:v:0 26 -qp_p:v:0 26", text)
        self.assertIn("-rc:v:1 cqp -qp_i:v:1 34 -qp_p:v:1 34", text)
        self.assertIn("-g:v:0 240 -force_key_frames:v:0 expr:eq(mod(n,240),0)", text)
        self.assertIn("-g:v:1 120 -force_key_frames:v:1 expr:eq(mod(n,120),0)", text)
        self.assertIn("-forced_idr 1", text)

    def test_hls_apple_h264_preserves_cast_boundary_controls(self):
        plan = H.SegmentPlan(Fraction(30), 300, 1280, 720)
        args = H.build_ffmpeg_cmd(plan, 2, 1280, 720, "h264_videotoolbox", "segments", None)
        text = " ".join(args)
        self.assertIn("-c:v h264_videotoolbox", text)
        self.assertIn("-g 120 -force_key_frames expr:eq(mod(n,120),0)", text)
        self.assertIn("-bf 0 -profile:v high", text)
        self.assertIn("-flags +cgop", text)
        self.assertNotIn("-forced-idr", args)
        self.assertIn("-maxrate", args)
        # HLS keeps its H.264 contract for queue/export HEVC choices.
        args = H.build_ffmpeg_cmd(plan, 0, 1280, 720, "hevc_videotoolbox", "segments", None)
        self.assertEqual(args[args.index("-c:v") + 1], "libx264")


class CompareGuardTest(unittest.TestCase):
    @staticmethod
    def meta(w, h, fps):
        return types.SimpleNamespace(video_width=w, video_height=h, video_fps_exact=Fraction(fps))

    def test_apple_is_hardware_with_conservative_multitrack_limit(self):
        layers = ("processed", "original")
        for encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
            self.assertTrue(EB.is_hardware_encoder(encoder))
            self.assertTrue(CS.compare_ok(self.meta(1920, 1080, 30), layers, encoder)[0])
            ok, reason = CS.compare_ok(self.meta(1920, 1080, 60), layers, encoder)
            self.assertFalse(ok)
            self.assertIn("VideoToolbox", reason)
            info = Q.SourceInfo(Fraction(30), 300, 1920, 1080, 0.0, 10.0, True)
            self.assertTrue(Q.compare_plan({"compare": "on", "compare_original": "encode"}, info, encoder).on)
        self.assertTrue(CS.compare_ok(self.meta(1920, 1080, 60), layers, "hevc_nvenc")[0])
        self.assertFalse(CS.compare_ok(self.meta(1920, 1080, 30), layers, "libx265")[0])


if __name__ == "__main__":
    unittest.main()
