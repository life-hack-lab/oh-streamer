"""Behavior of the standalone AMD development probe without GPU dependencies."""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))

import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import probe_amd as P
from oh_streamer.processing import encoding_backend as E


class TorchReportTests(unittest.TestCase):
    def test_nvidia_build_is_unavailable_for_amd_and_skips_gpu_ops(self):
        checks = [
            P._check("torch-backend", "unavailable", "NVIDIA CUDA PyTorch build detected", {"backend": "nvidia-cuda"}, required=True),
            P._check("torch-tensor", "skip", "requires a reachable ROCm/HIP GPU"),
            P._check("torchvision-deform-conv2d", "skip", "requires a reachable ROCm/HIP GPU"),
        ]
        payload = {"backend": "nvidia-cuda", "selected_device": None}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, "_worker_check", return_value=(payload, *checks)), \
                patch.object(P, "_amf_encode", return_value=P._check("amf-encode", "pass", "ok")):
            report = P.build_report(root=directory, skip_amf=True)
        self.assertEqual(report["status"], "unavailable")
        by_id = {row["id"]: row for row in report["checks"]}
        self.assertEqual(by_id["torch-backend"]["status"], "unavailable")
        self.assertEqual(by_id["torch-tensor"]["status"], "skip")
        self.assertEqual(by_id["torchvision-deform-conv2d"]["status"], "skip")

    def test_rocm_tensor_and_operator_success_can_pass_with_explicitly_skipped_amf(self):
        payload = {"backend": "rocm-hip", "selected_device": "cuda:0"}
        checks = [
            P._check("torch-backend", "pass", "ROCm/HIP GPU", required=True),
            P._check("torch-tensor", "pass", "GPU tensor arithmetic passed", required=True),
            P._check("torchvision-deform-conv2d", "pass", "operator ran on GPU", required=True),
        ]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, "_worker_check", return_value=(payload, *checks)):
            report = P.build_report(root=directory, skip_amf=True)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["selected_device"], "cuda:0")
        self.assertTrue(all(row["status"] == "pass" for row in report["checks"] if row["required"]))

    def test_child_environment_does_not_let_pythonpath_shadow_venv_torch(self):
        with patch.dict(P.os.environ, {"PYTHONPATH": "C:/old/nvidia/site-packages"}):
            self.assertNotIn("PYTHONPATH", P._child_env("."))

    def test_rife_uses_supported_cli_arguments_and_reports_missing_optional_dependency(self):
        output = '{"status":"fail","error":"ModuleNotFoundError: no module named example"}'
        result = {"returncode": 1, "timed_out": False, "elapsed_s": .2,
                  "stdout": output, "stderr": ""}
        with patch.object(P, "run_command", return_value=result) as run:
            check = P._rife_check(".", "cuda:0", 120.0, {})
        command = run.call_args.args[0]
        self.assertIn("--device", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command[command.index("--size") + 1], "64x64")
        self.assertEqual(command[command.index("--iterations") + 1], "1")
        self.assertEqual(command[command.index("--timeout") + 1], "120.0")
        self.assertIn("--json", command)
        self.assertEqual(check["status"], "unavailable")
        self.assertEqual(check["details"]["worker_returncode"], 1)

    def test_optional_model_smoke_is_skipped_after_gpu_tensor_failure(self):
        payload = {"backend": "rocm-hip", "selected_device": "cuda:0"}
        checks = [
            P._check("torch-backend", "pass", "HIP GPU", required=True),
            P._check("torch-tensor", "fail", "hipErrorInvalidKernelFile", required=True),
            P._check("torchvision-deform-conv2d", "fail", "kernel unavailable", required=True),
        ]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, "_worker_check", return_value=(payload, *checks)), \
                patch.object(P, "_optional_model_checks") as model_checks:
            report = P.build_report(root=directory, skip_amf=True, model_smoke=True, rife=True,
                                    scratch_root=directory)
        model_checks.assert_not_called()
        by_id = {row["id"]: row for row in report["checks"]}
        self.assertEqual(by_id["model-basicvsrpp"]["status"], "skip")
        self.assertIn("blocked", by_id["model-yolo"]["message"])
        self.assertEqual(by_id["model-rife"]["status"], "skip")

    def test_relative_reference_site_packages_resolve_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            project = _Path(directory)
            dependency_dir = project / "lada" / ".venv" / "Lib" / "site-packages"
            dependency_dir.mkdir(parents=True)
            payload = {"restoration_status": "unavailable", "detector_status": "unavailable",
                       "restoration_error": "missing", "detector_error": "missing"}
            with patch.object(P, "_invoke_json", return_value=payload) as invoke:
                P._optional_model_checks(project, "cuda:0", 10, directory, None, "ffmpeg",
                                         ["lada/.venv/Lib/site-packages"], {})
            command = invoke.call_args.args[0]
            index = command.index("--reference-site-packages")
            self.assertEqual(command[index + 1], str(dependency_dir.resolve()))


class EncoderTests(unittest.TestCase):
    def test_short_amf_encode_uses_project_live_encoder_settings(self):
        listing = " V..... h264_amf AMD AMF H.264 Encoder\n"

        def encode(argv, **_kwargs):
            _Path(argv[-1]).write_bytes(b"encoded")
            return {"returncode": 0, "timed_out": False, "elapsed_s": .2,
                    "stdout": "", "stderr": ""}

        valid_media = {"streams": [{"codec_type": "video", "codec_name": "h264"}],
                       "format": {"duration": "2.0"}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, "_encoder_list", return_value=listing), \
                patch.object(P, "run_command", side_effect=encode) as run, \
                patch.object(P, "_probe_media", return_value=valid_media):
            result = P._amf_encode("ffmpeg", "ffprobe", source=None, duration=2.0,
                                   scratch=directory, timeout=5, env={})
        self.assertEqual(result["status"], "pass")
        command = run.call_args.args[0]
        encoder_options = E.encoder_args("h264_amf", purpose="live")
        start = command.index("-c:v")
        self.assertEqual(command[start:start + len(encoder_options)], encoder_options)

    def test_segment_probe_normalizes_fps_and_uses_shared_segment_settings(self):
        listing = " V..... h264_amf AMD AMF H.264 Encoder\n"
        packet_data = {"packets": [{"pts_time": "0.000", "flags": "K_"},
                                    {"pts_time": "0.500", "flags": "__"}]}

        def encode(argv, **_kwargs):
            segment_dir = _Path(argv[-1]).parent
            segment_dir.mkdir(parents=True, exist_ok=True)
            (segment_dir / "segment_00.mkv").write_bytes(b"0")
            (segment_dir / "segment_01.mkv").write_bytes(b"1")
            return {"returncode": 0, "timed_out": False, "elapsed_s": .2,
                    "stdout": "", "stderr": ""}

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, "_encoder_list", return_value=listing), \
                patch.object(P, "run_command", side_effect=encode) as run, \
                patch.object(P, "_probe_media", return_value=packet_data):
            result = P._segment_probe("ffmpeg", "ffprobe", source="clip.mp4", duration=4,
                                      scratch=directory, timeout=5, env={})
        self.assertEqual(result["status"], "pass")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-vf") + 1], "fps=30")
        expected = (E.encoder_args("h264_amf", purpose="segmented") +
                    E.quality_args("h264_amf") + E.gop_args("h264_amf", 30))
        start = command.index("-c:v")
        self.assertEqual(command[start:start + len(expected)], expected)

    def test_basicvsrpp_receives_a_string_weight_path(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed to exercise the model worker")

        with tempfile.TemporaryDirectory() as directory:
            root = _Path(directory)
            weights = root / "lada" / "model_weights" / "lada_mosaic_restoration_model_generic_v1.2.pth"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"stub")

            fake_torch = types.ModuleType("torch")
            fake_torch.__version__ = "test-rocm"
            fake_torch.version = types.SimpleNamespace(hip="10.1", cuda=None)
            fake_torch.device = lambda name: name
            fake_torch.cuda = types.SimpleNamespace(synchronize=lambda _device: None)
            inference_module = types.ModuleType("lada.models.basicvsrpp.inference")
            load_model = Mock(return_value=object())
            inference_module.load_model = load_model
            inference_module.inference = lambda _model, frames, _device: frames
            lada = types.ModuleType("lada")
            lada.__path__ = []
            models = types.ModuleType("lada.models")
            models.__path__ = []
            basicvsr = types.ModuleType("lada.models.basicvsrpp")
            basicvsr.__path__ = []
            lada.models = models
            models.basicvsrpp = basicvsr
            basicvsr.inference = inference_module
            modules = {"torch": fake_torch, "lada": lada, "lada.models": models,
                       "lada.models.basicvsrpp": basicvsr,
                       "lada.models.basicvsrpp.inference": inference_module}
            lada_root = str((root / "lada").resolve())
            with patch.dict("sys.modules", modules):
                try:
                    result = P._model_worker(root, "cuda:0", None, [])
                finally:
                    if lada_root in P.sys.path:
                        P.sys.path.remove(lada_root)
        self.assertEqual(result["restoration_status"], "pass")
        self.assertIsInstance(load_model.call_args.args[1], str)
        self.assertEqual(load_model.call_args.args[1], str(weights))

    def test_listed_amf_encoder_does_not_pass_when_real_encode_fails(self):
        listing = " V..... h264_amf AMD AMF H.264 Encoder\n"
        failure = {"returncode": 1, "timed_out": False, "elapsed_s": .2,
                   "stdout": "", "stderr": "AMF device init failed"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, "_encoder_list", return_value=listing), \
                patch.object(P, "run_command", return_value=failure) as run:
            result = P._amf_encode("ffmpeg", "ffprobe", source=None, duration=2,
                                   scratch=directory, timeout=5, env={})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result["status"], "fail")
        self.assertIn("actual short encode failed", result["message"])

    def test_encoder_missing_from_ffmpeg_is_unavailable(self):
        with patch.object(P, "_encoder_list", return_value=" V..... libx264\n"), \
                patch.object(P, "run_command") as run:
            result = P._amf_encode("ffmpeg", "ffprobe", source=None, duration=2,
                                   scratch=".", timeout=5, env={})
        run.assert_not_called()
        self.assertEqual(result["status"], "unavailable")

    def test_segment_packet_validation_requires_keyframe_and_monotonic_pts(self):
        valid = {"packets": [{"pts_time": "0.000", "flags": "K_"},
                              {"pts_time": "0.033", "flags": "__"}]}
        result = P.validate_segment_packets(valid)
        self.assertEqual(result["packet_count"], 2)
        self.assertTrue(result["starts_with_keyframe"])
        with self.assertRaisesRegex(ValueError, "keyframe"):
            P.validate_segment_packets({"packets": [{"pts_time": "0", "flags": "__"}]})
        with self.assertRaisesRegex(ValueError, "monotonic"):
            P.validate_segment_packets({"packets": [{"pts_time": "0", "flags": "K_"},
                                                      {"pts_time": "0.2", "flags": "__"},
                                                      {"pts_time": "0.1", "flags": "__"}]})


if __name__ == "__main__":
    unittest.main()
