"""Backend policy plus small real CPU/CUDA RIFE checks (optional weights/devices explicitly skip)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import oh_streamer.processing.rife_interp as R

try:
    import torch
except ImportError:
    torch = None


def options(device, **overrides):
    values = dict(fp16=None, channels_last=None, cuda_graph=None, own_stream=None, experimental_mps=False)
    values.update(overrides)
    return R._backend_options(device, **values)


class BackendPolicyTest(unittest.TestCase):
    def test_defaults_and_ignored_cuda_requests(self):
        self.assertEqual(options("cpu", cuda_graph=True, own_stream=True),
                         dict(fp16=False, channels_last=False, cuda_graph=False, own_stream=False))
        self.assertEqual(options("cuda:1"),
                         dict(fp16=True, channels_last=True, cuda_graph=True, own_stream=True))
        self.assertEqual(options("mps", experimental_mps=True, cuda_graph=True, own_stream=True),
                         dict(fp16=False, channels_last=False, cuda_graph=False, own_stream=False))

    def test_unsafe_cpu_fp16_and_unvalidated_mps_require_explicit_choices(self):
        with self.assertRaisesRegex(ValueError, "CPU inference requires FP32"):
            options("cpu", fp16=True)
        with self.assertRaisesRegex(ValueError, "MPS is unvalidated"):
            R.Interpolator("mps")
        with self.assertRaisesRegex(ValueError, "does not support"):
            options("xpu")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_interpolator_actual_constructor_options_and_labels(self):
        for device, kwargs, expected in (
            ("cpu", {}, "cpu FP32 eager"),
            ("mps", {"experimental_mps": True}, "experimental MPS, Mac validation pending"),
            ("cuda:1", {}, "cuda:1 FP16 CUDA graph"),
            ("cuda:0", {"cuda_graph": False, "fp16": False}, "cuda:0 FP32 eager"),
        ):
            with self.subTest(device=device, kwargs=kwargs):
                interp = R.Interpolator(device, log=lambda _: None, **kwargs)
                fake = SimpleNamespace(pw=32, ph=32, graph=object() if interp._options["cuda_graph"] else None,
                                       capture_error=None, middle=lambda *_: None)
                interp._key = (32, 32)
                with patch.object(R, "Rife", return_value=fake) as constructor, \
                        patch.object(interp, "_weights", return_value={}):
                    interp._build(32, 32)
                self.assertIsNone(interp.error_for(32, 32))
                self.assertIs(interp.get(32, 32), fake)
                self.assertIn(expected, interp.info)
                for key, value in interp._options.items():
                    self.assertEqual(constructor.call_args.kwargs[key], value)


@unittest.skipIf(torch is None, "torch is not installed")
class ProtectStaticTest(unittest.TestCase):
    """protect_static: static text over a moving background comes from the source frames, everything else from RIFE."""

    def frames(self, h=180, w=320):
        g = torch.Generator().manual_seed(0)
        bg = torch.rand((1, 3, h, w + 8), generator=g)
        f0, f1 = bg[..., :w].clone(), bg[..., 8:].clone()           # background moves 8 px
        text = torch.zeros((1, 3, 20, 120))
        text[..., ::4, :] = 1.0                                       # stripes: high-contrast "strokes"
        text[..., :, ::5] = 1.0
        for f in (f0, f1):
            f[..., 150:170, 100:220] = text                           # same overlay in both frames
        return f0, f1

    def test_static_text_from_sources_moving_background_from_rife(self):
        f0, f1 = self.frames()
        out = torch.full_like(f0, 0.5)                                # "RIFE": anything different from the sources
        res = R.protect_static(out, f0, f1, torch.nn.functional)
        torch.testing.assert_close(res[..., 152:168, 104:216], f0[..., 152:168, 104:216])
        torch.testing.assert_close(res[..., :120, :], out[..., :120, :])   # moving area untouched

    def test_nothing_static_or_flat_static_is_untouched(self):
        f0, f1 = self.frames()
        f1[..., 150:170, 100:220] = 0.3                               # the "text" changes -> not static
        out = torch.full_like(f0, 0.5)
        torch.testing.assert_close(R.protect_static(out, f0, f1, torch.nn.functional), out)
        flat = torch.full_like(f0, 0.7)                               # static but no edges (wall, sky)
        torch.testing.assert_close(R.protect_static(out, flat, flat.clone(), torch.nn.functional), out)

    def test_grid_resolution_for_tall_frames(self):
        f0, f1 = self.frames(1080, 1920)
        out = torch.full_like(f0, 0.5)
        res = R.protect_static(out, f0, f1, torch.nn.functional)
        self.assertEqual(res.shape, out.shape)
        torch.testing.assert_close(res[..., 152:168, 104:216], f0[..., 152:168, 104:216])
        torch.testing.assert_close(res[..., 400:, :], out[..., 400:, :])


@unittest.skipIf(torch is None, "torch is not installed")
class RealInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.old_threads))
        path = R.RIFE_DIR / "weights" / "flownet_v4.6.pkl"
        if not path.is_file():
            torch.set_num_threads(cls.old_threads)
            raise unittest.SkipTest(f"optional RIFE weights missing: {path}")
        cls.sd = R.load_state_dict("4.6")
        cls.frames = [((torch.arange(25 * 33 * 3).reshape(25, 33, 3) % 128) + 32 + i * 4).to(torch.uint8)
                      for i in range(3)]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_cpu_fp32_real_output_padding_and_scene_cut_without_cuda_calls(self):
        with patch.object(torch.cuda, "Stream", side_effect=AssertionError("CPU called CUDA Stream")), \
                patch.object(R.Rife, "_capture", side_effect=AssertionError("CPU called CUDA capture")):
            rife = R.Rife("4.6", 33, 25, 1.0, "cpu", state_dict=self.sd, cuda_graph=True, own_stream=True)
            self.assertEqual(rife.dtype, torch.float32)
            self.assertIsNone(rife.stream)
            self.assertIsNone(rife.graph)
            with torch.inference_mode():
                self.assertIsNone(rife.step(rife.prep(self.frames[0])))
                raw = rife.step(rife.prep(self.frames[1]))
                self.assertEqual(raw.dtype, torch.float32)
                self.assertEqual(tuple(raw.shape), (1, 3, 32, 64))
                self.assertTrue(torch.isfinite(raw).all().item())
                out = rife.middle(self.frames[0], self.frames[1])
                self.assertIsNotNone(out)
                self.assertEqual(out.shape, (25, 33, 3))
                self.assertEqual(str(out.dtype), "uint8")
                self.assertIsNone(rife.middle(torch.zeros_like(self.frames[0]), torch.full_like(self.frames[0], 255)))

    def test_real_cpu_interpolator_defaults_and_ready_label(self):
        interp = R.Interpolator("cpu", log=lambda _: None)
        interp._sd = self.sd
        interp.prepare(33, 25)
        interp.wait_idle(timeout=15)
        self.assertFalse(interp.building())
        self.assertIsNone(interp.error_for(33, 25))
        self.assertIsNotNone(interp.get(33, 25))
        self.assertIn("cpu FP32 eager", interp.info)
        self.assertNotIn("CUDA graph", interp.info)
        interp.release()
        self.assertIsNone(interp.get(33, 25))

    def _raw_sequence(self, rife):
        result = []
        with rife._ctx(), torch.inference_mode():
            for frame in self.frames:
                out = rife.step(rife.prep(frame))
                if out is not None:
                    result.append(out.detach().float().cpu().clone())
        torch.cuda.synchronize(rife.device)
        return result

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_eager_and_graph_same_weights_both_model_families(self):
        device = torch.device("cuda:0")
        for model in ("4.6", "4.22.lite"):
            if not (R.RIFE_DIR / "weights" / f"flownet_v{model}.pkl").is_file():
                continue
            with self.subTest(model=model):
                sd = self.sd if model == "4.6" else R.load_state_dict(model)
                eager = R.Rife(model, 33, 25, 1.0, device, fp16=True, channels_last=True, state_dict=sd,
                               own_stream=True)
                graph = R.Rife(model, 33, 25, 1.0, device, fp16=True, channels_last=True, state_dict=sd,
                               cuda_graph=True, own_stream=True)
                self.assertIsNotNone(graph.graph, graph.capture_error)
                for expected, actual in zip(self._raw_sequence(eager), self._raw_sequence(graph)):
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
                for rife in (eager, graph):
                    out = rife.middle(self.frames[0], self.frames[1])
                    self.assertEqual(out.shape, (25, 33, 3))
                    self.assertIsNone(rife.middle(torch.zeros_like(self.frames[0]),
                                                 torch.full_like(self.frames[0], 255)))
                torch.cuda.synchronize(device)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_numerical_equivalence_to_original_wrapper_if_available(self):
        original = _STREAMER_ROOT / "src" / "oh_streamer" / "processing" / "rife_interp.py"
        if not original.is_file() or original.resolve() == Path(R.__file__).resolve():
            self.skipTest("separate original CUDA wrapper is unavailable")
        spec = importlib.util.spec_from_file_location("original_rife_interp", original)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        device = torch.device("cuda:0")
        for graph_mode in (False, True):
            with self.subTest(cuda_graph=graph_mode):
                kw = dict(fp16=True, channels_last=True, cuda_graph=graph_mode, state_dict=self.sd, own_stream=True)
                old = baseline.Rife("4.6", 33, 25, 1.0, device, **kw)
                new = R.Rife("4.6", 33, 25, 1.0, device, **kw)
                if graph_mode:
                    self.assertIsNotNone(old.graph, old.capture_error)
                    self.assertIsNotNone(new.graph, new.capture_error)
                for expected, actual in zip(self._raw_sequence(old), self._raw_sequence(new)):
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


class ProbeCliTest(unittest.TestCase):
    def test_invalid_iteration_and_size_bounds(self):
        import probe_rife as probe
        import argparse
        for value in ("0", "21"):
            with self.assertRaises(argparse.ArgumentTypeError):
                probe._iterations(value)
        for value in ("0x32", "4096x4096", "x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                probe._size(value)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_cli_cpu_json_success_and_explicit_fp16_failure(self):
        if not (R.RIFE_DIR / "weights" / "flownet_v4.6.pkl").is_file():
            self.skipTest("optional RIFE weights missing")
        for fp16, status, code in (("auto", "pass", 0), ("on", "fail", 1)):
            command = [sys.executable, str(_STREAMER_ROOT / "probe_rife.py"), "--device", "cpu",
                       "--size", "33x25", "--iterations", "1", "--fp16", fp16, "--json", "--timeout", "30"]
            child = subprocess.run(command, capture_output=True, text=True, timeout=40)
            self.assertEqual(child.returncode, code, child.stderr + child.stdout)
            report = json.loads(child.stdout)
            self.assertEqual(report["status"], status)
            self.assertEqual(report["mac_validation"], "unvalidated")
            self.assertFalse(report["mac_player_rife_enabled"])
            if status == "pass":
                self.assertEqual(report["precision"], "fp32")
                self.assertTrue(all(report["checks"].values()))
            else:
                self.assertIn("CPU inference requires FP32", report["error"])


if __name__ == "__main__":
    unittest.main()
