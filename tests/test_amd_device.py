"""ROCm/HIP uses PyTorch's CUDA device API while keeping unvalidated RIFE paths opt-in."""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import builtins
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import oh_streamer.processing.device_backend as D
import oh_streamer.processing.rife_interp as R


def torch_fake(*, hip=None):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
        device=lambda name: SimpleNamespace(type=name.split(":", 1)[0], name=name),
    )


class FakeCudaDevice:
    type = "cuda"

    def __str__(self):
        return "cuda:0"


class RocmDevicePolicyTest(unittest.TestCase):
    def test_auto_selection_uses_cuda_api_and_identifies_hip(self):
        torch = torch_fake(hip="6.3.0")

        device = D.select_device("auto", torch_module=torch)

        self.assertEqual(device.name, "cuda:0")
        self.assertEqual(device.type, "cuda")
        self.assertTrue(D.rife_available(device))
        self.assertTrue(D.is_rocm(torch))
        self.assertEqual(D.backend_name(device, torch_module=torch), "ROCm/HIP")
        self.assertEqual(D.backend_name(device, torch_module=torch_fake()), "CUDA")

    def test_rocm_rife_defaults_are_eager_but_explicit_opt_in_is_kept(self):
        defaults = R._backend_options(
            "cuda:0", fp16=None, channels_last=None, cuda_graph=None, own_stream=None,
            experimental_mps=False, rocm=True)
        self.assertEqual(defaults, dict(fp16=True, channels_last=True, cuda_graph=False, own_stream=False))

        opted_in = R._backend_options(
            "cuda:0", fp16=None, channels_last=None, cuda_graph=True, own_stream=True,
            experimental_mps=False, rocm=True)
        self.assertTrue(opted_in["cuda_graph"])
        self.assertTrue(opted_in["own_stream"])

    def test_nvidia_cuda_defaults_stay_unchanged(self):
        options = R._backend_options(
            "cuda:1", fp16=None, channels_last=None, cuda_graph=None, own_stream=None,
            experimental_mps=False, rocm=False)
        self.assertEqual(options, dict(fp16=True, channels_last=True, cuda_graph=True, own_stream=True))

    def test_fp16_default_skips_nvidia_capability_probe_on_hip(self):
        device = SimpleNamespace(type="cuda")
        nvidia_check = Mock(return_value=True)

        self.assertFalse(D.fp16_default(device, nvidia_check, torch_module=torch_fake(hip="6.3.0")))
        nvidia_check.assert_not_called()
        self.assertTrue(D.fp16_default(device, nvidia_check, torch_module=torch_fake()))
        nvidia_check.assert_called_once_with(device)

    def test_interpolator_does_not_import_torch_until_build(self):
        real_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("Interpolator.__init__ imported torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_torch):
            interp = R.Interpolator(SimpleNamespace(type="cuda", __str__=lambda _self: "cuda:0"),
                                    log=lambda _message: None)
        self.assertTrue(interp._options["cuda_graph"])
        self.assertTrue(interp._options["own_stream"])

    def test_build_applies_rocm_defaults_and_labels_backend(self):
        torch = types.ModuleType("torch")
        torch.version = SimpleNamespace(hip="6.3.0")
        torch.uint8 = object()
        torch.zeros = Mock(return_value=object())
        torch.full_like = Mock(return_value=object())
        rife = SimpleNamespace(pw=32, ph=32, graph=None, capture_error=None, middle=Mock(), _last=True, _p1=True)
        device = FakeCudaDevice()
        interp = R.Interpolator(device, log=lambda _message: None)
        interp._key = (32, 32)

        with patch.dict(sys.modules, {"torch": torch}), \
                patch.object(interp, "_weights", return_value={}), \
                patch.object(R, "Rife", return_value=rife) as constructor:
            interp._build(32, 32)

        self.assertIsNone(interp.error_for(32, 32))
        self.assertFalse(constructor.call_args.kwargs["cuda_graph"])
        self.assertFalse(constructor.call_args.kwargs["own_stream"])
        self.assertTrue(constructor.call_args.kwargs["fp16"])
        self.assertIn("cuda:0 (ROCm/HIP) FP16 eager", interp.info)


if __name__ == "__main__":
    unittest.main()
