"""Runtime stand-ins for the lada site-packages edits (engines/lada/runtime_patches.py); CPU only, no lada import."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import inspect
import os
import types
import unittest
from unittest import mock

from oh_streamer.engines.lada import runtime_patches

_ENV_KEYS = ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "YOLO_OFFLINE")


def _fake_nms_module(default=0.05):
    """A module-like object whose non_max_suppression mimics ultralytics' signature and records what it got."""
    calls = []

    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300, nc=0, max_time_img=default, max_nms=30000):
        calls.append(max_time_img)
        return max_time_img

    return types.SimpleNamespace(non_max_suppression=non_max_suppression), calls


class ApplyEnvTests(unittest.TestCase):
    def test_sets_defaults(self):
        with mock.patch.dict(os.environ):
            for key in _ENV_KEYS:
                os.environ.pop(key, None)
            runtime_patches.apply_env()
            self.assertEqual(os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
            self.assertEqual(os.environ["YOLO_OFFLINE"], "true")

    def test_keeps_existing_values(self):
        with mock.patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "0", "YOLO_OFFLINE": "false"}):
            runtime_patches.apply_env()
            self.assertEqual(os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "0")
            self.assertEqual(os.environ["YOLO_OFFLINE"], "false")


class InstallNmsDefaultTests(unittest.TestCase):
    def test_default_becomes_0_3(self):
        module, calls = _fake_nms_module()
        self.assertTrue(runtime_patches.install_nms_default(module))
        self.assertEqual(module.non_max_suppression(object()), 0.3)
        self.assertEqual(module.non_max_suppression(object(), 0.25, 0.45, None, False, max_det=10, nc=2), 0.3)
        self.assertEqual(inspect.signature(module.non_max_suppression).parameters["max_time_img"].default, 0.3)

    def test_explicit_value_is_preserved(self):
        module, calls = _fake_nms_module()
        runtime_patches.install_nms_default(module)
        self.assertEqual(module.non_max_suppression(object(), max_time_img=0.01), 0.01)
        # positional: prediction .. nc are 9 parameters, max_time_img is the 10th
        self.assertEqual(module.non_max_suppression(object(), 0.25, 0.45, None, False, False, (), 300, 0, 0.7), 0.7)

    def test_idempotent(self):
        module, calls = _fake_nms_module()
        self.assertTrue(runtime_patches.install_nms_default(module))
        first = module.non_max_suppression
        self.assertFalse(runtime_patches.install_nms_default(module))
        self.assertIs(module.non_max_suppression, first)
        module.non_max_suppression(object())
        self.assertEqual(calls, [0.3])   # called once through a single wrapper

    def test_already_patched_site_packages_is_left_alone(self):
        module, calls = _fake_nms_module(default=0.3)
        original = module.non_max_suppression
        self.assertFalse(runtime_patches.install_nms_default(module))
        self.assertIs(module.non_max_suppression, original)
        self.assertEqual(module.non_max_suppression(object()), 0.3)

    def test_function_without_max_time_img_is_left_alone(self):
        module = types.SimpleNamespace(non_max_suppression=lambda prediction, conf_thres=0.25: None)
        original = module.non_max_suppression
        self.assertFalse(runtime_patches.install_nms_default(module))
        self.assertIs(module.non_max_suppression, original)


if __name__ == "__main__":
    unittest.main()
