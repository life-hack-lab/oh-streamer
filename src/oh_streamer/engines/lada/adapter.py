"""Lada engine adapter: the model factory and the names the player gets from Lada.

Moved from player/oh_player.py (engine-decoupling phase 2); behavior unchanged. This is the one place that imports
Lada's restoration pipeline (importing it initializes Lada's ModelFiles registry, so the caller must have set
LADA_MODEL_WEIGHTS_DIR before the first import).
"""

import gc
from dataclasses import replace
from fractions import Fraction

from oh_streamer.engines.lada import runtime_patches

runtime_patches.apply_env()   # before torch / lada / ultralytics are imported (a no-op when engine.py already did)

import torch  # noqa: E402
from lada import ModelFiles  # noqa: E402
from lada.restorationpipeline import load_models  # noqa: E402
from lada.restorationpipeline.frame_restorer import FrameRestorer  # noqa: E402
from lada.utils import threading_utils, video_utils  # noqa: E402,F401  (re-exported through oh_player)
from lada.utils.os_utils import gpu_has_fp16_acceleration as _upstream_fp16_acceleration  # noqa: E402
from lada.utils.threading_utils import STOP_MARKER, ErrorMarker  # noqa: E402,F401  (re-exported through oh_player)

from oh_streamer.engines.lada.cancel import cancellable_restoration_model, install_cancel_hooks  # noqa: E402
from oh_streamer.engines.lada.io_hooks import register_source  # noqa: E402
from oh_streamer.processing import device_backend  # noqa: E402
from oh_streamer.processing.frame_rate import planned_fps  # noqa: E402

runtime_patches.install_nms_default()   # lada calls ultralytics NMS without max_time_img


upstream_gpu_has_fp16_acceleration = _upstream_fp16_acceleration   # Lada's own check, no device policy on top


def gpu_has_fp16_acceleration(device=None):
    return device_backend.fp16_default(device, _upstream_fp16_acceleration, torch_module=torch)


class RestorerFactory:
    """Owns the loaded models and builds FrameRestorers around them.

    Building a restorer is cheap (models reused): quality switches and clip-length changes just make a new one.
    Models are only reloaded when the detection/restoration model or FP16 changes (settings menu); release() drops
    every reference the factory holds first so the old and new weights never sit in VRAM together. The caller must
    drop its restorers too (Session.release_restorer).
    """

    def __init__(self, device, max_clip_length: int):
        self.device, self.max_clip_length = device, max_clip_length
        self.restoration_fps = "original"
        self.det_name = self.res_name = self.fp16 = None
        self.det_model = self.res_model = self.pad_mode = None

    def load(self, det_name: str, res_name: str, fp16: bool):
        det = ModelFiles.get_detection_model_by_name(det_name)
        res = ModelFiles.get_restoration_model_by_name(res_name)
        if not det or not res:
            raise RuntimeError(f"unknown detection/restoration model name: {det_name if not det else res_name}")
        self.det_model, self.res_model, self.pad_mode = load_models(self.device, res_name, res.path, None, det.path,
                                                                    fp16, False)
        self.det_name, self.res_name, self.fp16 = det_name, res_name, fp16
        install_cancel_hooks(self.res_model)

    def release(self):
        self.det_model = self.res_model = self.pad_mode = None
        gc.collect()
        device_backend.empty_cache(self.device)

    def make(self, source: "Source") -> FrameRestorer:
        # FrameRestorer reads the metadata in __init__ (queue sizes, MosaicDetector) via the patched lookup
        native = getattr(source, "native_meta", source.meta)
        fps = planned_fps(native.video_fps_exact, self.restoration_fps)
        stride = int(Fraction(native.video_fps_exact) / fps)
        source.meta = replace(native, video_fps_exact=fps, video_fps=float(fps),
                              average_fps=native.average_fps / stride,
                              frames_count=(native.frames_count + stride - 1) // stride)
        source.meta.restoration_stride = stride
        register_source(source)
        return FrameRestorer(self.device, source.src, self.max_clip_length, self.res_name, self.det_model,
                             cancellable_restoration_model(self.res_model), self.pad_mode)
