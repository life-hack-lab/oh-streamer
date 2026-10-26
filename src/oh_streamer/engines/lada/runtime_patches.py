# SPDX-License-Identifier: AGPL-3.0-only
# Modifies the run-time behavior of upstream lada / ultralytics / torch weight loading (AGPL-3.0 software). Modified 2026-10-26; see THIRD_PARTY_NOTICES.md, section
# "Modifications to upstream software".
"""Runtime handling for the three behaviors lada used to get from edits to its installed files.

uv hardlinks installed packages from its cache, so editing site-packages corrupts the cache; the same effects are applied
here instead, from our own code. Keep this module import-light: apply_env() must run before torch / ultralytics load
weights or import, so it touches nothing but os.environ; install_nms_default() imports ultralytics itself.

  1. mmengine weight loading: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch >= 2.6 defaults to weights_only=True).
  2. ultralytics telemetry + DNS lookups at import: YOLO_OFFLINE=true.
  3. NMS time budget: ultralytics' non_max_suppression default max_time_img 0.05 -> 0.3 s (lada calls it through
     ultralytics.utils.nms without passing max_time_img).
"""

import functools
import inspect
import os

NMS_MAX_TIME_IMG = 0.3
_MARK = "_oh_streamer_nms_default"


def apply_env() -> None:
    """Set the environment lada / torch / ultralytics read at load time (never overrides a value already set)."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")


def install_nms_default(module=None, default: float = NMS_MAX_TIME_IMG) -> bool:
    """Make module.non_max_suppression default max_time_img to `default`; True when a wrapper was installed.

    `module` is ultralytics.utils.nms unless given. An explicit max_time_img (keyword or positional) is passed through.
    Idempotent (an installed wrapper is recognized), and a no-op when the function already defaults to `default`
    (dev site-packages that still carry the old edit) or has no max_time_img parameter (other ultralytics versions).
    """
    if module is None:
        from ultralytics.utils import nms as module  # noqa: PLC0415 - heavy import, after apply_env()
    func = module.non_max_suppression
    if getattr(func, _MARK, False):
        return False
    sig = inspect.signature(func)
    param = sig.parameters.get("max_time_img")
    if param is None or param.default == default:
        return False
    names = list(sig.parameters)
    position = names.index("max_time_img")

    @functools.wraps(func)
    def non_max_suppression(*args, **kwargs):
        if "max_time_img" not in kwargs and len(args) <= position:
            kwargs["max_time_img"] = default
        return func(*args, **kwargs)

    # inspect.signature follows __wrapped__ and would report the old default; show the effective one
    non_max_suppression.__signature__ = sig.replace(
        parameters=[p.replace(default=default) if p.name == "max_time_img" else p for p in sig.parameters.values()])
    setattr(non_max_suppression, _MARK, True)
    module.non_max_suppression = non_max_suppression
    return True
