"""Exact frame-rate planning for sampling before detection and restoration."""
from fractions import Fraction
import math


def planned_fps(fps, mode):
    fps = Fraction(str(fps)) if isinstance(fps, float) else Fraction(fps)
    if fps <= 0:
        raise ValueError("frame rate must be positive")
    if mode not in ("original", "30"):
        raise ValueError("unknown restoration frame rate")
    return fps / max(1, math.ceil(fps / 30)) if mode == "30" else fps


def selected_frame(index, stride):
    if stride < 1:
        raise ValueError("stride must be positive")
    return index % stride == 0
