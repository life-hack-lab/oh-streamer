r"""Torch flavors (the uv extras) in one place. Stdlib only.

Adding a flavor (stage 4: intel, amd) starts here: its extra must exist in the shipped uv.lock before in_lock is True,
and offered is flipped only after it was tested on real hardware. The setup UI's status texts (setup.js STATUS_TEXT)
and gpu_detect.recommend() still have to learn about a new vendor. selfcheck.SNIPPET checks CUDA only; a flavor whose
device is not CUDA (intel: xpu) needs its own self-check before it can be offered.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flavor:
    name: str                                   # uv extra name; also stored in current.json
    label: str                                  # option text in the setup window
    in_lock: bool                               # the shipped uv.lock has this extra, so the installer can sync it
    listed: bool                                # shown in the setup window's version list
    offered: bool                               # selectable and installable now
    cards: str = ""                             # supported cards in plain words, listed when nothing can be installed


ALL = (
    Flavor("nvidia", "NVIDIA（CUDA 12.8）", in_lock=True, listed=True, offered=True,
           cards="NVIDIA GTX 16 系列、RTX 20～50 系列（驅動 570 以上）"),
    Flavor("intel", "Intel（Arc／Core Ultra）", in_lock=False, listed=True, offered=False),     # stage 4: own pyproject / lock
    Flavor("amd", "AMD（ROCm，實驗性）", in_lock=False, listed=True, offered=False),            # stage 4: own pyproject / lock
    Flavor("cpu", "CPU", in_lock=True, listed=False, offered=False),                           # development tests only
)


def in_lock() -> tuple[str, ...]:
    return tuple(f.name for f in ALL if f.in_lock)


def offered() -> tuple[str, ...]:
    return tuple(f.name for f in ALL if f.offered)


def supported_cards() -> tuple[str, ...]:
    return tuple(f.cards for f in ALL if f.offered and f.cards)


def listed() -> tuple[Flavor, ...]:
    return tuple(f for f in ALL if f.listed)
