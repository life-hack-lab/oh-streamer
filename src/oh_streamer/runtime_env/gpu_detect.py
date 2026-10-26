r"""Detect the machine's GPUs (WMI + nvidia-smi) and pick which torch flavor to install. Judgement only: nothing is installed.

run is an injectable function `run(args: list[str]) -> str | None` returning stdout, or None when the command is missing,
fails or times out; tests pass a fake. Stdlib only: first-run setup happens before torch exists.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable

from . import flavors

Runner = Callable[[list], "str | None"]

MIN_NVIDIA_DRIVER = 570                                                 # CUDA 12.8 wheels (Blackwell) need driver >= 570
MIN_MODERN_CC = (7, 5)                                                  # GTX 16xx / RTX 20 (Turing) and newer
MIN_LEGACY_CC = (5, 0)                                                  # Maxwell..Volta: nvidia-legacy flavor (deferred)

_WMI_COMMAND = ("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress")
_SMI_COMMAND = ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"]

_NVIDIA_MODERN = re.compile(r"\bRTX\b|\bGTX\s*16\d{2}(?!\d)", re.I)     # every RTX-branded card is Turing or newer
_NVIDIA_LEGACY = re.compile(r"\bGTX\s*(?:9\d{2}|10\d{2})(?!\d)", re.I)  # Maxwell / Pascal


@dataclass
class GpuInfo:
    name: str
    vendor: str                                                         # "nvidia" | "intel" | "amd" | "other"
    driver_version: str | None = None                                   # NVIDIA: marketing form (e.g. "560.94"), else as WMI reports it
    compute_capability: tuple[int, int] | None = None                   # NVIDIA only, from nvidia-smi


@dataclass
class Detection:
    gpus: list[GpuInfo] = field(default_factory=list)
    nvidia_smi_ok: bool = False                                         # nvidia-smi answered with at least one GPU


@dataclass
class Recommendation:
    flavor: str | None                                                  # "nvidia" | "intel" | "amd" | None (= CPU only / nothing to install)
    status: str                                                         # the UI maps this to a message
    warnings: list[str] = field(default_factory=list)                   # codes, not message text


def _is_windows() -> bool:
    return sys.platform == "win32"


def _default_run(args: list, timeout: float = 15.0) -> str | None:
    kwargs = {}
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _call(run: Runner | None, args: list) -> str | None:
    try:
        out = (run or _default_run)(args)
    except Exception:
        return None
    return out if isinstance(out, str) else None


def vendor_of(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or re.search(r"\b(?:rtx|gtx)\b", low):
        return "nvidia"
    if "intel" in low:
        return "intel"
    if "amd" in low or "radeon" in low or re.search(r"\bati\b", low):
        return "amd"
    return "other"


def _wmi_nvidia_driver(version: str | None) -> str | None:
    """Windows driver string "31.0.15.5123" -> NVIDIA's "551.23" (last five digits of the 3rd+4th parts)."""
    parts = (version or "").strip().split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return None
    digits = (parts[2] + parts[3])[-5:]
    return f"{int(digits[:3])}.{digits[3:]}" if len(digits) == 5 else None


def query_wmi_gpus(run: Runner | None = None) -> list[GpuInfo]:
    """Every display adapter Windows knows about. Empty off Windows, or when PowerShell fails / prints junk."""
    if not _is_windows():
        return []
    text = _call(run, ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                       "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + _WMI_COMMAND])
    if not text:
        return []
    try:
        data = json.loads(text.lstrip("﻿").strip())
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    gpus = []
    for item in data:
        name = item.get("Name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        vendor = vendor_of(name)
        driver = item.get("DriverVersion")
        driver = driver.strip() if isinstance(driver, str) and driver.strip() else None
        if vendor == "nvidia":
            driver = _wmi_nvidia_driver(driver)
        gpus.append(GpuInfo(name, vendor, driver))
    return gpus


def _clean(field_text: str) -> str | None:
    value = field_text.strip()
    return None if not value or value.upper().strip("[]") in ("N/A", "NOT SUPPORTED") else value


def _parse_cc(value: str | None) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", value or "")
    return (int(match.group(1)), int(match.group(2) or 0)) if match else None


def query_nvidia_smi(run: Runner | None = None) -> list[GpuInfo]:
    """NVIDIA GPUs as nvidia-smi sees them. Empty when it is missing, fails, or reports no device. Never raises."""
    text = _call(run, _SMI_COMMAND)
    if not text:
        return []
    gpus = []
    for row in csv.reader(text.splitlines()):
        if len(row) < 2:                                                # blank line or an error message, not a data row
            continue
        name = _clean(row[0])
        if name is None:
            continue
        cc = _parse_cc(_clean(row[2])) if len(row) > 2 else None
        gpus.append(GpuInfo(name, "nvidia", _clean(row[1]), cc))
    return gpus


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _merge_nvidia(wmi: list[GpuInfo], smi: list[GpuInfo]) -> list[GpuInfo]:
    """nvidia-smi's driver and compute capability win; a WMI-only or smi-only card is still kept."""
    pool = list(smi)
    pairs: list[tuple[GpuInfo, GpuInfo | None]] = []
    for w in wmi:
        match = next((s for s in pool if _norm(s.name) == _norm(w.name)), None)
        if match is not None:
            pool.remove(match)
        pairs.append((w, match))
    unmatched = [i for i, (_, s) in enumerate(pairs) if s is None]
    if len(unmatched) == 1 and len(pool) == 1:                          # names formatted differently, but only one candidate each
        pairs[unmatched[0]] = (pairs[unmatched[0]][0], pool.pop())
    merged = [GpuInfo(w.name, "nvidia", (s.driver_version if s and s.driver_version else w.driver_version),
                      s.compute_capability if s else None) for w, s in pairs]
    return merged + pool


def detect(run: Runner | None = None) -> Detection:
    wmi = query_wmi_gpus(run)
    smi = query_nvidia_smi(run)
    others = [g for g in wmi if g.vendor != "nvidia"]
    nvidia = _merge_nvidia([g for g in wmi if g.vendor == "nvidia"], smi)
    return Detection(nvidia + others, nvidia_smi_ok=bool(smi))


def _nvidia_class(gpu: GpuInfo) -> str:
    """"modern" | "legacy" | "unknown" | "unsupported" (older than anything we could ship a wheel for)."""
    cc = gpu.compute_capability
    if cc is not None:
        if cc >= MIN_MODERN_CC:
            return "modern"
        return "legacy" if cc >= MIN_LEGACY_CC else "unsupported"
    if _NVIDIA_MODERN.search(gpu.name):
        return "modern"
    if _NVIDIA_LEGACY.search(gpu.name):
        return "legacy"
    return "unknown"


def _driver_major(version: str | None) -> int | None:
    match = re.match(r"\s*(\d+)", version or "")
    return int(match.group(1)) if match else None


def _judge_nvidia(nvidia: list[GpuInfo]) -> tuple[str | None, str, list[str]]:
    classes = {id(g): _nvidia_class(g) for g in nvidia}
    for kind, status in (("modern", None), ("unknown", "unknown_nvidia"), ("legacy", "legacy_nvidia_deferred"),
                         ("unsupported", "cpu_only")):
        group = [g for g in nvidia if classes[id(g)] == kind]
        if not group:
            continue
        if status is not None:
            return None, status, []
        major = next((m for m in (_driver_major(g.driver_version) for g in group) if m is not None), None)
        if major is not None and major < MIN_NVIDIA_DRIVER:
            return "nvidia", "driver_too_old", [f"driver_below_{MIN_NVIDIA_DRIVER}"]
        return "nvidia", "ok", []
    return None, "cpu_only", []


def recommend(gpus: list[GpuInfo], supported: tuple[str, ...] | None = None) -> Recommendation:
    """NVIDIA wins over everything else (a dGPU beats an iGPU). A flavor outside `supported` (default: flavors.offered()) is not offered."""
    if supported is None:
        supported = flavors.offered()
    nvidia = [g for g in gpus if g.vendor == "nvidia"]
    if nvidia:
        flavor, status, warnings = _judge_nvidia(nvidia)
        if flavor is None:                                              # legacy / unknown / unsupported keep their own status
            return Recommendation(None, status, warnings)
    elif any(g.vendor == "intel" for g in gpus):
        flavor, status, warnings = "intel", "ok", []
    elif any(g.vendor == "amd" for g in gpus):
        flavor, status, warnings = "amd", "ok", []
    else:
        return Recommendation(None, "cpu_only")
    if flavor not in supported:
        return Recommendation(None, {"intel": "intel_not_offered", "amd": "amd_not_offered"}.get(flavor, "cpu_only"), warnings)
    return Recommendation(flavor, status, warnings)


if __name__ == "__main__":
    found = detect()
    print(json.dumps({"detection": found, "recommendation": recommend(found.gpus)}, default=lambda o: o.__dict__, indent=2))
