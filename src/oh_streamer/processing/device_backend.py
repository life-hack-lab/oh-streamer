"""Inference device policy; torch is imported only when a caller needs it."""

import re
import sys


def validate_device(value):
    if not isinstance(value, str) or not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", value):
        raise ValueError(f"unsupported device {value!r}; use auto, cpu, mps or cuda:N")
    return value


def select_device(requested="auto", *, torch_module=None):
    validate_device(requested)
    if torch_module is None:
        import torch as torch_module
    t = torch_module
    mps = getattr(getattr(t, "backends", None), "mps", None)
    has_mps = bool(mps and mps.is_available())
    has_cuda = t.cuda.is_available()
    if requested == "auto":
        requested = "cuda:0" if has_cuda else "mps" if has_mps else "cpu"
    if requested == "mps" and not has_mps:
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch/device")
    if requested.startswith("cuda"):
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if not has_cuda or index >= t.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} was requested but is unavailable")
    return t.device(requested)


def is_rocm(torch_module) -> bool:
    """Whether this already-imported PyTorch module was built with the ROCm/HIP backend."""
    return bool(getattr(getattr(torch_module, "version", None), "hip", None))


def backend_name(device, *, torch_module=None) -> str:
    """A display label for the selected backend; never imports torch just to label a device."""
    backend = getattr(device, "type", str(device).split(":", 1)[0])
    if backend == "cuda":
        # A caller that already has torch can distinguish HIP from NVIDIA CUDA. Without it, keep this helper
        # lightweight and use the conventional CUDA label.
        torch_module = torch_module if torch_module is not None else sys.modules.get("torch")
        return "ROCm/HIP" if torch_module is not None and is_rocm(torch_module) else "CUDA"
    return backend.upper()


def fp16_default(device, fallback, torch_module=None) -> bool:
    """Use FP32 on HIP until validated; otherwise preserve the caller's existing precision policy.

    If no torch module is supplied, inspect only an already-loaded module so this helper never imports torch itself.
    """
    backend = getattr(device, "type", str(device).split(":", 1)[0])
    if backend == "cuda":
        torch_module = torch_module if torch_module is not None else sys.modules.get("torch")
        if torch_module is not None and is_rocm(torch_module):
            return False
    return bool(fallback(device))


def rife_available(device) -> bool:
    # Mac RIFE is deliberately deferred until its eager MPS path is validated.
    return sys.platform != "darwin" and device.type == "cuda"


def empty_cache(device, *, torch_module=None):
    if device is None or device.type not in ("cuda", "mps"):
        return
    if torch_module is None:
        import torch as torch_module
    backend = getattr(torch_module, device.type, None)
    if device.type in ("cuda", "mps") and backend is not None:
        backend.empty_cache()


def allocated_gib(device, *, torch_module=None):
    if device is None or device.type not in ("cuda", "mps"):
        return None
    if torch_module is None:
        import torch as torch_module
    backend = getattr(torch_module, device.type, None)
    query = getattr(backend, "current_allocated_memory" if device.type == "mps" else "memory_allocated", None)
    if query is None:
        return None
    return (query(device) if device.type == "cuda" else query()) / 2**30
