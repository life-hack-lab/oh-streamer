"""Read-only environment diagnostic; heavyweight imports run in a bounded child."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:   # lets the script import oh_streamer without the removed root shims
    sys.path.insert(0, _SRC)

from oh_streamer.processing import encoding_backend
from portability_common import (SCHEMA_VERSION, child_environment, environment, last_json,
                                positive_seconds, project_root, run_command, write_json)

REQUIRED = ("torch", "torchvision", "numpy", "av", "ultralytics", "mmengine",
            "opencv-python", "psutil", "yt-dlp", "pywebview", "pystray", "Pillow")
OPTIONAL = ("pyatv", "playwright", "nvidia-ml-py")


def torch_worker(requested):
    import torch
    from oh_streamer.processing import device_backend
    selected = device_backend.select_device(requested)
    mps = getattr(torch.backends, "mps", None)
    lada = {}
    try:   # the Lada engine's version against the verified ones (informational: never fails the worker)
        from oh_streamer.engines.lada import compat, engine
        lada = {"lada_version": engine.LadaEngine().info().version}
        lada["lada_verified"] = compat.evaluate(lada["lada_version"]).ok
        lada["lada_verified_versions"] = list(compat.VERIFIED)
    except Exception as error:
        lada = {"lada_error": str(error)}
    return {**lada, "version": torch.__version__, "selected_device": str(selected),
            "cuda_available": torch.cuda.is_available(),
            "cuda_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else None,
            "mps_built": bool(mps and mps.is_built()), "mps_available": bool(mps and mps.is_available())}


def build_report(*, root=None, device="auto", probe_encoders=False, require_mps=False,
                 require_videotoolbox=False, timeout=30):
    root = project_root(root)
    checks = []

    def add(identifier, status, message, details=None, required=True):
        checks.append({"id": identifier, "status": status, "message": message,
                       "required": required, "details": details or {}})

    add("python", "pass" if sys.version_info >= (3, 12) else "fail", f"Python {environment()['python']}; requires 3.12+")
    for package in REQUIRED + OPTIONAL + (("pyobjc-core", "pyobjc-framework-Cocoa") if sys.platform == "darwin" else ()):
        required = package not in OPTIONAL
        try:
            version = importlib.metadata.version(package)
            add("package:" + package, "pass", version, required=required)
        except importlib.metadata.PackageNotFoundError:
            add("package:" + package, "fail" if required else "skip", "not installed", required=required)

    for name in ("ffmpeg", "ffprobe", "mpv"):
        path = shutil.which(name)
        if name == "mpv" and not path and sys.platform == "win32":
            candidate = Path(r"C:\Program Files\MPV Player\mpv.exe")
            path = str(candidate) if candidate.is_file() else None
        if not path:
            add("tool:" + name, "fail", "not found; install or add to PATH")
            continue
        result = run_command([path, "--version" if name == "mpv" else "-version"], timeout=timeout)
        text = result["stdout"] or result["stderr"]
        add("tool:" + name, "pass" if result["returncode"] == 0 and not result["timed_out"] else "fail",
            text.splitlines()[0][:240] if text else "version command failed",
            {"timed_out": result["timed_out"]})

    models = Path(child_environment(root)["LADA_MODEL_WEIGHTS_DIR"])
    weights = [{"name": p.relative_to(models).as_posix(), "size_bytes": p.stat().st_size}
               for p in sorted(models.rglob("*")) if p.is_file() and p.suffix in (".pt", ".pth")]
    detection = any("detection" in p["name"] and p["size_bytes"] > 0 for p in weights)
    restoration = any("restoration" in p["name"] and p["size_bytes"] > 0 for p in weights)
    add("models", "pass" if detection and restoration else "fail",
        "detection/restoration weight files found" if detection and restoration else "expected detection/restoration weights missing",
        {"files": weights, "note": "file inventory only; actual loading is checked by acceptance"})
    child = run_command([sys.executable, __file__, "--torch-worker", "--device", device],
                        timeout=timeout, env=child_environment(root))
    torch_info = {}
    try:
        torch_info = last_json(child["stdout"])
        ok = not child["timed_out"] and child["returncode"] == 0 and torch_info.get("status") == "pass"
    except ValueError:
        ok = False
    add("torch-device", "pass" if ok else "fail",
        f"selected {torch_info.get('selected_device')}" if ok else "PyTorch import/device selection failed",
        {**torch_info, "timed_out": child["timed_out"], "error": child["stderr"][-1500:] if not ok else ""})
    if torch_info.get("lada_version"):
        verified = bool(torch_info.get("lada_verified"))
        verified_text = ", ".join(torch_info.get("lada_verified_versions") or [])
        add("engine:lada", "pass" if verified else "skip",
            f"lada {torch_info['lada_version']} (verified)" if verified else
            f"lada {torch_info['lada_version']} not verified; verified: {verified_text}",
            {k: torch_info[k] for k in ("lada_version", "lada_verified", "lada_verified_versions") if k in torch_info},
            required=False)
    if require_mps:
        add("required:mps", "pass" if torch_info.get("mps_available") else "fail", "MPS availability required")
    hardware = ("h264_videotoolbox", "hevc_videotoolbox") if sys.platform == "darwin" else ("h264_nvenc", "hevc_nvenc")
    if require_videotoolbox:
        hardware = tuple(dict.fromkeys(hardware + ("h264_videotoolbox", "hevc_videotoolbox")))
    for encoder in hardware:
        required = require_videotoolbox and encoder.endswith("_videotoolbox")
        if not probe_encoders and not required:
            add("encoder:" + encoder, "skip", "use --probe-encoders for an actual encode", required=False)
        else:
            available = encoding_backend.encoder_usable(encoder, timeout=timeout)
            add("encoder:" + encoder, "pass" if available else "fail" if required else "skip",
                "actual short encode passed" if available else "hardware encode unavailable; software fallback may be used", required=required)
    return {"schema_version": SCHEMA_VERSION, "kind": "portability-doctor", "environment": environment(),
            "status": "fail" if any(c["status"] == "fail" and c["required"] for c in checks) else "pass",
            "checks": checks, "mac_validated": False,
            "note": "This report does not validate MPS inference, Cocoa GUI, or VideoToolbox segment correctness."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--probe-encoders", action="store_true")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--require-videotoolbox", action="store_true")
    parser.add_argument("--timeout", type=positive_seconds, default=30)
    parser.add_argument("--torch-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.torch_worker:
        try:
            report = {"status": "pass", **torch_worker(args.device)}
        except Exception as error:
            report = {"status": "fail", "error": str(error)}
        print(json.dumps(report, ensure_ascii=False))
    else:
        report = build_report(root=args.project_root, device=args.device, probe_encoders=args.probe_encoders,
                              require_mps=args.require_mps, require_videotoolbox=args.require_videotoolbox, timeout=args.timeout)
        if args.output:
            write_json(args.output, report)
        for check in report["checks"]:
            print(f"{check['status'].upper():4} {check['id']}: {check['message']}")
        print(f"Overall: {report['status']}; Mac runtime validation remains pending")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
