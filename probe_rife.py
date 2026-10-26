"""Bounded standalone eager RIFE probe; does not enable interpolation in the Mac player/queue.

Examples: python probe_rife.py --device cpu --size 33x25 --json
          python probe_rife.py --device mps --fp16 auto --output rife-mps.json
An explicit --device mps is experimental opt-in, not evidence of full Mac validation.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:   # lets the script import oh_streamer without the removed root shims
    sys.path.insert(0, _SRC)

from oh_streamer.processing.rife_interp import MODELS, MODEL


def _size(value):
    try:
        w, h = (int(part) for part in value.lower().split("x"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from None
    if min(w, h) < 16 or max(w, h) > 2048 or w * h > 1920 * 1080:
        raise argparse.ArgumentTypeError("size must be at least 16x16, at most 2048 per side and 1920x1080 pixels")
    return w, h


def _iterations(value):
    n = int(value)
    if not 1 <= n <= 20:
        raise argparse.ArgumentTypeError("iterations must be between 1 and 20")
    return n


def _timeout(value):
    n = float(value)
    if not math.isfinite(n) or not 1 <= n <= 300:
        raise argparse.ArgumentTypeError("timeout must be between 1 and 300 seconds")
    return n


def _report(args):
    return {
        "schema_version": 1,
        "status": "fail",
        "requested_device": args.device,
        "model": args.model,
        "size": list(args.size),
        "requested_fp16": args.fp16,
        "iterations": args.iterations,
        "mode": "eager",
        "mac_validation": "unvalidated",
        "mac_player_rife_enabled": False,
        "checks": {},
    }


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_probe(args):
    """Run inside the bounded child process; failures produce a JSON fail report."""
    report = _report(args)
    try:
        import torch
        from oh_streamer.processing.device_backend import select_device
        from oh_streamer.processing.rife_interp import Rife, model_scale

        device = select_device(args.device)
        report["device"] = str(device)
        report["torch_version"] = str(torch.__version__)
        if device.type == "mps" and args.device != "mps":
            raise ValueError("RIFE MPS remains unvalidated; explicitly select --device mps for the experimental probe")
        if device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        fp16 = None if args.fp16 == "auto" else args.fp16 == "on"
        w, h = args.size
        started = time.perf_counter()
        rife = Rife(args.model, w, h, model_scale(h), device, fp16=fp16,
                    experimental_mps=args.device == "mps")
        _sync(torch, device)
        report["load_s"] = round(time.perf_counter() - started, 6)
        report["precision"] = "fp16" if rife.dtype == torch.half else "fp32"
        report["cuda_graph"] = rife.graph is not None
        report["own_stream"] = rife.stream is not None
        report["padded_size"] = [rife.pw, rife.ph]
        report["experimental_mps"] = device.type == "mps"
        # Related images make the ordinary pair distinct from the black/white scene-cut pair.
        frame0 = ((torch.arange(h * w * 3).reshape(h, w, 3) % 128) + 32).to(torch.uint8)
        frame1 = (frame0.to(torch.int16) + 4).clamp(0, 255).to(torch.uint8)
        with torch.inference_mode():
            started = time.perf_counter()
            first = rife.step(rife.prep(frame0))
            raw = rife.step(rife.prep(frame1))
            _sync(torch, device)
            report["first_inference_s"] = round(time.perf_counter() - started, 6)
            report["checks"]["first_step_empty"] = first is None
            report["checks"]["raw_shape"] = list(raw.shape) == [1, 3, rife.ph, rife.pw]
            report["checks"]["raw_finite"] = bool(torch.isfinite(raw).all().item())
            report["checks"]["raw_precision"] = raw.dtype == rife.dtype
            samples = []
            for _ in range(args.iterations):
                _sync(torch, device)
                started = time.perf_counter()
                out = rife.middle(frame0, frame1)
                _sync(torch, device)
                samples.append((time.perf_counter() - started) * 1000)
                if out is None or list(out.shape) != [h, w, 3] or str(out.dtype) != "uint8":
                    raise RuntimeError("ordinary pair did not yield a uint8 HWC frame of the requested size")
            report["checks"]["output_shape_dtype"] = True
            black, white = torch.zeros_like(frame0), torch.full_like(frame0, 255)
            report["checks"]["scene_cut_repeats"] = rife.middle(black, white) is None
        report["inference_ms"] = {
            "samples": [round(ms, 4) for ms in samples],
            "median": round(statistics.median(samples), 4),
            "measurement": "wall time for upload/preparation, SSIM, eager model, postprocess/download; backend synchronized",
        }
        if not all(report["checks"].values()):
            raise RuntimeError("one or more RIFE correctness checks failed")
        report["status"] = "pass"
    except Exception as exc:  # noqa: BLE001 - missing dependencies/operators are probe results
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:N or explicit experimental mps")
    parser.add_argument("--model", choices=MODELS, default=MODEL)
    parser.add_argument("--size", type=_size, default=(64, 48))
    parser.add_argument("--fp16", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--iterations", type=_iterations, default=2)
    parser.add_argument("--timeout", type=_timeout, default=60.0, help="total worker timeout in seconds, including load")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a text summary")
    parser.add_argument("--output", type=Path, help="save the JSON report")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        report = run_probe(args)
    else:
        w, h = args.size
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--device", args.device,
                   "--model", args.model, "--size", f"{w}x{h}", "--fp16", args.fp16,
                   "--iterations", str(args.iterations), "--json"]
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            report = json.loads(child.stdout)
            if child.returncode not in (0, 1) or report.get("status") not in ("pass", "fail"):
                raise RuntimeError(f"RIFE worker exited unexpectedly ({child.returncode})")
        except subprocess.TimeoutExpired:
            report = _report(args)
            report["error"] = f"RIFE worker exceeded {args.timeout:g}s timeout"
        except Exception as exc:  # noqa: BLE001
            report = _report(args)
            report["error"] = f"RIFE worker failed: {type(exc).__name__}: {exc}"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(f"RIFE {report['status']}: {report.get('device', args.device)} / {args.model} / eager / "
              f"{report.get('precision', args.fp16)}")
        if "error" in report:
            print(report["error"])
        elif "inference_ms" in report:
            print(f"Median pair: {report['inference_ms']['median']:.4f} ms; all correctness checks passed")
        print("Mac validation: unvalidated; Mac player/queue RIFE remains gated")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
