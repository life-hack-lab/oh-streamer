"""Bounded local-video benchmark using the streamer's actual restoration pipeline.

Each detection-model / clip-length pair gets a fresh process. No player settings
are read or written. Default output discards frames; --encoder enables a real
video encode to FFmpeg's null muxer, or video-only MKV files with --save.
Timings are wall-clock pipeline timings, never a claim of pure model inference.
"""

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

STREAMER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAMER_ROOT))
sys.path.insert(0, str(STREAMER_ROOT / "src"))

import oh_streamer.processing.device_backend as device_backend
import oh_streamer.processing.encoding_backend as encoding_backend

SCHEMA_VERSION = 1
RESULT_PREFIX = "BENCH_RESULT "


def finite_number(value, *, positive=True):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a finite number") from None
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise argparse.ArgumentTypeError("must be finite and " + ("positive" if positive else "nonnegative"))
    return number


def clip_lengths(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("clip lengths must be comma-separated positive integers") from None
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("clip lengths must be positive integers")
    return list(dict.fromkeys(values))


def model_names(value):
    values = [part.strip() for part in value.split(",")]
    if not values or any(not v for v in values):
        raise argparse.ArgumentTypeError("models must be nonempty comma-separated names")
    return list(dict.fromkeys(values))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="local video path; no URL resolution or saved player settings")
    ap.add_argument("--device", type=device_backend.validate_device, default="auto")
    ap.add_argument("--encoder", choices=("null", "auto") + encoding_backend.ENCODERS, default="null",
                    help="null discards frames; other values encode video (default: null)")
    ap.add_argument("--duration", type=finite_number, default=5.0, help="positive seconds of source video")
    ap.add_argument("--start", type=lambda s: finite_number(s, positive=False), default=0.0)
    ap.add_argument("--clip-lengths", type=clip_lengths, default=[30])
    ap.add_argument("--models", type=model_names, default=["v4-fast"], help="detection model names")
    ap.add_argument("--restoration-model", default="basicvsrpp-v1.2")
    ap.add_argument("--fp16", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--timeout", type=finite_number, default=180.0,
                    help="wall-clock limit per child, including imports, probing, load and cleanup")
    ap.add_argument("--output", required=True, help="JSON report path; logs are retained beside it")
    ap.add_argument("--save", help="directory for video-only restored MKV segments; requires encoding")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config-index", type=int, default=0, help=argparse.SUPPRESS)
    return ap


def select_fp16(requested, device, acceleration):
    return bool(acceleration(device)) if requested == "auto" else requested == "on"


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class MemoryMetrics:
    """CUDA allocator peak or MPS current allocation samples; CPU is unavailable."""

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.maximum = None
        self.samples = 0
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        try:
            value = int(self.torch.mps.current_allocated_memory())
            self.maximum = value if self.maximum is None else max(self.maximum, value)
            self.samples += 1
        except Exception as exc:
            self.error = str(exc)

    def start(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "mps":
            self.sample()

            def poll():
                while not self.stop_event.wait(0.05):
                    self.sample()
            self.thread = threading.Thread(target=poll, daemon=True)
            self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
            self.sample()
        if self.device.type == "cuda":
            return {"status": "available", "method": "torch.cuda.max_memory_allocated",
                    "scope": "model load and pipeline, selected device, PyTorch allocated tensors only",
                    "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device))}
        if self.device.type == "mps":
            return {"status": "available" if self.samples else "unavailable",
                    "method": "torch.mps.current_allocated_memory sampled every 50 ms",
                    "scope": "model load and pipeline; sampled maximum, not a true allocator peak or system RAM",
                    "sampled_max_current_allocated_bytes": self.maximum, "samples": self.samples,
                    "sampling_error": self.error}
        return {"status": "unavailable", "method": None, "scope": "CPU memory is not measured"}


class DetectionCounts:
    """Match consumed detection queue entries to delivered frames despite lookahead."""

    def __init__(self, restorer):
        self.pending = {}
        self.lock = threading.Lock()
        self.total = self.frames_with_mosaic = self.observed_frames = 0
        original_get = restorer.frame_detection_queue.get

        def get(*args, **kwargs):
            elem = original_get(*args, **kwargs)
            if isinstance(elem, tuple) and len(elem) == 2:
                with self.lock:
                    self.pending[int(elem[0])] = int(elem[1])
            return elem
        restorer.frame_detection_queue.get = get

    def delivered(self, frame_num):
        with self.lock:
            count = self.pending.pop(frame_num, None)
        if count is None:
            raise RuntimeError(f"missing detection count for output frame {frame_num}")
        self.observed_frames += 1
        self.total += count
        self.frames_with_mosaic += int(count > 0)

    def report(self):
        return {"scope": "delivered output frames only; mosaics are detected regions per frame",
                "observed_frames": self.observed_frames, "frames_with_mosaic": self.frames_with_mosaic,
                "mosaics": self.total, "restoration_workload_present": self.total > 0,
                "note": "zero detections exercises decoding/detection, but does not measure dense restoration"}


def timing_metrics(frames, start, first, end, fps):
    if frames < 1 or first is None:
        raise RuntimeError("no frames were produced in the requested interval")
    elapsed = max(1e-9, end - start)
    steady = (frames - 1) / max(1e-9, end - first) if frames > 1 else None
    return {"first_frame_s": first - start, "processing_s": elapsed,
            "processing_fps": frames / elapsed, "steady_output_fps": steady,
            "steady_output_scope": "delivery after first output; may drain prefetched bursts, especially in short samples",
            "x_realtime": frames / elapsed / fps,
            "method": "perf_counter wall time with selected-device synchronization at load, first output and end",
            "scope": "start of FrameRestorer.start through final output; decoding, detection, restoration, "
                     "queueing and optional encoder writes; excludes imports/probe/model load/cleanup/encoder drain"}


def start_encoder(meta, encoder, output):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{meta.video_width}x{meta.video_height}",
           "-framerate", str(meta.video_fps_exact), "-i", "pipe:0", "-an"]
    cmd += encoding_backend.encoder_args(encoder, purpose="segmented") + encoding_backend.quality_args(encoder)
    cmd += [str(output)] if output else ["-f", "null", "-"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)


def run_configuration(args, result, *, ls=None):
    if ls is None:
        import lada_ns  # noqa: E402  (benchmarks/lada_ns.py: oh_player no longer re-exports Lada's names)
        ls = lada_ns.load()
    device = device_backend.select_device(args.device, torch_module=ls.torch)
    fp16 = select_fp16(args.fp16, device, ls.gpu_has_fp16_acceleration)
    result["selected"] = {"device": str(device), "fp16": fp16, "encoder": None,
                          "detection_model": None, "restoration_model": None}
    for kind, name, lookup in (("detection", args.models[0], ls.ModelFiles.get_detection_model_by_name),
                               ("restoration", args.restoration_model, ls.ModelFiles.get_restoration_model_by_name)):
        model = lookup(name)
        if model is None or not Path(model.path).is_file():
            raise RuntimeError(f"missing or unknown {kind} model: {name}")
    source = ls.Source(str(Path(args.input).resolve()), {})
    meta = source.meta
    fps = float(meta.video_fps_exact)
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("source frame rate must be finite and positive")
    if meta.duration > 0 and args.start >= meta.duration:
        raise RuntimeError("no frames: requested start is at or past source duration")
    target_frames = max(1, math.ceil(args.duration * fps))
    result["source"] = {"width": meta.video_width, "height": meta.video_height, "fps": fps,
                        "start_s": args.start, "requested_duration_s": args.duration,
                        "target_frames": target_frames}
    selected_encoder = None
    if args.encoder != "null":
        selected_encoder = encoding_backend.resolve_encoder(args.encoder)
        result["selected"]["encoder"] = selected_encoder
    memory = MemoryMetrics(ls.torch, device)
    restorer = muxer = None
    memory.start()
    try:
        synchronize(ls.torch, device)
        load_start = time.perf_counter()
        factory = ls.RestorerFactory(device, args.clip_lengths[0])
        factory.load(args.models[0], args.restoration_model, fp16)
        synchronize(ls.torch, device)
        result["model_load_s"] = time.perf_counter() - load_start
        result["model_load_scope"] = "RestorerFactory.load including model initialization and built-in warmup; excludes imports/probe"
        result["selected"].update(detection_model=factory.det_name, restoration_model=factory.res_name)
        result["selected"].update(
            detection_backend=type(factory.det_model).__name__,
            restoration_backend=type(factory.res_model).__name__,
            detection_dtype=str(getattr(factory.det_model, "dtype", "unknown")),
            restoration_dtype=str(getattr(factory.res_model, "dtype", "unknown")))
        restorer = factory.make(source)
        counts = DetectionCounts(restorer)
        output = None
        if args.save:
            output = Path(args.save) / f"config_{args.config_index:03d}.mkv"
            output.parent.mkdir(parents=True, exist_ok=True)
            result["saved_video"] = str(output.resolve())
        if selected_encoder:
            muxer = start_encoder(meta, selected_encoder, output)
        synchronize(ls.torch, device)
        t0 = time.perf_counter()
        first = None
        frames = 0
        restorer.start(start_ns=int(args.start * 1e9))
        for elem in restorer:
            if elem is ls.STOP_MARKER or isinstance(elem, ls.ErrorMarker):
                raise RuntimeError(f"restorer stopped: {getattr(elem, 'error', elem)}")
            frame, _pts = elem
            if first is None:
                synchronize(ls.torch, device)
                first = time.perf_counter()
            counts.delivered(restorer.start_frame + frames)
            if muxer:
                array = frame.detach().cpu().numpy() if isinstance(frame, ls.torch.Tensor) else frame
                muxer.stdin.write(memoryview(array.reshape(-1)))
            frames += 1
            if frames >= target_frames:
                break
        synchronize(ls.torch, device)
        end = time.perf_counter()
        result.update(frames=frames, reached_requested_frames=frames >= target_frames,
                      timings=timing_metrics(frames, t0, first, end, fps), detections=counts.report())
        if muxer:
            drain_start = time.perf_counter()
            muxer.stdin.close()
            code = muxer.wait(timeout=min(args.timeout, 15.0))
            result["encoder_drain_s"] = time.perf_counter() - drain_start
            if code:
                raise RuntimeError(f"encoder exited with status {code}")
    finally:
        # Capture metrics before teardown allocations and cancellation.
        result["memory"] = memory.finish()
        if muxer and muxer.poll() is None:
            muxer.kill()
            muxer.wait(timeout=5)
        if restorer:
            cancel = getattr(restorer.mosaic_restoration_model, "cancel_event", None)
            if cancel:
                cancel.set()
            restorer.stop()


def requested_config(args):
    return {"device": args.device, "encoder": args.encoder, "fp16": args.fp16,
            "detection_model": args.models[0], "restoration_model": args.restoration_model,
            "clip_length": args.clip_lengths[0]}


def terminate_process_tree(process):
    """Stop child and its FFmpeg descendants on a timeout."""
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        return
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def child_command(args, model, clip, index):
    cmd = [sys.executable, str(Path(__file__).resolve()), str(Path(args.input).resolve()), "--one",
           "--models", model, "--clip-lengths", str(clip), "--restoration-model", args.restoration_model,
           "--device", args.device, "--encoder", args.encoder, "--fp16", args.fp16,
           "--start", str(args.start), "--duration", str(args.duration), "--timeout", str(args.timeout),
           "--output", str(Path(args.output).resolve()), "--config-index", str(index)]
    if args.save:
        cmd += ["--save", str(Path(args.save).resolve())]
    return cmd


def run_child(args, model, clip, index, log_dir):
    requested = {**requested_config(args), "detection_model": model, "clip_length": clip}
    result = {"status": "fail", "requested": requested}
    log_path = log_dir / f"config_{index:03d}.log"
    result["log"] = str(log_path.resolve())
    started = time.perf_counter()
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP) \
        if sys.platform == "win32" else 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(child_command(args, model, clip, index), stdout=log, stderr=log,
                                   creationflags=flags, start_new_session=sys.platform != "win32")
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process)
            result["error"] = f"configuration timed out after {args.timeout:g} seconds"
            code = None
        except BaseException:
            terminate_process_tree(process)
            raise
    if code is not None:
        payload = None
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if line.startswith(RESULT_PREFIX):
                    try:
                        payload = json.loads(line[len(RESULT_PREFIX):])
                    except ValueError:
                        pass
        if isinstance(payload, dict) and payload.get("status") in ("pass", "fail"):
            result.update(payload)
        else:
            result["error"] = "child did not emit a valid benchmark result"
        if code != 0:
            result["status"] = "fail"
            result.setdefault("error", f"child exited with status {code}")
        result["exit_code"] = code
    result["child_wall_s"] = time.perf_counter() - started
    result["child_wall_scope"] = "includes interpreter startup, imports, probe, model load, pipeline and cleanup"
    return result


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.save and args.encoder == "null":
        ap.error("--save requires --encoder auto or a concrete encoder")
    if args.one and (len(args.models) != 1 or len(args.clip_lengths) != 1):
        ap.error("child requires one model and one clip length")
    if args.one:
        result = {"status": "fail", "requested": requested_config(args)}
        try:
            if not Path(args.input).is_file():
                raise RuntimeError("input must be an existing local video file")
            run_configuration(args, result)
            result["status"] = "pass"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)
        print(RESULT_PREFIX + json.dumps(result, allow_nan=False), flush=True)
        return 0 if result["status"] == "pass" else 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output.parent / (output.stem + ".logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (model, clip) in enumerate(itertools.product(args.models, args.clip_lengths)):
        print(f"Benchmark {model}, clip={clip}, device={args.device}", file=sys.stderr, flush=True)
        try:
            results.append(run_child(args, model, clip, index, log_dir))
        except Exception as exc:
            results.append({"status": "fail", "requested": {**requested_config(args),
                           "detection_model": model, "clip_length": clip}, "error": f"{type(exc).__name__}: {exc}"})
    report = {"schema_version": SCHEMA_VERSION, "status": "pass" if all(r["status"] == "pass" for r in results) else "fail",
              "results": results, "limitations": ["Pipeline throughput is not pure neural inference speed.",
              "Synthetic or zero-detection footage does not validate dense mosaic restoration performance.",
              "MPS reports sampled current allocation; CPU memory is unavailable; CUDA peak is PyTorch allocation only.",
              "Saved segments contain restored video only, without source audio."]}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
