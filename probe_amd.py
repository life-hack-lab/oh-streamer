"""Bounded AMD development probe for ROCm/HIP, torchvision, AMF, and optional models.

This reports what the current Python/FFmpeg environment actually executes. A
listed encoder or device name alone is never counted as a successful probe.
It makes no package, driver, or system changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:   # lets the script import oh_streamer without the removed root shims
    sys.path.insert(0, _SRC)

from portability_common import SCHEMA_VERSION, child_environment, environment, last_json, project_root, run_command, write_json


STATUSES = ("pass", "fail", "unavailable", "skip")
DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 600.0
HERE = Path(__file__).resolve().parent


def _check(identifier, status, message, details=None, *, required=False, requested=True):
    if status not in STATUSES:
        raise ValueError(f"invalid check status {status!r}")
    return {"id": identifier, "status": status, "message": message,
            "required": bool(required), "requested": bool(requested), "details": details or {}}


def _optional_failure_status(error):
    """Missing optional packages/files are unavailable; executed kernels are failures."""
    return "unavailable" if isinstance(error, (ImportError, FileNotFoundError)) else "fail"


def _timeout(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("timeout must be a number") from None
    if not math.isfinite(number) or not 1 <= number <= MAX_TIMEOUT:
        raise argparse.ArgumentTypeError(f"timeout must be between 1 and {MAX_TIMEOUT:g} seconds")
    return number


def _device(value):
    if not isinstance(value, str) or not re.fullmatch(r"auto|cuda(?::\d+)?", value):
        raise argparse.ArgumentTypeError("device must be auto, cuda or cuda:N")
    return value


def _child_env(root):
    """Pass project paths without letting an inherited PYTHONPATH shadow the venv torch."""
    env = child_environment(root)
    env.pop("PYTHONPATH", None)
    return env


def _torch_worker(requested_device):
    """Import and exercise PyTorch in an isolated, bounded subprocess."""
    report = {"backend": "unknown", "backend_status": "unavailable",
              "tensor_status": "skip", "deform_conv2d_status": "skip",
              "checks": {}, "torch_version": None, "hip_version": None,
              "cuda_build_version": None, "selected_device": None, "device_name": None}
    try:
        import torch
    except Exception as error:  # missing/incompatible torch is an unavailable runtime
        report["reason"] = f"PyTorch import failed: {type(error).__name__}: {error}"
        return report

    report["torch_version"] = str(torch.__version__)
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_build_version = getattr(version, "cuda", None)
    report["hip_version"] = str(hip_version) if hip_version else None
    report["cuda_build_version"] = str(cuda_build_version) if cuda_build_version else None
    if hip_version:
        report["backend"] = "rocm-hip"
    elif cuda_build_version:
        report["backend"] = "nvidia-cuda"
        report["reason"] = "NVIDIA CUDA PyTorch build detected; it does not validate the AMD GPU"
        return report
    else:
        report["backend"] = "cpu-or-unknown"
        report["reason"] = "PyTorch build does not report a HIP runtime"
        return report

    try:
        if not torch.cuda.is_available():
            report["reason"] = "PyTorch reports a HIP build, but no GPU is available to this process"
            return report
        count = int(torch.cuda.device_count())
        index = int(requested_device.split(":", 1)[1]) if requested_device.startswith("cuda:") else 0
        if count <= index:
            report["backend_status"] = "fail"
            report["reason"] = f"requested GPU {index}, but PyTorch reports {count} device(s)"
            return report
        device = torch.device(f"cuda:{index}")
        report["selected_device"] = str(device)
        report["device_name"] = str(torch.cuda.get_device_name(index))
        report["backend_status"] = "pass"
    except Exception as error:
        report["backend_status"] = "fail"
        report["reason"] = f"HIP GPU discovery failed: {type(error).__name__}: {error}"
        return report

    try:
        a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        result = a @ a.T
        torch.cuda.synchronize(device)
        report["checks"]["tensor_shape"] = list(result.shape) == [4, 4]
        report["checks"]["tensor_finite"] = bool(torch.isfinite(result).all().item())
        report["checks"]["tensor_value"] = float(result[0, 0].item()) == 14.0
        if not all(report["checks"].values()):
            raise RuntimeError("tensor smoke returned an unexpected shape or value")
        report["tensor_status"] = "pass"
    except Exception as error:
        report["tensor_status"] = "fail"
        report["tensor_error"] = f"{type(error).__name__}: {error}"

    try:
        import torchvision
        from torchvision.ops import deform_conv2d

        image = torch.arange(64, dtype=torch.float32, device=device).reshape(1, 1, 8, 8)
        weight = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        offset = torch.zeros((1, 18, 6, 6), dtype=torch.float32, device=device)
        output = deform_conv2d(image, offset, weight)
        torch.cuda.synchronize(device)
        if list(output.shape) != [1, 1, 6, 6] or not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"unexpected deform_conv2d output: shape={list(output.shape)}")
        report["deform_conv2d_status"] = "pass"
        report["torchvision_version"] = str(getattr(torchvision, "__version__", "unknown"))
        report["deform_conv2d_shape"] = list(output.shape)
    except Exception as error:
        report["deform_conv2d_status"] = "fail"
        report["deform_conv2d_error"] = f"{type(error).__name__}: {error}"
    return report


def _invoke_json(argv, *, timeout, cwd, env):
    result = run_command(argv, timeout=timeout, cwd=cwd, env=env)
    if result["timed_out"]:
        raise TimeoutError(f"worker exceeded {timeout:g}s")
    if result["returncode"] != 0:
        raise RuntimeError(f"worker exited {result['returncode']}: {result['stderr'][-1200:]}")
    try:
        return last_json(result["stdout"])
    except ValueError as error:
        raise RuntimeError(f"worker did not return JSON: {result['stderr'][-1200:]}") from error


def _worker_check(root, device, timeout):
    try:
        payload = _invoke_json([sys.executable, __file__, "--torch-worker", "--device", device],
                               timeout=timeout, cwd=HERE, env=_child_env(root))
    except TimeoutError as error:
        return None, _check("torch-backend", "fail", str(error), required=True), \
            _check("torch-tensor", "skip", "worker timed out before tensor smoke"), \
            _check("torchvision-deform-conv2d", "skip", "worker timed out before operator smoke")
    except Exception as error:
        details = {"error": f"{type(error).__name__}: {error}"}
        return None, _check("torch-backend", "fail", "PyTorch worker failed", details, required=True), \
            _check("torch-tensor", "skip", "PyTorch worker did not return results"), \
            _check("torchvision-deform-conv2d", "skip", "PyTorch worker did not return results")

    backend_status = payload.get("backend_status", "fail")
    if backend_status not in STATUSES:
        backend_status = "fail"
    backend_message = {
        "pass": f"ROCm/HIP GPU {payload.get('selected_device')} ({payload.get('device_name')})",
        "unavailable": payload.get("reason", f"PyTorch backend is {payload.get('backend', 'unknown')}"),
        "fail": payload.get("reason", "HIP backend probe failed"),
    }.get(backend_status, "HIP backend was not identified")
    details = {key: payload.get(key) for key in (
        "backend", "torch_version", "hip_version", "cuda_build_version", "selected_device", "device_name", "reason")
        if payload.get(key) is not None}
    backend_check = _check("torch-backend", backend_status, backend_message, details, required=True)
    if backend_status != "pass":
        return payload, backend_check, \
            _check("torch-tensor", "skip", "requires a reachable ROCm/HIP GPU"), \
            _check("torchvision-deform-conv2d", "skip", "requires a reachable ROCm/HIP GPU")

    tensor_status = payload.get("tensor_status", "fail")
    tensor_message = "GPU tensor arithmetic passed" if tensor_status == "pass" else payload.get("tensor_error", "GPU tensor arithmetic failed")
    tensor = _check("torch-tensor", tensor_status, tensor_message,
                    {"checks": payload.get("checks", {})}, required=True)
    deform_status = payload.get("deform_conv2d_status", "fail")
    deform_message = "torchvision deform_conv2d executed on the selected GPU" if deform_status == "pass" else payload.get("deform_conv2d_error", "deform_conv2d failed")
    deform = _check("torchvision-deform-conv2d", deform_status, deform_message,
                    {key: payload.get(key) for key in ("torchvision_version", "deform_conv2d_shape", "deform_conv2d_error") if payload.get(key) is not None},
                    required=True)
    return payload, backend_check, tensor, deform


def _encoder_list(ffmpeg, *, timeout, env):
    result = run_command([ffmpeg, "-hide_banner", "-encoders"], timeout=timeout, cwd=HERE, env=env)
    if result["timed_out"]:
        raise TimeoutError("FFmpeg encoder inventory timed out")
    if result["returncode"] is None:
        raise FileNotFoundError(result["stderr"] or "FFmpeg could not be started")
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"][-1200:] or "FFmpeg encoder inventory failed")
    return result["stdout"] + "\n" + result["stderr"]


def _probe_media(ffprobe, path, *, timeout, env, packets=False):
    argv = [ffprobe, "-v", "error", "-of", "json"]
    if packets:
        argv += ["-select_streams", "v:0", "-show_packets", "-show_entries", "packet=pts_time,dts_time,flags"]
    else:
        argv += ["-show_streams", "-show_format"]
    result = run_command(argv + [str(path)], timeout=timeout, cwd=HERE, env=env)
    if result["returncode"] is None:
        raise FileNotFoundError(result["stderr"] or f"ffprobe could not be started: {ffprobe}")
    if result["timed_out"] or result["returncode"] != 0:
        raise RuntimeError("ffprobe timed out" if result["timed_out"] else result["stderr"][-1200:])
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe did not return JSON: {error}") from error


def validate_segment_packets(data):
    """Validate a reset-timestamp segment's packet ordering and initial keyframe."""
    packets = data.get("packets") if isinstance(data, dict) else None
    if not packets:
        raise ValueError("segment has no video packets")
    first = packets[0]
    if "K" not in first.get("flags", ""):
        raise ValueError("first video packet is not a keyframe")
    try:
        times = [float(row["pts_time"]) for row in packets]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("segment packet is missing a valid presentation timestamp") from error
    if any(not math.isfinite(value) for value in times):
        raise ValueError("segment contains a non-finite presentation timestamp")
    if times[0] < -0.01 or times[0] > 0.1:
        raise ValueError(f"reset segment starts at {times[0]:.6f}s, expected approximately zero")
    if any(after < before for before, after in zip(times, times[1:])):
        raise ValueError("video presentation timestamps are not monotonic")
    return {"packet_count": len(packets), "first_pts_s": times[0], "last_pts_s": times[-1],
            "starts_with_keyframe": True, "timestamps_monotonic": True}


def _amf_encode(ffmpeg, ffprobe, *, source, duration, scratch, timeout, env):
    try:
        listing = _encoder_list(ffmpeg, timeout=min(timeout, 15), env=env)
    except FileNotFoundError as error:
        return _check("amf-encode", "unavailable", f"FFmpeg is unavailable: {error}")
    except TimeoutError as error:
        return _check("amf-encode", "fail", str(error))
    except Exception as error:
        return _check("amf-encode", "fail", f"FFmpeg encoder inventory failed: {error}")
    listed = bool(re.search(r"(?m)^\s*V[^\n]*\bh264_amf\b", listing))
    details = {"encoder_listed": listed, "encoder": "h264_amf", "duration_requested_s": duration}
    if not listed:
        return _check("amf-encode", "unavailable", "this FFmpeg build does not list h264_amf; no AMF encode was attempted", details)

    output = Path(scratch) / "amf-smoke.mkv"
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if source is None:
        argv += ["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30"]
    else:
        argv += ["-i", str(source)]
    try:
        from oh_streamer.processing import encoding_backend as E
        argv += ["-map", "0:v:0", "-an", "-t", str(duration)]
        # Exercise the same AMF live quality/latency settings used by Lada.
        argv += E.encoder_args("h264_amf", purpose="live")
    except Exception as error:
        return _check("amf-encode", "fail", f"cannot configure the project's AMF encoder: {error}", details)
    argv += ["-f", "matroska", str(output)]
    encoded = run_command(argv, timeout=timeout, cwd=HERE, env=env)
    details["encode_elapsed_s"] = encoded.get("elapsed_s")
    details["encode_returncode"] = encoded.get("returncode")
    if encoded["timed_out"]:
        details["error"] = "actual AMF encode timed out"
        return _check("amf-encode", "fail", details["error"], details)
    if encoded["returncode"] is None:
        details["error"] = (encoded.get("stderr") or "FFmpeg could not be started")[-2000:]
        return _check("amf-encode", "unavailable", "FFmpeg could not be started", details)
    if encoded["returncode"] != 0 or not output.is_file() or output.stat().st_size == 0:
        details["error"] = (encoded.get("stderr") or "actual AMF encode produced no output")[-2000:]
        return _check("amf-encode", "fail", "h264_amf was listed, but an actual short encode failed", details)
    if not ffprobe:
        details["encode_succeeded"] = True
        return _check("amf-encode", "unavailable", "encode completed, but ffprobe is unavailable for output validation", details)
    try:
        data = _probe_media(ffprobe, output, timeout=min(timeout, 20), env=env)
        streams = data.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        actual_duration = float(data.get("format", {}).get("duration", 0))
        if not video or video.get("codec_name") != "h264":
            raise ValueError("ffprobe output does not contain an H.264 video stream")
        if not math.isfinite(actual_duration) or actual_duration <= 0 or abs(actual_duration - duration) > max(.3, duration * .15):
            raise ValueError(f"encoded output duration {actual_duration:g}s is outside the expected range")
        details.update({"codec_name": video.get("codec_name"), "duration_s": actual_duration,
                        "output_bytes": output.stat().st_size})
        return _check("amf-encode", "pass", "a short h264_amf encode completed and ffprobe validated its H.264 output", details, required=True)
    except Exception as error:
        details["validation_error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, FileNotFoundError):
            details["encode_succeeded"] = True
            return _check("amf-encode", "unavailable", "encode completed, but ffprobe could not be started", details)
        return _check("amf-encode", "fail", "AMF encode output did not pass ffprobe validation", details, required=True)


def _segment_probe(ffmpeg, ffprobe, *, source, duration, scratch, timeout, env):
    details = {"encoder": "h264_amf", "segment_seconds": 1, "duration_requested_s": duration}
    if not ffmpeg or not ffprobe:
        return _check("amf-segments", "unavailable", "requires both ffmpeg and ffprobe")
    try:
        listing = _encoder_list(ffmpeg, timeout=min(timeout, 15), env=env)
    except FileNotFoundError as error:
        return _check("amf-segments", "unavailable", f"FFmpeg is unavailable: {error}")
    except Exception as error:
        return _check("amf-segments", "fail", f"FFmpeg encoder inventory failed: {error}")
    if not re.search(r"(?m)^\s*V[^\n]*\bh264_amf\b", listing):
        return _check("amf-segments", "unavailable", "this FFmpeg build does not list h264_amf")
    directory = Path(scratch) / "segments"
    directory.mkdir(parents=True, exist_ok=True)
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if source is None:
        argv += ["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30"]
    else:
        argv += ["-i", str(source)]
    # Normalize arbitrary source frame rates because the project's 30-frame GOP
    # is intended to align to one-second segment boundaries.
    argv += ["-vf", "fps=30"]
    # Reuse the project's segmented-AMF settings, including forced IDRs at each boundary.
    try:
        from oh_streamer.processing import encoding_backend as E
        argv += ["-map", "0:v:0", "-an", "-t", str(duration)]
        argv += E.encoder_args("h264_amf", purpose="segmented")
        argv += E.quality_args("h264_amf")
        argv += E.gop_args("h264_amf", 30)
    except Exception as error:
        return _check("amf-segments", "fail", f"cannot configure the project's segmented AMF encoder: {error}")
    argv += ["-f", "segment", "-segment_format", "matroska", "-segment_time", "1",
             "-reset_timestamps", "1", str(directory / "segment_%02d.mkv")]
    encoded = run_command(argv, timeout=timeout, cwd=HERE, env=env)
    details["encode_elapsed_s"] = encoded.get("elapsed_s")
    if encoded["returncode"] is None:
        details["error"] = (encoded.get("stderr") or "FFmpeg could not be started")[-2000:]
        return _check("amf-segments", "unavailable", "FFmpeg could not be started", details)
    if encoded["timed_out"] or encoded["returncode"] != 0:
        details["error"] = ("segment encode timed out" if encoded["timed_out"] else (encoded.get("stderr") or "segment encode failed")[-2000:])
        return _check("amf-segments", "fail", "actual segmented h264_amf encode failed", details)
    files = sorted(directory.glob("segment_*.mkv"))
    details["segment_count"] = len(files)
    if len(files) < 2:
        return _check("amf-segments", "fail", f"expected at least two one-second segments, found {len(files)}", details)
    try:
        segments = []
        for path in files:
            packet_data = _probe_media(ffprobe, path, timeout=min(timeout, 20), env=env, packets=True)
            segments.append({"file": path.name, **validate_segment_packets(packet_data)})
        details["segments"] = segments
        details["validation"] = "each segment starts near zero, has monotonic PTS, and begins with a video keyframe"
        return _check("amf-segments", "pass", "actual segmented encode passed per-segment ffprobe timestamp/keyframe checks", details)
    except FileNotFoundError as error:
        details["validation_error"] = f"{type(error).__name__}: {error}"
        return _check("amf-segments", "unavailable", "segment encode completed, but ffprobe could not be started", details)
    except Exception as error:
        details["validation_error"] = f"{type(error).__name__}: {error}"
        return _check("amf-segments", "fail", "segmented output failed ffprobe timestamp/keyframe validation", details)


def _decode_two_frames(ffmpeg, path, scratch, timeout, env):
    if not Path(path).is_file():
        raise FileNotFoundError(f"video input does not exist: {path}")
    output = Path(scratch) / "model_frames.rgb"
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(path),
            "-frames:v", "2", "-vf", "scale=256:256", "-pix_fmt", "rgb24", "-f", "rawvideo", str(output)]
    result = run_command(argv, timeout=timeout, cwd=HERE, env=env)
    expected = 2 * 256 * 256 * 3
    if result["timed_out"] or result["returncode"] != 0:
        raise RuntimeError("video frame decode timed out" if result["timed_out"] else (result.get("stderr") or "ffmpeg frame decode failed")[-1200:])
    if not output.is_file() or output.stat().st_size != expected:
        raise RuntimeError(f"expected exactly two 256x256 RGB frames ({expected} bytes), got {output.stat().st_size if output.exists() else 0}")
    return output


def _resolve_reference_paths(root, references):
    project = Path(root).resolve()
    resolved = []
    for value in references:
        path = Path(value).expanduser()
        resolved.append(str((path if path.is_absolute() else project / path).resolve()))
    return resolved


def _model_worker(root, device, frame_file, reference_site_packages):
    """Run small real Lada inference while keeping the current env's torch first."""
    result = {"restoration_status": "skip", "detector_status": "skip", "checks": {}}
    try:
        import torch
    except Exception as error:
        result["restoration_status"] = result["detector_status"] = "unavailable"
        result["reason"] = f"PyTorch import failed: {type(error).__name__}: {error}"
        return result
    result["torch_version"] = str(torch.__version__)
    if not getattr(getattr(torch, "version", None), "hip", None):
        result["restoration_status"] = result["detector_status"] = "unavailable"
        result["reason"] = "model worker did not import a ROCm/HIP PyTorch build"
        return result
    # These paths are appended only after importing this venv's torch; they supply
    # optional app dependencies without allowing the NVIDIA torch to shadow ROCm.
    appended = []
    for value in reference_site_packages:
        path = str(Path(value).resolve())
        if Path(path).is_dir() and path not in sys.path:
            sys.path.append(path)
            appended.append(path)
    result["reference_site_packages_appended"] = appended
    project = Path(root).resolve()
    lada_root = project / "lada"
    sys.path.insert(0, str(lada_root))
    os.environ["LADA_MODEL_WEIGHTS_DIR"] = str(lada_root / "model_weights")
    try:
        import numpy as np
        if frame_file:
            raw = np.fromfile(frame_file, dtype=np.uint8)
            if raw.size != 2 * 256 * 256 * 3:
                raise ValueError(f"decoded frame file has {raw.size} bytes")
            frames = [raw[:256 * 256 * 3].reshape(256, 256, 3), raw[256 * 256 * 3:].reshape(256, 256, 3)]
        else:
            grid = np.arange(256 * 256 * 3, dtype=np.uint16).reshape(256, 256, 3)
            first = (grid % 220).astype(np.uint8)
            second = ((grid + 9) % 220).astype(np.uint8)
            frames = [first, second]
    except Exception as error:
        result["restoration_status"] = result["detector_status"] = _optional_failure_status(error)
        result["reason"] = f"frame setup failed: {type(error).__name__}: {error}"
        return result

    selected = torch.device(device)
    restoration_path = lada_root / "model_weights" / "lada_mosaic_restoration_model_generic_v1.2.pth"
    if not restoration_path.is_file():
        result["restoration_status"] = "unavailable"
        result["restoration_error"] = f"restoration weights missing: {restoration_path}"
    else:
        try:
            started = time.perf_counter()
            from lada.models.basicvsrpp.inference import inference, load_model
            model = load_model(None, str(restoration_path), selected, fp16=False)
            output = inference(model, frames, selected)
            torch.cuda.synchronize(selected)
            if len(output) != 2 or any(frame.shape != (256, 256, 3) or frame.dtype != np.uint8 for frame in output):
                raise RuntimeError("BasicVSR++ inference returned unexpected output frames")
            if not all(np.isfinite(frame).all() for frame in output):
                raise RuntimeError("BasicVSR++ inference returned non-finite output")
            result["restoration_status"] = "pass"
            result["restoration_model"] = restoration_path.name
            result["restoration_output_shapes"] = [list(frame.shape) for frame in output]
            result["restoration_elapsed_s"] = round(time.perf_counter() - started, 4)
        except Exception as error:
            result["restoration_status"] = _optional_failure_status(error)
            result["restoration_error"] = f"{type(error).__name__}: {error}"

    detector_path = lada_root / "model_weights" / "lada_mosaic_detection_model_v4_fast.pt"
    if not detector_path.is_file():
        result["detector_status"] = "unavailable"
        result["detector_error"] = f"detector weights missing: {detector_path}"
    else:
        try:
            started = time.perf_counter()
            from ultralytics import YOLO
            detector = YOLO(str(detector_path))
            predictions = detector.predict(source=frames[0], device=str(selected), imgsz=160,
                                           verbose=False, half=False)
            torch.cuda.synchronize(selected)
            if not predictions:
                raise RuntimeError("YOLO returned no prediction result")
            result["detector_status"] = "pass"
            result["detector_model"] = detector_path.name
            result["detector_elapsed_s"] = round(time.perf_counter() - started, 4)
            result["detector_result_count"] = len(predictions)
        except Exception as error:
            result["detector_status"] = _optional_failure_status(error)
            result["detector_error"] = f"{type(error).__name__}: {error}"
    return result


def _optional_model_checks(root, device, timeout, scratch, video, ffmpeg, references, env):
    if device is None:
        return [_check("model-basicvsrpp", "skip", "requires a successfully probed ROCm/HIP GPU"),
                _check("model-yolo", "skip", "requires a successfully probed ROCm/HIP GPU")]
    references = _resolve_reference_paths(root, references)
    frame_file = None
    if video is not None:
        if not ffmpeg:
            return [_check("model-basicvsrpp", "unavailable", "--model-smoke with --video requires ffmpeg"),
                    _check("model-yolo", "unavailable", "--model-smoke with --video requires ffmpeg")]
        try:
            frame_file = _decode_two_frames(ffmpeg, video, scratch, timeout, env)
        except FileNotFoundError as error:
            return [_check("model-basicvsrpp", "fail", str(error)), _check("model-yolo", "fail", str(error))]
        except Exception as error:
            details = {"error": f"{type(error).__name__}: {error}"}
            return [_check("model-basicvsrpp", "fail", "could not decode two frames for model smoke", details),
                    _check("model-yolo", "fail", "could not decode a frame for detector smoke", details)]
    command = [sys.executable, __file__, "--model-worker", "--project-root", str(root),
               "--device", device]
    if frame_file:
        command += ["--model-frame-file", str(frame_file)]
    for path in references:
        command += ["--reference-site-packages", str(path)]
    try:
        payload = _invoke_json(command, timeout=timeout, cwd=HERE, env=env)
    except Exception as error:
        details = {"error": f"{type(error).__name__}: {error}"}
        return [_check("model-basicvsrpp", "fail", "model worker failed", details),
                _check("model-yolo", "fail", "model worker failed", details)]
    checks = []
    for identifier, key, err_key, title in (
            ("model-basicvsrpp", "restoration_status", "restoration_error", "two-frame BasicVSR++ inference"),
            ("model-yolo", "detector_status", "detector_error", "one-frame YOLO detector inference")):
        status = payload.get(key, "fail")
        if status not in STATUSES:
            status = "fail"
        error = payload.get(err_key)
        message = title + " passed" if status == "pass" else (error or f"{title} {status}")
        details = {k: v for k, v in payload.items() if k not in (key, "restoration_status", "detector_status", "restoration_error", "detector_error")}
        if error:
            details["error"] = error
        checks.append(_check(identifier, status, message, details, required=True))
    return checks


def _rife_check(root, device, timeout, env, *, tensor_status="pass"):
    if device is None:
        return _check("model-rife", "skip", "requires a successfully probed ROCm/HIP GPU")
    if tensor_status != "pass":
        return _check("model-rife", "skip", "blocked because the ROCm GPU tensor smoke did not pass",
                      {"blocking_check": "torch-tensor", "blocking_status": tensor_status})
    command = [sys.executable, HERE / "probe_rife.py", "--device", device,
               "--size", "64x64", "--iterations", "1", "--timeout", str(min(timeout, 300)), "--json"]
    result = run_command(command, timeout=timeout + 5, cwd=HERE, env=env)
    if result["timed_out"]:
        return _check("model-rife", "fail", f"RIFE worker exceeded {timeout + 5:g}s timeout")
    if result["returncode"] is None:
        return _check("model-rife", "unavailable", "RIFE worker could not be started",
                      {"error": result.get("stderr", "")[-1200:]})
    try:
        # probe_rife intentionally exits 1 when its JSON report says "fail".
        # Parse that report before interpreting the exit code so missing optional
        # model dependencies can be reported as unavailable.
        payload = last_json(result["stdout"])
    except ValueError as error:
        return _check("model-rife", "fail", "RIFE worker did not return a JSON report",
                      {"returncode": result["returncode"], "error": result.get("stderr", "")[-1200:] or str(error)})
    status = payload.get("status")
    if status == "pass" and result["returncode"] == 0:
        return _check("model-rife", "pass", "one small RIFE inference passed", payload, required=True)
    error = str(payload.get("error", "RIFE worker reported failure"))
    unavailable = any(token in error.lower() for token in ("modulenotfounderror", "filenotfounderror", "no such file"))
    final_status = "unavailable" if unavailable else "fail"
    if result["returncode"] not in (0, 1):
        final_status = "fail"
        error = f"RIFE worker exited unexpectedly ({result['returncode']}): {error}"
    payload["worker_returncode"] = result["returncode"]
    return _check("model-rife", final_status, error, payload, required=True)


def build_report(*, root=None, device="auto", video=None, segments=False, model_smoke=False,
                 rife=False, reference_site_packages=(), timeout=DEFAULT_TIMEOUT,
                 ffmpeg=None, ffprobe=None, skip_amf=False, scratch_root=None):
    root = project_root(root)
    root.mkdir(parents=True, exist_ok=True)
    reference_site_packages = _resolve_reference_paths(root, reference_site_packages)
    source = Path(video).expanduser().resolve() if video else None
    if source is not None and not source.is_file():
        raise ValueError(f"video input does not exist: {source}")
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    env = _child_env(root)
    scratch_root = Path(scratch_root) if scratch_root else root / "testdata" / "amd-probe-tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix="run-", dir=scratch_root) as scratch:
        torch_payload, backend, tensor, deform = _worker_check(root, device, timeout)
        checks.extend((backend, tensor, deform))
        selected = None
        if torch_payload and backend["status"] == "pass":
            selected = torch_payload.get("selected_device")

        if skip_amf:
            checks.append(_check("amf-encode", "skip", "disabled by --skip-amf", requested=False))
        elif not ffmpeg:
            checks.append(_check("amf-encode", "unavailable", "ffmpeg was not found on PATH"))
        else:
            checks.append(_amf_encode(ffmpeg, ffprobe, source=source, duration=2.0,
                                      scratch=scratch, timeout=timeout, env=env))

        if segments:
            checks.append(_segment_probe(ffmpeg, ffprobe, source=source, duration=4.0,
                                         scratch=scratch, timeout=timeout, env=env))
        else:
            checks.append(_check("amf-segments", "skip", "use --segments to encode and inspect short segments", requested=False))

        if model_smoke:
            if tensor["status"] != "pass":
                blocked = {"blocking_check": "torch-tensor", "blocking_status": tensor["status"]}
                checks.extend((_check("model-basicvsrpp", "skip", "blocked because the ROCm GPU tensor smoke did not pass", blocked),
                               _check("model-yolo", "skip", "blocked because the ROCm GPU tensor smoke did not pass", blocked)))
            else:
                checks.extend(_optional_model_checks(root, selected, timeout, scratch, source,
                                                     ffmpeg, reference_site_packages, env))
        else:
            checks.extend((_check("model-basicvsrpp", "skip", "use --model-smoke for two-frame BasicVSR++ inference", requested=False),
                           _check("model-yolo", "skip", "use --model-smoke for one-frame YOLO inference", requested=False)))

        if rife:
            checks.append(_rife_check(root, selected, timeout, env, tensor_status=tensor["status"]))
        else:
            checks.append(_check("model-rife", "skip", "use --rife for a small RIFE inference", requested=False))

    failures = [row for row in checks if row["status"] == "fail" and row["requested"]]
    if failures:
        overall = "fail"
    elif backend["status"] != "pass":
        overall = "unavailable"
    elif any(row["status"] == "unavailable" and row["requested"] for row in checks):
        overall = "partial"
    else:
        overall = "pass"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "amd-development-probe",
        "environment": environment(),
        "status": overall,
        "requested_device": device,
        "selected_device": selected,
        "options": {"video": str(source) if source else None, "segments": segments,
                    "model_smoke": model_smoke, "rife": rife, "skip_amf": skip_amf,
                    "reference_site_packages": reference_site_packages,
                    "timeout_s": timeout},
        "checks": checks,
        "note": "A pass covers only checks reported as pass. It does not establish AMD product support, full Lada compatibility, quality, throughput, or whole-video correctness.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", help="repository containing lada/ and streamer/")
    parser.add_argument("--device", type=_device, default="auto", help="auto, cuda or cuda:N")
    parser.add_argument("--video", help="optional local source video used for AMF and model smoke")
    parser.add_argument("--segments", action="store_true", help="also encode short AMF segments and validate timestamps/keyframes")
    parser.add_argument("--model-smoke", action="store_true", help="load BasicVSR++ and YOLO weights and run tiny GPU inference")
    parser.add_argument("--rife", action="store_true", help="run the existing small eager RIFE inference probe")
    parser.add_argument("--reference-site-packages", action="append", default=[],
                        help="append dependency-only site-packages after the ROCm venv torch; repeat as needed")
    parser.add_argument("--timeout", type=_timeout, default=DEFAULT_TIMEOUT, help="per subprocess timeout, 1-600 seconds")
    parser.add_argument("--ffmpeg", help="FFmpeg executable, otherwise PATH")
    parser.add_argument("--ffprobe", help="ffprobe executable, otherwise PATH")
    parser.add_argument("--skip-amf", action="store_true", help="skip the default short actual AMF encode")
    parser.add_argument("--output", help="write the JSON report to this path")
    parser.add_argument("--torch-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-frame-file", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.torch_worker:
        print(json.dumps(_torch_worker(args.device), ensure_ascii=False))
        return 0
    if args.model_worker:
        print(json.dumps(_model_worker(args.project_root, args.device, args.model_frame_file,
                                       args.reference_site_packages), ensure_ascii=False))
        return 0
    try:
        report = build_report(root=args.project_root, device=args.device, video=args.video,
                              segments=args.segments, model_smoke=args.model_smoke, rife=args.rife,
                              reference_site_packages=args.reference_site_packages, timeout=args.timeout,
                              ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, skip_amf=args.skip_amf)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        write_json(args.output, report)
    print(payload)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
