"""Benchmark RIFE frame interpolation (30 -> 60 fps: one middle frame per frame pair) on this GPU, alone and while
Lada restores a video in the same process. The RIFE wrapper (Rife) lives in rife_interp.py, which the player's
adaptive interpolation uses too.

RIFE code + weights live in third_party/rife (see the README there). Every config runs in its own subprocess (clean
VRAM peak, no cross-talk between models).

    lada\\.venv\\Scripts\\python streamer\\benchmarks\\bench_rife.py alone [--models 4.6,4.22.lite] [--res 720,1080]
    lada\\.venv\\Scripts\\python streamer\\benchmarks\\bench_rife.py contention testdata\\restore_bench.mp4 --start 120 --duration 40
    lada\\.venv\\Scripts\\python streamer\\benchmarks\\bench_rife.py all testdata\\restore_bench.mp4 --out testdata\\bench\\rife_result.txt

Modes of one RIFE measurement ("alone"):
  gpu      frames already on the GPU as padded fp16 NCHW; per interpolation: encode(new frame) + flownet. This is the
           pure model cost.
  full     frames come in as CPU uint8 HWC BGR torch tensors (exactly what FrameRestorer yields); per interpolation:
           upload + convert/pad + encode + flownet + crop/convert + download to a numpy-backed uint8 HWC frame (what
           the NVENC pipe needs). This is what the pipeline would pay per output frame.
  xfer     only the upload + download part of "full" (no model).
"contention": FrameRestorer runs (its own worker threads, as in the player) and the consuming thread interpolates
between consecutive restored frames ("full" path) before (optionally) piping both frames to an h264_nvenc ffmpeg, i.e.
RIFE sits where it would sit in the real pipeline: after the restorer, before the encoder, same process / CUDA context.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

STREAMER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STREAMER_ROOT.parent
DEFAULT_MODELS = "4.6,4.22.lite,4.25.lite,4.26"

sys.path.insert(0, str(STREAMER_ROOT))
sys.path.insert(0, str(STREAMER_ROOT / "src"))
from oh_streamer.processing.rife_interp import MODELS, RIFE_DIR, Rife  # noqa: E402,F401  (the same RIFE wrapper the player uses)

RES = {"540": (960, 540), "720": (1280, 720), "1080": (1920, 1080)}


def gpu_busy_note() -> dict:
    """What else is on the GPU right now (WDDM often hides per-process usage, so also record total memory / util)."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return {"gpu_before": gpu, "gpu_apps_before": apps or "none"}
    except Exception as e:  # noqa: BLE001
        return {"gpu_before": f"nvidia-smi failed: {e}"}


# ------------------------------------------------------------------------------------------------------- frame sources

def decode_frames(path: str, n: int, w: int, h: int, start: float = 0.0):
    """n consecutive frames as uint8 HWC BGR CPU torch tensors (what FrameRestorer yields), resized to w x h."""
    import av
    import torch
    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if start > 0:
            c.seek(int(start / st.time_base), stream=st)
        for fr in c.decode(st):
            if fr.width != w or fr.height != h:
                fr = fr.reformat(width=w, height=h)
            out.append(torch.from_numpy(fr.to_ndarray(format="bgr24")))
            if len(out) >= n:
                break
    return out


# -------------------------------------------------------------------------------------------------------- alone bench

def run_alone(model: str, res: str, scale: float, iters: int, channels_last: bool, cuda_graph: bool,
              cudnn_bench: bool, src: str) -> dict:
    import torch
    torch.backends.cudnn.benchmark = cudnn_bench
    device = torch.device("cuda:0")
    w, h = RES[res]
    frames = decode_frames(src, 48, w, h)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    rife = Rife(model, w, h, scale, device, True, channels_last, cuda_graph)
    with torch.inference_mode():
        rife.step(rife.prep(frames[0]))
        o = rife.step(rife.prep(frames[1]))
        rife.post(o).cpu()
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0  # model load + first interpolation (cudnn init / autotune, graph capture)

    res_out = {"model": model, "res": f"{w}x{h}", "pad": f"{rife.pw}x{rife.ph}", "scale": scale,
               "channels_last": channels_last, "cuda_graph": cuda_graph, "cudnn_benchmark": cudnn_bench,
               "cold_start_s": round(cold_s, 2)}

    with torch.inference_mode():
        # gpu: pre-uploaded frames
        gpu_frames = [rife.prep(f) for f in frames]
        for i in range(20):  # warmup
            rife.step(gpu_frames[i % len(gpu_frames)])
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(iters):
            rife.step(gpu_frames[i % len(gpu_frames)])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        res_out["gpu_ms"] = round(dt / iters * 1000, 2)
        res_out["gpu_ips"] = round(iters / dt, 1)
        del gpu_frames

        # full: CPU uint8 in -> CPU uint8 out (download synchronizes each frame, like a pipe writer would)
        for i in range(10):
            o = rife.step(rife.prep(frames[i % len(frames)]))
            if o is not None:
                rife.post(o).cpu().numpy()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(iters):
            o = rife.step(rife.prep(frames[i % len(frames)]))
            arr = rife.post(o).cpu().numpy()
        dt = time.perf_counter() - t
        res_out["full_ms"] = round(dt / iters * 1000, 2)
        res_out["full_ips"] = round(iters / dt, 1)

        # xfer: upload + download only
        n = min(iters, 200)
        t = time.perf_counter()
        for i in range(n):
            g = frames[i % len(frames)].to(device, non_blocking=True)
            g.cpu().numpy()
        dt = time.perf_counter() - t
        res_out["xfer_ms"] = round(dt / n * 1000, 2)
    res_out["vram_peak_mib"] = round((torch.cuda.max_memory_allocated() - base_alloc) / 2**20)
    res_out["vram_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
    del arr
    return res_out


def run_compile_probe(model: str, res: str, src: str) -> dict:
    """torch.compile probe: does inductor (needs triton) / cudagraphs backend work here?"""
    import torch
    device = torch.device("cuda:0")
    w, h = RES[res]
    frames = decode_frames(src, 4, w, h)
    out = {"model": model, "res": f"{w}x{h}"}
    for backend in ("inductor", "cudagraphs"):
        try:
            rife = Rife(model, w, h, 1.0, device)
            rife.flownet = torch.compile(rife.flownet, backend=backend)
            with torch.inference_mode():
                rife.step(rife.prep(frames[0]))
                for i in range(3):
                    rife.step(rife.prep(frames[1 + i % 3]))
                torch.cuda.synchronize()
                t = time.perf_counter()
                for i in range(100):
                    rife.step(rife.prep(frames[i % 4]))
                torch.cuda.synchronize()
            out[backend] = f"ok {(time.perf_counter() - t) * 10:.2f} ms/interp (incl. upload)"
        except Exception as e:  # noqa: BLE001
            out[backend] = f"FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return out


# --------------------------------------------------------------------------------------------------- contention bench

def run_contention(src: str, model: str | None, det_name: str, clip_len: int, start: float, duration: float,
                   encode: bool, cuda_graph: bool, cudnn_bench: bool, channels_last: bool, scale: float) -> dict:
    sys.path.insert(0, str(STREAMER_ROOT))
    sys.path.insert(0, str(STREAMER_ROOT / "src"))
    import lada_ns  # noqa: E402  (benchmarks/lada_ns.py: oh_player no longer re-exports Lada's names)
    ls = lada_ns.load()
    torch = ls.torch
    torch.backends.cudnn.benchmark = cudnn_bench
    http = ls.FFmpegHttpOptions({})
    probe = ls.ffprobe(src, http)
    meta = ls.build_metadata(src, probe)
    ls.install_lada_io_hooks(meta, http)
    w, h = meta.video_width, meta.video_height

    device = torch.device("cuda:0")
    fp16 = ls.gpu_has_fp16_acceleration()
    det = ls.ModelFiles.get_detection_model_by_name(det_name)
    res = ls.ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2")
    det_model, res_model, pad_mode = ls.load_models(device, "basicvsrpp-v1.2", res.path, None, det.path, fp16, False)
    restorer = ls.FrameRestorer(device, src, clip_len, "basicvsrpp-v1.2", det_model, res_model, pad_mode)

    rife = None
    rife_load_s = 0.0
    if model:
        t = time.perf_counter()
        rife = Rife(model, w, h, scale, device, True, channels_last, cuda_graph)
        with torch.inference_mode():  # warm it up so the timed run sees steady state (cold cost reported separately)
            z = torch.zeros((h, w, 3), dtype=torch.uint8)
            rife.step(rife.prep(z))
            rife.post(rife.step(rife.prep(z))).cpu()
            rife.prev = None
        torch.cuda.synchronize()
        rife_load_s = time.perf_counter() - t

    enc = None
    if encode:
        fps_out = float(meta.video_fps_exact) * (2 if rife else 1)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps_out:.5f}", "-i", "-",
               "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-pix_fmt", "yuv420p",
               "-f", "null", "-"]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=0)

    fps = float(meta.video_fps_exact)
    max_frames = int(duration * fps)
    frames = 0
    rife_time = 0.0
    write_time = 0.0
    t_first = None
    t0 = time.perf_counter()
    restorer.start(start_ns=int(start * 1e9))
    try:
        with torch.inference_mode():
            for elem in restorer:
                if elem is ls.STOP_MARKER or isinstance(elem, ls.ErrorMarker):
                    raise RuntimeError(f"restorer stopped: {getattr(elem, 'error', elem)}")
                frame, _pts = elem
                if t_first is None:
                    t_first = time.perf_counter()
                mid = None
                if rife is not None:
                    tr = time.perf_counter()
                    o = rife.step(rife.prep(frame))
                    if o is not None:
                        mid = rife.post(o).cpu().numpy()  # .cpu() synchronizes
                    rife_time += time.perf_counter() - tr
                if enc is not None:
                    tw = time.perf_counter()
                    if mid is not None:
                        enc.stdin.write(memoryview(mid.reshape(-1)))
                    arr = frame.numpy() if isinstance(frame, torch.Tensor) else frame
                    enc.stdin.write(memoryview(arr.reshape(-1)))
                    write_time += time.perf_counter() - tw
                frames += 1
                if frames >= max_frames:
                    break
    finally:
        t_end = time.perf_counter()
        restorer.stop()
        if enc is not None:
            enc.stdin.close()
            enc.wait()
    steady = (frames - 1) / max(1e-6, t_end - t_first)
    return {
        "src": Path(src).name, "res": f"{w}x{h}", "rife": model or "none", "det": f"{det_name}:{clip_len}",
        "scale": scale, "encode_nvenc": encode, "cuda_graph": cuda_graph, "channels_last": channels_last, "cudnn_benchmark": cudnn_bench,
        "frames": frames, "first_frame_s": round(t_first - t0, 2),
        "restored_fps": round(steady, 1), "x_realtime": round(steady / fps, 2),
        "output_fps": round(steady * (2 if rife else 1), 1),
        "rife_ms_in_loop": round(rife_time / max(1, frames - 1) * 1000, 2) if rife else None,
        "write_ms_in_loop": round(write_time / max(1, frames) * 1000, 2) if enc else None,
        "rife_load_warm_s": round(rife_load_s, 2) if rife else None,
        "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),  # Lada + RIFE, whole process
        "vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


# --------------------------------------------------------------------------------------------------------------- driver

def sub(args_list: list[str], log) -> dict | None:
    note = gpu_busy_note()
    mode = ["alone"] if args_list[0].startswith("--") else ["contention"]  # positional mode is required by argparse
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one"] + mode + args_list
    print("--- " + " ".join(args_list), file=sys.stderr, flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT / "lada"))
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        err = p.stderr[-3000:]
        print(err, file=sys.stderr)
        log.write(json.dumps({"args": args_list, "error": err[-1500:], **note}) + "\n")
        log.flush()
        return None
    r = json.loads(line[7:])
    r.update(note)
    print(json.dumps(r), flush=True)
    log.write(json.dumps(r) + "\n")
    log.flush()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["alone", "compile", "contention", "all"])
    ap.add_argument("input", nargs="?", default=str(PROJECT_ROOT / "testdata" / "smoke.mp4"),
                    help="video for contention (should contain mosaics); alone uses testdata/smoke.mp4 frames")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--res", default="720,1080")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--start", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--det", default="v4-fast:300", help="detection model:max_clip_length for contention")
    ap.add_argument("--input720", default=None, help="720p version of the contention input (optional)")
    ap.add_argument("--c1080", default="4.6,4.6@0.5,4.22.lite,4.22.lite@0.5", help="model[@scale] list, input")
    ap.add_argument("--c720", default="4.6,4.22.lite,4.25.lite,4.26", help="model[@scale] list, --input720")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "testdata" / "bench" / "rife_result.txt"))
    # internal: one measurement in this process
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--kind", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--scale", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--cl", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--graph", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--nobench", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--encode", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    src = str(Path(args.input).resolve())
    smoke = str(PROJECT_ROOT / "testdata" / "smoke.mp4")

    if args.one:
        if args.kind == "alone":
            r = run_alone(args.model, args.res, args.scale, args.iters, args.cl, args.graph, not args.nobench, smoke)
        elif args.kind == "compile":
            r = run_compile_probe(args.model, args.res, smoke)
        else:
            det, clip = args.det.split(":")
            r = run_contention(src, args.model if args.model != "none" else None, det, int(clip), args.start,
                               args.duration, args.encode, args.graph, not args.nobench, args.cl, args.scale)
        print("RESULT " + json.dumps(r), flush=True)
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as log:
        log.write(f"# bench_rife {args.mode} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.mode in ("alone", "all"):
            for model in args.models.split(","):
                for res in args.res.split(","):
                    base = ["--kind", "alone", "--model", model, "--res", res, "--iters", str(args.iters)]
                    sub(base, log)                                   # plain NCHW, eager
                    sub(base + ["--cl", "--graph"], log)             # channels_last + CUDA graph (best)
                    if res == "1080":
                        sub(base + ["--scale", "0.5", "--cl", "--graph"], log)
            # single-knob variants on one model: channels_last only, CUDA graph only, cudnn.benchmark off
            m0 = args.models.split(",")[0]
            for res in args.res.split(","):
                base = ["--kind", "alone", "--model", m0, "--res", res, "--iters", str(args.iters)]
                sub(base + ["--cl"], log)
                sub(base + ["--graph"], log)
                sub(base + ["--nobench", "--cl", "--graph"], log)
        if args.mode in ("compile", "all"):
            sub(["--kind", "compile", "--model", args.models.split(",")[0], "--res", "720"], log)
        if args.mode in ("contention", "all"):
            # cudnn.benchmark stays off here (Lada's default; with channels_last + CUDA graph it made no steady-state
            # difference alone but costs ~4 s autotune at start)
            runs = [(src, args.start, args.c1080)]
            if args.input720:
                runs.append((str(Path(args.input720).resolve()), 0.0, args.c720))
            for inp, start, specs in runs:
                common = [inp, "--kind", "contention", "--det", args.det, "--start", str(start),
                          "--duration", str(args.duration), "--nobench"]
                sub(common + ["--model", "none"], log)
                sub(common + ["--model", "none", "--encode"], log)
                for spec in specs.split(","):
                    m, _, sc = spec.partition("@")
                    sub(common + ["--model", m, "--scale", sc or "1.0", "--graph", "--cl", "--encode"], log)


if __name__ == "__main__":
    main()
