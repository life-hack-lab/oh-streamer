"""Compare Lada detection model / max-clip-length combinations on the same video segment.

Each config runs in its own subprocess (clean VRAM peak). Reports throughput, first-frame latency, VRAM peak and how
many frames the detector found mosaics in. With --save, also writes each restored segment so they can be compared
side by side.

    lada\\.venv\\Scripts\\python streamer\\benchmarks\\bench_settings.py <file> --start 120 --duration 60 --save testdata\\bench
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STREAMER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAMER_ROOT))
sys.path.insert(0, str(STREAMER_ROOT / "src"))
import lada_ns  # noqa: E402  (benchmarks/lada_ns.py: oh_player no longer re-exports Lada's names)
ls = lada_ns.load()

DEFAULT_CONFIGS = "v4-fast:180,v4-fast:250,v4-accurate:180,v4-accurate:250"


def run_one(src: str, det_name: str, clip_len: int, start: float, duration: float, save: str | None) -> dict:
    http = ls.FFmpegHttpOptions({})
    probe = ls.ffprobe(src, http)
    meta = ls.build_metadata(src, probe)
    has_audio = any(s.get("codec_type") == "audio" for s in probe["streams"])
    fmt_start = float(probe["format"].get("start_time") or 0)
    ls.install_lada_io_hooks(meta, http)

    device = ls.torch.device("cuda:0")
    fp16 = ls.gpu_has_fp16_acceleration()
    det = ls.ModelFiles.get_detection_model_by_name(det_name)
    res = ls.ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2")
    det_model, res_model, pad_mode = ls.load_models(device, "basicvsrpp-v1.2", res.path, None, det.path, fp16, False)
    restorer = ls.FrameRestorer(device, src, clip_len, "basicvsrpp-v1.2", det_model, res_model, pad_mode)

    # count detections as the restoration worker consumes them
    # detection runs ahead of the output frames: density = mosaics / detection entries, not / frames returned
    counts = {"detection_frames": 0, "frames_with_mosaic": 0, "mosaics": 0}
    q = restorer.frame_detection_queue
    orig_get = q.get

    def counting_get(*a, **kw):
        elem = orig_get(*a, **kw)
        if isinstance(elem, tuple) and len(elem) == 2:
            n = elem[1]
            counts["detection_frames"] += 1
            counts["mosaics"] += n
            counts["frames_with_mosaic"] += 1 if n else 0
        return elem
    q.get = counting_get

    fps = float(meta.video_fps_exact)
    max_frames = int(duration * fps)
    muxer = None
    t0 = time.perf_counter()
    t_first = None
    frames = 0
    restorer.start(start_ns=int(start * 1e9))
    try:
        for elem in restorer:
            if elem is ls.STOP_MARKER or isinstance(elem, ls.ErrorMarker):
                raise RuntimeError(f"restorer stopped: {getattr(elem, 'error', elem)}")
            frame, pts = elem
            if t_first is None:
                t_first = time.perf_counter()
            if save and muxer is None:
                first_s = float(pts * meta.time_base) if pts is not None else start
                out = str(Path(save) / f"{det_name}_{clip_len}.mkv")
                muxer = ls.start_muxer(src, meta, has_audio, max(0.0, first_s - fmt_start), http, out, "h264_nvenc")
            if muxer is not None:
                arr = frame.numpy() if isinstance(frame, ls.torch.Tensor) else frame
                muxer.stdin.write(memoryview(arr.reshape(-1)))
            frames += 1
            if frames >= max_frames:
                break
    finally:
        t_end = time.perf_counter()
        restorer.stop()
        if muxer is not None:
            muxer.stdin.close()
            muxer.wait()
    steady = frames / max(1e-6, t_end - t_first)
    return {
        "config": f"{det_name}:{clip_len}",
        "frames": frames,
        "first_frame_s": round(t_first - t0, 2),
        "fps": round(steady, 1),
        "x_realtime": round(steady / fps, 2),
        "vram_gib": round(ls.torch.cuda.max_memory_allocated() / 2**30, 2),
        "density": round(counts["mosaics"] / max(1, counts["detection_frames"]), 3),
        **counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="comma-separated det_model:max_clip_length")
    ap.add_argument("--save", default=None, help="directory for restored segments (optional)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    src = str(Path(args.input).resolve())

    if args.one:
        det_name, clip_len = args.configs.split(":")
        print("RESULT " + json.dumps(run_one(src, det_name, int(clip_len), args.start, args.duration, args.save)),
              flush=True)
        return

    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in args.configs.split(","):
        cmd = [sys.executable, str(Path(__file__).resolve()), src, "--one", "--configs", cfg,
               "--start", str(args.start), "--duration", str(args.duration)]
        if args.save:
            cmd += ["--save", args.save]
        print(f"--- {cfg}", file=sys.stderr, flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(p.stderr[-2000:], file=sys.stderr)
            continue
        r = json.loads(line[7:])
        results.append(r)
        print(json.dumps(r), flush=True)

    print(f"\n{'config':<18}{'fps':>7}{'x RT':>7}{'1st frame':>11}{'VRAM':>7}{'frames w/ mosaic':>18}{'mosaics':>9}")
    for r in results:
        print(f"{r['config']:<18}{r['fps']:>7}{r['x_realtime']:>7}{r['first_frame_s']:>10}s{r['vram_gib']:>6}G"
              f"{r['frames_with_mosaic']:>12}/{r['frames']:<5}{r['mosaics']:>9}")


if __name__ == "__main__":
    main()
