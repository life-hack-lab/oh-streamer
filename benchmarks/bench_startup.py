"""Does holding the detector back during a segment's startup make the first frames come out sooner?

After a start / seek, nothing is playable until the first mosaic clip covering the target is detected and restored,
but Lada's detector keeps running ahead (1000+ frames) and shares the GPU with that restoration. This bench runs
the same seek points with:

- off:      unmodified FrameRestorer
- startup:  the detector may not run more than max_clip_length + MARGIN frames ahead of the restored output until
            RELEASE_S seconds of video have been output, then runs freely (what the player would do)
- always:   the same limit for the whole run (shows what a permanent limit costs in throughput)

and reports the time until 1 / 2 / 4 / 10 / 20 s of video have come out, plus the restoration speed of the first
clip. Models are loaded once and every run builds a fresh FrameRestorer, like the player.

    lada\\.venv\\Scripts\\python streamer\\benchmarks\\bench_startup.py testdata\\restore_bench.mp4 --starts 60,180,300,420 --reps 2
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

STREAMER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAMER_ROOT))
sys.path.insert(0, str(STREAMER_ROOT / "src"))
import lada_ns  # noqa: E402  (benchmarks/lada_ns.py: oh_player no longer re-exports Lada's names)
ls = lada_ns.load()

MARGIN = 30          # frames beyond max_clip_length the detector may run ahead (a clip closes one frame after it ends)
RELEASE_S = 2.0      # seconds of output after which the startup gate opens (preroll to the target + seek_prebuffer)
MARKS_S = (1, 2, 4, 10, 20)


class Gate:
    """Blocks the detector's per-frame put while it is too far ahead of the restored output."""

    def __init__(self, restorer, limit: int, release_frames: int | None):
        self.restorer, self.limit, self.release_frames = restorer, limit, release_frames
        self.det_n = self.out_n = 0
        self.max_ahead = 0
        self.blocked_s = 0.0
        self.cond = threading.Condition()
        self.open = limit is None

        det_q, out_q = restorer.frame_detection_queue, restorer.frame_restoration_queue
        det_put, out_put = det_q.put, out_q.put

        def gated_det_put(item, *a, **kw):
            if isinstance(item, tuple):
                with self.cond:
                    t = None
                    while not self.open and not restorer.stop_requested and self.det_n - self.out_n > self.limit:
                        t = t or time.perf_counter()
                        self.cond.wait(0.05)
                    if t:
                        self.blocked_s += time.perf_counter() - t
                    self.det_n += 1
                    self.max_ahead = max(self.max_ahead, self.det_n - self.out_n)
            return det_put(item, *a, **kw)

        def counting_out_put(item, *a, **kw):
            if isinstance(item, tuple):
                with self.cond:
                    self.out_n += 1
                    if self.release_frames is not None and self.out_n >= self.release_frames:
                        self.open = True
                    self.cond.notify_all()
            return out_put(item, *a, **kw)

        det_q.put, out_q.put = gated_det_put, counting_out_put

    def release(self):
        with self.cond:
            self.open = True
            self.cond.notify_all()


def track_first_clip(restorer):
    """Time the first clip's restoration (mosaic_clip_queue.get -> restored_clip_queue.put, both in the worker)."""
    info = {}
    mq, rq = restorer.mosaic_clip_queue, restorer.restored_clip_queue
    mq_get, rq_put = mq.get, rq.put

    def get(*a, **kw):
        clip = mq_get(*a, **kw)
        if "t_get" not in info and hasattr(clip, "frame_start"):
            info["t_get"], info["len"] = time.perf_counter(), len(clip)
        return clip

    def put(clip, *a, **kw):
        if "t_done" not in info and "t_get" in info and hasattr(clip, "frame_start"):
            info["t_done"] = time.perf_counter()
        return rq_put(clip, *a, **kw)

    mq.get, rq.put = get, put
    return info


def run(factory_args, src, meta, start, mode, clip_len, total_s):
    det_model, res_model, pad_mode, device = factory_args
    fps = float(meta.video_fps_exact)
    restorer = ls.FrameRestorer(device, src, clip_len, "basicvsrpp-v1.2", det_model, res_model, pad_mode)
    limit = None if mode == "off" else clip_len + MARGIN
    release = int(RELEASE_S * fps) if mode == "startup" else None
    gate = Gate(restorer, limit, release)
    clip = track_first_clip(restorer)

    marks = {s: None for s in MARKS_S}
    total = int(total_s * fps)
    frames = 0
    t0 = time.perf_counter()
    t_first = None
    restorer.start(start_ns=int(start * 1e9))
    try:
        for elem in restorer:
            if elem is ls.STOP_MARKER or isinstance(elem, ls.ErrorMarker):
                raise RuntimeError(f"restorer stopped: {getattr(elem, 'error', elem)}")
            frames += 1
            now = time.perf_counter() - t0
            t_first = t_first or now
            for s in MARKS_S:
                if marks[s] is None and frames >= int(s * fps):
                    marks[s] = round(now, 2)
            if frames >= total:
                break
    finally:
        gate.release()
        restorer.stop()
    r = {"start": start, "mode": mode, "first_s": round(t_first, 2), **{f"t{s}s": marks[s] for s in MARKS_S},
         "max_ahead": gate.max_ahead, "det_blocked_s": round(gate.blocked_s, 2)}
    if "t_done" in clip:
        r["clip1_len"] = clip["len"]
        r["clip1_wait_s"] = round(clip["t_get"] - t0, 2)          # until the detector closed the first clip
        r["clip1_fps"] = round(clip["len"] / (clip["t_done"] - clip["t_get"]), 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--starts", default="60,180,300,420", help="comma-separated seek positions (s)")
    ap.add_argument("--modes", default="off,startup,always")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--clip", type=int, default=300, help="max_clip_length")
    ap.add_argument("--total", type=float, default=20.0, help="seconds of video to output per run")
    ap.add_argument("--det", default="v4-fast")
    ap.add_argument("--out", default=None, help="append JSON results to this file")
    args = ap.parse_args()
    src = str(Path(args.input).resolve())

    http = ls.FFmpegHttpOptions({})
    probe = ls.ffprobe(src, http)
    meta = ls.build_metadata(src, probe)
    ls.install_lada_io_hooks(meta, http)
    device = ls.torch.device("cuda:0")
    fp16 = ls.gpu_has_fp16_acceleration()
    det = ls.ModelFiles.get_detection_model_by_name(args.det)
    res = ls.ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2")
    det_model, res_model, pad_mode = ls.load_models(device, "basicvsrpp-v1.2", res.path, None, det.path, fp16, False)
    fa = (det_model, res_model, pad_mode, device)

    starts = [float(s) for s in args.starts.split(",")]
    modes = args.modes.split(",")
    print("warm-up", file=sys.stderr, flush=True)
    run(fa, src, meta, starts[0], "off", args.clip, 4.0)

    results = []
    for rep in range(args.reps):
        for start in starts:
            order = modes if (rep + starts.index(start)) % 2 == 0 else modes[::-1]   # alternate order (thermals)
            for mode in order:
                r = run(fa, src, meta, start, mode, args.clip, args.total)
                r["rep"] = rep
                results.append(r)
                print(json.dumps(r), flush=True)
                if args.out:
                    with open(args.out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")

    keys = ["first_s"] + [f"t{s}s" for s in MARKS_S] + ["clip1_fps", "max_ahead"]
    print(f"\nmedians (clip {args.clip}, {args.det}, {len(starts)} starts x {args.reps} reps)")
    print(f"{'mode':<9}" + "".join(f"{k:>11}" for k in keys))
    for mode in modes:
        rs = [r for r in results if r["mode"] == mode]
        vals = []
        for k in keys:
            xs = [r[k] for r in rs if r.get(k) is not None]
            vals.append(f"{statistics.median(xs):>11}" if xs else f"{'-':>11}")
        print(f"{mode:<9}" + "".join(vals))


if __name__ == "__main__":
    main()
