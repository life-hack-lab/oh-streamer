"""Optional NVIDIA VSR encoder for HLS. No driver control-panel changes.

BGR -> FFmpeg (colour conversion only) -> NVEncC NGX VSR + H.264 -> HLS remux.
NVEncC and its DLLs are an optional portable dependency, not a Python package.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import time

from oh_streamer.paths import checkout_root, upstream_root


def executable():
    if sys.platform != "win32":
        raise RuntimeError("投放 RTX 超解析目前僅支援 Windows NVIDIA RTX")
    configured = os.environ.get("OH_STREAMER_NVENC_C")
    candidates = [configured] if configured else [
        str(checkout_root() / "tools" / "nvencc" / "NVEncC64.exe"),
        str(upstream_root() / "tools" / "nvencc" / "NVEncC64.exe"),   # install layout: <root>/tools beside the code
        str(Path(sys.executable).parent / "tools" / "nvencc" / "NVEncC64.exe"),
        shutil.which("NVEncC64"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("找不到 NVEncC64.exe；請安裝 NVEncC（含 DLL）或設定 OH_STREAMER_NVENC_C")


def output_size(width, height, mode="2x"):
    """Fit within 4K, optionally limiting enlargement to 2x; never downscale."""
    if mode not in ("2x", "4k"):
        raise ValueError(f"Unknown cast VSR size mode: {mode}")
    scale = min(3840 / width, 2160 / height)
    if mode == "2x":
        scale = min(2.0, scale)
    if scale <= 1:
        raise ValueError("影片已達投放超解析的 4K 上限，請關閉投放超解析")
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def encoder_command(exe, width, height, gop, level, output="-"):
    return [exe, "--y4m", "-i", "-", "-o", output, "--output-format", "mpegts",
            "--codec", "h264", "--profile", "high", "--level", level,
            "--output-res", f"{width}x{height}", "--vpp-resize", "algo=ngx-vsr,vsr-quality=1",
            "--preset", "P4", "--qvbr", "20", "--max-bitrate", "20000", "--vbv-bufsize", "40000",
            "--gop-len", str(gop), "--strict-gop", "--no-i-adapt", "--bframes", "0",
            "--repeat-headers", "--lookahead", "0", "--output-buf", "0", "--lowlatency"]


def probe():
    """Actually run VSR, not just an executable/encoder availability check."""
    exe = executable()
    with tempfile.TemporaryDirectory(prefix="oh_vsr_probe_") as tmp:
        output = Path(tmp) / "probe.ts"
        cmd = encoder_command(exe, 640, 360, 30, "4.2", str(output))
        frame = bytes([80]) * (320 * 180) + bytes([128]) * (320 * 180 // 2)
        data = b"YUV4MPEG2 W320 H180 F30:1 Ip A1:1 C420jpeg\n" + (b"FRAME\n" + frame) * 2
        try:
            result = subprocess.run(cmd, input=data, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("NVIDIA VSR 測試逾時，請檢查驅動或改用 NVEncC 9.35") from e
        detail = result.stderr.decode("utf-8", "replace")
        if (result.returncode or not output.exists() or output.stat().st_size == 0
                or "nvsdk-ngx vsr:" not in detail):
            detail = detail[-2000:]
            raise RuntimeError("NVIDIA VSR 測試失敗，請檢查 RTX 驅動與 NVEncC DLL：\n" + detail)
    return exe


class Pipeline:
    """Small Popen-compatible pipeline; owns/reaps every child on failure or seek."""
    def __init__(self, exe, plan, in_w, in_h, mux_cmd, stderr=None):
        self.processes = []
        self.stdin = self.stdout = None
        conversion = ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "bgr24",
                      "-s", f"{in_w}x{in_h}", "-framerate", str(plan.fps), "-i", "pipe:0",
                      "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                      "-f", "yuv4mpegpipe", "pipe:1"]
        commands = [conversion, encoder_command(exe, plan.width, plan.height,
                                                plan.frames_per_segment, plan.level), mux_cmd]
        try:
            previous = None
            for cmd in commands:
                p = subprocess.Popen(cmd, stdin=previous.stdout if previous else subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=stderr, bufsize=0,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.processes.append(p)
                if previous:
                    previous.stdout.close()  # children own this pipe now
                else:
                    self.stdin = p.stdin
                previous = p
            self.stdout = previous.stdout
            self.pid = previous.pid
            self.pids = {p.pid for p in self.processes}
        except BaseException:
            self.kill()
            self.wait(timeout=10)
            for p in self.processes:
                for stream in (p.stdin, p.stdout):
                    if stream:
                        stream.close()
            raise

    @property
    def returncode(self):
        codes = [p.poll() for p in self.processes]
        if any(c is None for c in codes):
            return None
        return next((c for c in codes if c), 0)

    def poll(self):
        return self.returncode

    def kill(self):
        for p in reversed(self.processes):
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        for p in self.processes:
            p.wait(timeout=max(0, deadline - time.monotonic()) if deadline else None)
        return self.returncode
