"""Repeatable local acceptance checks; GUI/device-specific validation stays explicit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:   # lets the script import oh_streamer without the removed root shims
    sys.path.insert(0, _SRC)

from oh_streamer.processing import encoding_backend as E
from oh_streamer.paths import resource_root
from portability_common import (SCHEMA_VERSION, child_environment, environment, last_json,
                                positive_seconds, project_root, run_command, write_json)

HERE = Path(__file__).resolve().parent
CHECKS = ("diagnostic", "fixture", "encode", "segments", "ipc", "restore", "queue", "benchmark", "rife")
DEFAULT_CHECKS = CHECKS[:5]


def load_retry(path):
    previous = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(previous, dict) or type(previous.get("schema_version")) is not int or previous.get("schema_version") != SCHEMA_VERSION or previous.get("kind") != "portability-acceptance":
        raise ValueError("unsupported report schema")
    if not isinstance(previous.get("environment"), dict) or not isinstance(previous.get("options"), dict) or not isinstance(previous.get("checks"), list):
        raise ValueError("report environment/options/checks have invalid types")
    if any(previous["environment"].get(k) != environment()[k] for k in ("os", "architecture")):
        raise ValueError("different OS/architecture; run all checks on the new machine")
    options = previous["options"]
    for key in ("device", "encoder", "project_root", "input_path"):
        if key in options and options[key] is not None and not isinstance(options[key], str):
            raise ValueError(f"invalid retry option {key}")
    for key in ("require_mps", "require_videotoolbox"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"invalid retry option {key}")
    seen = set()
    for check in previous["checks"]:
        if not isinstance(check, dict) or check.get("id") not in CHECKS or check.get("status") not in ("pass", "fail", "skip") or check["id"] in seen:
            raise ValueError("invalid or duplicate retry check")
        seen.add(check["id"])
    return previous


class Acceptance:
    def __init__(self, output_dir, *, root=None, device="auto", encoder="auto", input_path=None,
                 timeout=120, require_mps=False, require_videotoolbox=False):
        self.directory = Path(output_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.root = project_root(root)
        self.device, self.encoder = device, encoder
        self.input_path = Path(input_path).resolve() if input_path else None
        self.timeout = timeout
        self.require_mps, self.require_videotoolbox = require_mps, require_videotoolbox
        self.fixture = self.directory / "fixture.mp4"
        self.env = child_environment(self.root)
        self.env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE / "src"), self.env.get("PYTHONPATH")]))
        self.selected_encoder = None

    def command(self, identifier, argv, *, timeout=None):
        result = run_command(argv, timeout=timeout or self.timeout, cwd=HERE, env=self.env,
                             log_path=self.directory / "logs" / (identifier + ".log"))
        if result["timed_out"] or result["returncode"] != 0:
            reason = "timed out" if result["timed_out"] else f"exit {result['returncode']}"
            raise RuntimeError(f"{identifier}: {reason}; see logs/{identifier}.log")
        return result

    def probe(self, path, identifier="probe", *, packets=False):
        args = ["ffprobe", "-v", "error", "-of", "json"]
        if packets:
            args += ["-select_streams", "v:0", "-read_intervals", "%+#1", "-show_packets"]
        else:
            args += ["-show_streams", "-show_format"]
        return json.loads(self.command(identifier, args + [str(path)])["stdout"])

    def ensure_fixture(self):
        if not self.fixture.is_file():
            from portability_fixtures import generate_fixture
            generate_fixture(self.fixture)

    def codec(self):
        if self.selected_encoder is None:
            requested = "h264_videotoolbox" if self.require_videotoolbox else self.encoder
            if self.require_videotoolbox and not E.encoder_usable(requested, timeout=self.timeout):
                raise RuntimeError("required VideoToolbox encoder unavailable; fallback is not accepted")
            self.selected_encoder = E.resolve_encoder(requested)
        return self.selected_encoder

    def check_diagnostic(self):
        import portability_doctor
        report = portability_doctor.build_report(root=self.root, device=self.device, probe_encoders=True,
            require_mps=self.require_mps, require_videotoolbox=self.require_videotoolbox, timeout=min(self.timeout, 30))
        write_json(self.directory / "diagnostic.json", report)
        if report["status"] != "pass":
            failed = [row["id"] for row in report["checks"] if row["status"] == "fail" and row["required"]]
            raise RuntimeError("diagnostic failed: " + ", ".join(failed))
        return {"report": "diagnostic.json"}

    def check_fixture(self):
        self.ensure_fixture()
        data = self.probe(self.fixture, "fixture-probe")
        video = next(s for s in data["streams"] if s["codec_type"] == "video")
        audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
        if abs(float(data["format"]["duration"]) - 4) > .2:
            raise RuntimeError("fixture duration differs from expected 4 seconds")
        return {"video": video["codec_name"], "audio": audio["codec_name"], "duration_s": float(data["format"]["duration"]),
                "purpose": "synthetic transport/audio; does not prove real mosaic restoration"}

    def check_encode(self):
        self.ensure_fixture()
        encoder = self.codec()
        output = self.directory / "encoded.mkv"
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(self.fixture),
                "-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac"]
        args += E.encoder_args(encoder) + [str(output)]
        self.command("encode", args)
        data = self.probe(output, "encode-probe")
        if not any(s.get("codec_type") == "audio" for s in data["streams"]):
            raise RuntimeError("encoded output lost audio")
        duration = float(data["format"]["duration"])
        if abs(duration - 4) > .2:
            raise RuntimeError("encoded output duration is wrong")
        return {"encoder": encoder, "duration_s": duration}

    def check_segments(self):
        self.ensure_fixture()
        encoder = self.codec()
        # Use a fresh directory on every retry, avoiding stale last segments.
        import tempfile
        folder = Path(tempfile.mkdtemp(prefix="segments-", dir=self.directory))
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(self.fixture),
                "-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac"]
        args += E.encoder_args(encoder, purpose="segmented") + E.quality_args(encoder) + E.gop_args(encoder, 30)
        args += ["-f", "segment", "-segment_format", "matroska", "-segment_time", "1", "-reset_timestamps", "1",
                 str(folder / "segment_%03d.mkv")]
        self.command("segments", args)
        files = sorted(folder.glob("segment_*.mkv"))
        if len(files) != 4:
            raise RuntimeError(f"expected four one-second segments, found {len(files)}")
        for index, path in enumerate(files):
            packet = self.probe(path, f"segment-{index}-packet", packets=True).get("packets", [])
            if not packet or "K" not in packet[0].get("flags", ""):
                raise RuntimeError(f"segment {index} does not start with a video keyframe")
        concat = folder / "concat.txt"
        concat.write_text("".join(f"file '{p.name}'\n" for p in files), encoding="utf-8")
        output = folder / "merged.mkv"
        self.command("merge", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat",
                               "-safe", "0", "-i", str(concat), "-c", "copy", str(output)])
        data = self.probe(output, "merge-probe")
        duration = float(data["format"]["duration"])
        if abs(duration - 4) > .3 or not any(s.get("codec_type") == "audio" for s in data["streams"]):
            raise RuntimeError("segment merge lost duration or audio")
        return {"encoder": encoder, "segments": len(files), "duration_s": duration,
                "note": "keyframe and duration checks; perceptual A/V sync remains a manual check"}

    def check_ipc(self):
        from oh_streamer.player.mpv_ipc import MpvEndpoint, MpvIpc
        self.ensure_fixture()
        mpv = shutil.which("mpv")
        if not mpv and sys.platform == "win32":
            candidate = Path(r"C:\Program Files\MPV Player\mpv.exe")
            mpv = str(candidate) if candidate.is_file() else None
        if not mpv:
            raise RuntimeError("mpv not found")
        endpoint = MpvEndpoint()
        process = ipc = None
        log = self.directory / "logs" / "mpv.log"
        log.parent.mkdir(exist_ok=True)
        try:
            kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {"start_new_session": True}
            process = subprocess.Popen([mpv, "--no-config", "--no-terminal", "--vo=null", "--ao=null", "--idle=yes",
                "--force-window=no", "--pause=yes", "--hwdec=no", f"--input-ipc-server={endpoint.path}",
                f"--script={resource_root() / 'oh_seek.lua'}", f"--script-opts=oh-platform={sys.platform},oh-vsr=off",
                f"--log-file={log}"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
            ipc = MpvIpc(endpoint.path, process, timeout=min(self.timeout, 15))
            # Socket reads need a deadline too; Windows pipe is bounded by the watchdog.
            import threading
            watchdog = threading.Timer(self.timeout, lambda: process.kill() if process.poll() is None else None)
            watchdog.daemon = True
            watchdog.start()
            try:
                version = ipc.get("mpv-version")
                ipc.set("user-data/oh/processed", "1")
                ipc.set("user-data/oh/expect-load", str(self.fixture))
                ipc.command("loadfile", str(self.fixture), "replace")
                deadline = time.monotonic() + min(self.timeout, 10)
                duration = None
                while time.monotonic() < deadline:
                    duration = ipc.get("duration")
                    if duration:
                        break
                    time.sleep(.05)
                if not duration:
                    raise RuntimeError("mpv did not load the fixture")
                ipc.command("seek", 2, "absolute+exact")
                deadline = time.monotonic() + min(self.timeout, 5)
                position = None
                while time.monotonic() < deadline:
                    position = ipc.get("time-pos")
                    if position is not None and abs(position - 2) < .2:
                        break
                    time.sleep(.05)
                if position is None or abs(position - 2) >= .2:
                    raise RuntimeError(f"mpv seek did not reach two seconds: {position}")
                return {"mpv": version, "duration_s": duration, "seek_position_s": position,
                        "transport": "named-pipe" if sys.platform == "win32" else "unix-socket", "gui_tested": False}
            finally:
                watchdog.cancel()
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if ipc is not None:
                ipc.close()
            endpoint.close()

    def source(self):
        self.ensure_fixture()
        source = self.input_path or self.fixture
        if not source.is_file():
            raise RuntimeError("model-test source must be an existing local file")
        return source

    def check_restore(self):
        output = self.directory / "restored.mkv"
        self.command("restore", [sys.executable, HERE / "oh_player.py", self.source(), "--device", self.device,
            "--encoder", self.codec(), "--output", output, "--duration", "2", "--max-clip-length", "30",
            "--interpolation", "off", "--vsr", "off", "--settings-file", self.directory / "isolated-settings.json"])
        data = self.probe(output, "restore-probe")
        duration = float(data["format"]["duration"])
        if duration <= 0:
            raise RuntimeError("restoration produced an empty output")
        return {"duration_s": duration, "encoder": self.codec(), "source": self.source().name,
                "synthetic_source": self.input_path is None}

    def check_queue(self):
        folder = self.directory / "queue"
        folder.mkdir(exist_ok=True)
        job = {"source": str(self.source()), "output": str(folder / "output.mkv"), "work_dir": str(folder / "work"),
               "segment_seconds": 1, "settings": {"device": self.device, "encoder": self.codec(),
               "interpolation": "off", "max_clip_length": 30, "fp16": None}}
        write_json(folder / "job.json", job)
        result = self.command("queue", [sys.executable, "-m", "oh_streamer.queue.queue_worker", "--job", folder / "job.json"])
        final = last_json(result["stdout"])
        if final.get("type") != "done" or not (folder / "output.mkv").is_file():
            raise RuntimeError("queue worker did not report a finished output")
        duration = float(self.probe(folder / "output.mkv", "queue-probe")["format"]["duration"])
        if duration <= 0:
            raise RuntimeError("queue output is empty")
        return {"duration_s": duration, "encoder": self.codec(), "synthetic_source": self.input_path is None}

    def check_benchmark(self):
        output = self.directory / "benchmark.json"
        self.command("benchmark", [sys.executable, HERE / "benchmarks" / "bench_portability.py", self.source(), "--device", self.device,
            "--models", "v4-fast", "--clip-lengths", "30", "--duration", "2", "--output", output,
            "--timeout", str(self.timeout)], timeout=self.timeout + 15)
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("status") != "pass":
            raise RuntimeError("benchmark configurations did not all pass")
        return {"report": "benchmark.json", "status": report.get("status"), "synthetic_source": self.input_path is None}

    def check_rife(self):
        result = self.command("rife", [sys.executable, HERE / "probe_rife.py", "--device", self.device,
            "--size", "64x64", "--iterations", "1", "--json", "--output", self.directory / "rife.json"])
        report = last_json(result["stdout"])
        if report.get("status") != "pass":
            raise RuntimeError("RIFE standalone probe did not pass")
        return {"report": "rife.json", "mac_player_rife_enabled": False}

    def run(self, selected):
        checks = []
        for identifier in selected:
            start = time.perf_counter()
            try:
                details = getattr(self, "check_" + identifier)()
                row = {"id": identifier, "status": "pass", "details": details}
            except Exception as error:
                row = {"id": identifier, "status": "fail", "message": str(error)}
            row["elapsed_s"] = round(time.perf_counter() - start, 4)
            checks.append(row)
            print(f"{row['status'].upper():4} {identifier}: {row.get('message', 'completed')}", flush=True)
        return {"schema_version": SCHEMA_VERSION, "kind": "portability-acceptance", "environment": environment(),
                "status": "fail" if any(r["status"] == "fail" for r in checks) else "pass", "checks": checks,
                "options": {"device": self.device, "encoder": self.encoder, "project_root": str(self.root),
                            "input_path": str(self.input_path) if self.input_path else None,
                            "require_mps": self.require_mps, "require_videotoolbox": self.require_videotoolbox},
                "manual_pending": ["Cocoa/Finder/tray", "perceptual A/V sync", "real mosaic workload",
                                   "sleep assertion release", "Apple TV", "macOS app/signing"],
                "note": "pass applies only to checks listed here; no GUI or full Mac certification implied"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--device")
    parser.add_argument("--encoder")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--checks", help=",".join(CHECKS))
    parser.add_argument("--with-models", action="store_true")
    parser.add_argument("--retry-failed", metavar="REPORT")
    parser.add_argument("--timeout", type=positive_seconds, default=120)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--require-videotoolbox", action="store_true")
    args = parser.parse_args(argv)
    previous, options = None, {}
    if args.retry_failed:
        try:
            previous = load_retry(args.retry_failed)
            options = previous["options"]
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"cannot retry report: {error}")
    selected = args.checks.split(",") if args.checks else [r["id"] for r in previous["checks"] if r["status"] == "fail"] if previous else list(DEFAULT_CHECKS)
    if args.with_models:
        selected += [name for name in ("restore", "queue", "benchmark") if name not in selected]
    if any(name not in CHECKS for name in selected):
        parser.error("unknown check name")
    if (args.require_mps or args.require_videotoolbox or options.get("require_mps") or options.get("require_videotoolbox")) and "diagnostic" not in selected:
        selected.insert(0, "diagnostic")
    if not selected:
        print("No failed checks to retry; nothing executed")
        return 0
    runner = Acceptance(args.output_dir, root=args.project_root or options.get("project_root"),
        device=args.device or options.get("device", "auto"), encoder=args.encoder or options.get("encoder", "auto"),
        input_path=args.input_path or options.get("input_path"), timeout=args.timeout,
        require_mps=args.require_mps or options.get("require_mps", False),
        require_videotoolbox=args.require_videotoolbox or options.get("require_videotoolbox", False))
    report = runner.run(tuple(dict.fromkeys(selected)))
    if previous:
        report["retry_of"] = str(Path(args.retry_failed).resolve())
        report["not_retested"] = [r["id"] for r in previous["checks"] if r["id"] not in selected]
    write_json(runner.directory / "acceptance.json", report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
