"""One isolated subtitle stage per process, communicating only through JSON/files."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import runtime

# The installed app starts this worker without a console it could share with its ffmpeg/ffprobe children.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def report_phase(phase, percent, detail):
    emit(type="subtitle_progress", phase=phase, percent=percent, detail=detail)


def emit_subtitle_partial(covered_s, phase, srt, refined_count, total_count, **extra):
    emit(
        type="subtitle_partial",
        covered_s=covered_s,
        phase=phase,
        srt=srt,
        refined_count=refined_count,
        total_count=total_count,
        **extra,
    )


def run_mux_command(command, duration, start_time):
    """Report mux media time, reserving the final 10% for output validation."""
    command = [*command[:1], "-progress", "pipe:1", "-nostats", *command[1:]]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8", creationflags=NO_WINDOW) as proc:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and duration > 0:
                try:
                    elapsed = float(value) / 1_000_000 - start_time
                except ValueError:
                    continue
                report_phase("mux", min(90, max(0, 90 * elapsed / duration)), "封裝影音與字幕")
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, command)


def remux_command(source, output, tracks, offset=0.0, existing_subtitles=()):
    command = [
        runtime.resolve_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-copyts", "-i", str(source),
    ]
    for path, _, _ in tracks:
        command += ["-itsoffset", f"{offset:.6f}", "-i", str(path)]

    command += ["-map", "0"]
    # Replace our AI tracks on rerun, preserving all other streams and metadata.
    for index in existing_subtitles:
        command += ["-map", f"-0:{index}"]
    for index in range(len(tracks)):
        command += ["-map", f"{index + 1}:0"]

    command += ["-map_metadata", "0", "-c", "copy", "-avoid_negative_ts", "disabled"]
    return command


def mux(job):
    source, output = job["video"], job["output"]
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", source,
    ], encoding="utf-8", creationflags=NO_WINDOW))

    # PCM extraction starts at the first decoded audio sample. Anchor that zero
    # to the finished output's audio clock, including HLS copyts/nonzero starts.
    job = dict(job)
    if "audio_offset" not in job:
        audio = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), {})
        job["audio_offset"] = float(audio.get("start_time") or 0.0)

    # Optional "replace_titles" (subtitle combos) extends the AI tracks replaced on rerun.
    from .tracks import CONCISE_TITLE, FULL_TITLE
    replace_titles = {"繁體中文（AI）", CONCISE_TITLE, FULL_TITLE, "日文（AI）", *job.get("replace_titles", ())}
    old = [
        stream["index"]
        for stream in probe["streams"]
        if stream["codec_type"] == "subtitle"
        and stream.get("tags", {}).get("title") in replace_titles
    ]
    remaining = [
        stream
        for stream in probe["streams"]
        if stream["codec_type"] == "subtitle" and stream["index"] not in old
    ]
    command = remux_command(source, output, job["tracks"], job.get("audio_offset", 0.0), old)

    # Optional per-track "dispositions" (booleans); default keeps 1a: every chi track default.
    dispositions = job.get("dispositions")
    for track_number, (index, (_, language, title)) in enumerate(
        enumerate(job["tracks"], len(remaining))
    ):
        if dispositions is not None:
            disposition = "default" if dispositions[track_number] else "0"
        else:
            disposition = "default" if language == "chi" and title != FULL_TITLE else "0"
        command += [
            f"-metadata:s:s:{index}", f"language={language}",
            f"-metadata:s:s:{index}", f"title={title}",
            f"-disposition:s:{index}", disposition,
        ]
    command += ["-f", "matroska", output]

    start_time = float(probe["format"].get("start_time") or 0)
    duration = max(0, float(probe["format"].get("duration") or 0) - max(0, start_time))
    report_phase("mux", 0, "封裝影音與字幕")
    run_mux_command(command, duration, start_time)
    report_phase("mux", 90, "驗證影音軌與片長")
    after = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", output,
    ], encoding="utf-8", creationflags=NO_WINDOW))

    original_av = [
        (stream["codec_type"], stream["codec_name"])
        for stream in probe["streams"]
        if stream["codec_type"] in ("video", "audio")
    ]
    output_av = [
        (stream["codec_type"], stream["codec_name"])
        for stream in after["streams"]
        if stream["codec_type"] in ("video", "audio")
    ]
    if original_av != output_av:
        raise RuntimeError("字幕封裝改變了影音軌")
    if abs(float(probe["format"]["duration"]) - float(after["format"]["duration"])) > 0.1:
        raise RuntimeError("字幕封裝後影片長度不符")

    from .core import parse_srt

    report_phase("mux", 95, "驗證字幕文字與時間碼")
    for index, (path, _, _) in enumerate(job["tracks"], len(remaining)):
        actual = parse_srt(subprocess.check_output([
            runtime.resolve_ffmpeg(), "-nostdin", "-v", "error", "-copyts", "-i", output,
            "-map", f"0:s:{index}", "-f", "srt", "pipe:1",
        ], encoding="utf-8", creationflags=NO_WINDOW))
        expected = parse_srt(Path(path).read_text("utf-8"))
        if len(actual) != len(expected) or any(
            actual_cue.text != expected_cue.text
            or abs(actual_cue.start - (expected_cue.start + job.get("audio_offset", 0.0))) > 0.002
            or abs(actual_cue.end - (expected_cue.end + job.get("audio_offset", 0.0))) > 0.002
            for actual_cue, expected_cue in zip(actual, expected)
        ):
            raise RuntimeError("字幕封裝後文字或時間碼不符")

    report_phase("mux", 100, "驗證完成")
    emit(type="done", output=output)


def run_asr_model_windows(job, on_phase=None):
    """One process owns one model; every window keeps its own ASR state."""
    from .asr import ASRSession, run_asr
    jobs = job["jobs"]
    if jobs:
        identities = {(item["asr"], item["model_path"],
                       json.dumps(item.get("generate_overrides"), sort_keys=True)) for item in jobs}
        if len(identities) != 1:
            raise ValueError("ASR window batch must use one model and decoding identity")
    errors = {}
    error_path = Path(job["folder"]) / "window-errors.json"
    # Reset a prior attempt's error status before touching any window output.
    runtime.atomic_write_json(error_path, errors)
    with ASRSession() as session:
        for index, item in enumerate(jobs):
            def forward(phase, percent, detail):
                if callable(on_phase):
                    if phase == "asr" and isinstance(percent, (int, float)):
                        percent = 100 * (index + max(0, min(100, percent)) / 100) / len(jobs)
                    on_phase(phase, percent, f"視窗 {index + 1}/{len(jobs)}：{detail}")
            failed = False
            try:
                run_asr(
                    item["source"], item["folder"], model=item["asr"], model_path=item["model_path"],
                    cache_dir=item.get("cache_dir") or Path(item["folder"]) / "cache",
                    segmentation="utterance", resume_partial=True,
                    gpu_managed=bool(item.get("gpu_managed")), force=bool(item.get("force")),
                    shared_plan=item.get("shared_plan"), quality_config=item.get("quality_config"),
                    vad_threshold=item.get("vad_threshold"),
                    vad_rescue_chunk_seconds=item.get("vad_rescue_chunk_seconds", 60.0),
                    generate_overrides=item.get("generate_overrides"), on_phase=forward, session=session,
                )
            except Exception as exc:
                errors[item["folder"]] = str(exc)[-1500:]
                failed = True
            if failed:
                # A failed job cannot leave cached decoder state for its neighbor.
                # Leave except first: its traceback can retain model/tensor references, which
                # would prevent close() from collecting them before the next window reload.
                session.close()
            runtime.atomic_write_json(error_path, errors)
    return errors


EXIT_HOOKS = []        # callables run (best effort) before watch_parent's os._exit: e.g. stop a llama-server


def _run_exit_hooks():
    for hook in list(EXIT_HOOKS):
        try:
            hook()
        except Exception:
            pass


def kill_children_with_me():
    """Windows: put this process in a kill-on-close job object, so every process it starts afterwards (llama-server)
    dies with it - however it ends (os._exit of watch_parent, a plain kill). Best effort, stdlib ctypes only."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount",
            "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x2000                  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        return False
    return bool(kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()))    # handle stays open on purpose


def watch_parent(pid):
    """Exit when the process that started us dies (its stdout pipe would go nowhere): a cancel that kills only
    the direct child still ends the model grandchildren. Stdlib only (the frozen exe has no psutil)."""
    import threading
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if pid <= 0:
        return
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.OpenProcess(0x00100000, False, pid)          # SYNCHRONIZE
        if not handle:
            return

        def wait():
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            _run_exit_hooks()
            os._exit(1)
    else:
        if os.getppid() != pid:
            return

        def wait():
            import time
            while os.getppid() == pid:
                time.sleep(0.5)
            _run_exit_hooks()
            os._exit(1)
    threading.Thread(target=wait, name="watch-parent", daemon=True).start()


def run_mt_session(job):
    """Local Sakura translation inside the subtitle runtime (stage mt_session): one llama-server for the whole
    session, OpenCC / residue filtering / sentence options as translate_file does them. Requests arrive as JSON
    lines on stdin ({"op": "translate", "id", "ja", "out", "sentence"} / {"op": "close"}); each answers with
    mt_rows (cumulative batches), then mt_done, or mt_error. stdin EOF (the queue died) ends the session."""
    from .cli import translate_file
    from .glossary import DEFAULT_TRANSLATE_MODEL
    from .llm import LlamaServerClient
    from .tracks import deterministic_sampling
    translation = job.get("translation") or {}
    glossary = job.get("glossary") or {}
    model_path, server, profile = job["model_path"], job.get("server"), job["profile"]
    sampling = deterministic_sampling({"seed": translation.get("seed")})
    log = Path(job["server_log"])
    holder = {}
    kill_children_with_me()

    def stop_server():
        process = getattr(holder.get("client"), "_process", None)
        if process is not None and process.poll() is None:
            process.kill()
    EXIT_HOOKS.append(stop_server)

    def client():
        if "client" not in holder:
            options = {"server_log_path": log}
            if profile != DEFAULT_TRANSLATE_MODEL:
                options["translate_model"] = profile
            if translation.get("gpu_layers", "auto") != "auto":
                options["gpu_layers"] = translation["gpu_layers"]
            if translation.get("context_size", 8192) != 8192:
                options["context_size"] = translation["context_size"]
            options.update(sampling)
            log.parent.mkdir(parents=True, exist_ok=True)
            holder["client"] = LlamaServerClient("sakura", model_path, server, **options)
        return holder["client"]

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            if request.get("op") == "close":
                break
            rid = request["id"]
            try:
                sentence = request.get("sentence")
                if sentence is None:
                    sentence = job.get("sentence") or {}
                ja = Path(request["ja"])
                result = translate_file(
                    ja, ja.parent, model="sakura", model_path=model_path, llama_server=server,
                    force=True, gpu_managed=True, on_phase=report_phase, translate_model=profile,
                    sentence_mode=sentence.get("sentence_mode", True),
                    show_interjections=sentence.get("show_interjections", False),
                    short_threshold=sentence.get("short_threshold", 1.0),
                    short_policy=sentence.get("short_policy", "whitelist"),
                    generic_glossary_only=glossary.get("generic_only", True), glossary=glossary.get("path"),
                    batch_size=translation.get("batch_size", 30),
                    prompt_mode=translation.get("prompt_mode", "official-numbered"),
                    context_size=translation.get("context_size", 8192),
                    gpu_layers=translation.get("gpu_layers", "auto"),
                    sampling=sampling, shared_client=client(),
                    on_batch=lambda rows, rid=rid: emit(type="mt_rows", id=rid, rows=rows))
                runtime.atomic_write_json(Path(request["out"]), {"segments": list(result.get("segments") or [])})
                emit(type="mt_done", id=rid)
            except Exception as exc:
                emit(type="mt_error", id=rid, message=str(exc))
    finally:
        if "client" in holder:
            holder["client"].close()


def run_vram_probe(job, mem_get_info=None, sleep=None, clock=None):
    """Stage vram_probe: wait up to wait_s (re-measuring every poll_s) until torch.cuda.mem_get_info reports at least
    need_mib free, and write {"ok", "free_mib", "total_mib", "need_mib", "waited_s", "measured"} to job["result"].
    No CUDA / torch error: ok=True, measured=False (the probe never blocks work it cannot judge). The seams
    (mem_get_info, sleep, clock) are for tests."""
    import time
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    need, wait_s, poll_s = float(job["need_mib"]), max(0.0, float(job.get("wait_s", 0.0))), max(0.05, float(job.get("poll_s", 1.0)))
    t0 = clock()
    result = {"ok": True, "free_mib": None, "total_mib": None, "need_mib": need, "waited_s": 0.0, "measured": False}
    try:
        if mem_get_info is None:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            mem_get_info = torch.cuda.mem_get_info
        while True:
            free, total = mem_get_info()
            result.update(free_mib=round(free / 2**20, 1), total_mib=round(total / 2**20, 1), measured=True,
                          ok=free / 2**20 >= need, waited_s=round(clock() - t0, 2))
            if result["ok"] or clock() - t0 >= wait_s:
                break
            sleep(min(poll_s, max(0.0, wait_s - (clock() - t0))))
    except Exception as exc:  # noqa: BLE001 - cannot judge: do not block the turn
        result.update(ok=True, measured=False, error=str(exc)[-200:], waited_s=round(clock() - t0, 2))
    runtime.atomic_write_json(Path(job["result"]), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text("utf-8"))
    os.environ.pop("OH_STREAMER_SUBTITLE_IDLE_UI_PIDS", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if job.get("parent_pid"):
        watch_parent(job["parent_pid"])
    stage = job["stage"]
    phase = {
        "asr": "prepare_asr", "asr_model": "prepare_asr", "asr_model_windows": "prepare_asr", "asr_windows": "prepare_asr", "asr_prepare": "prepare_asr", "vram_probe": "prepare_asr", "mt_session": "prepare_translate", "translate": "prepare_translate", "mux": "mux",
    }[stage]
    report_phase(phase, None, "啟動字幕程序")

    if stage == "asr":
        from .pipeline import run_production_asr

        run_production_asr(
            job["source"], job["folder"], models_dir=job.get("models") or Path(job["model_path"]).parent,
            asr_mode=job.get("asr_mode", "dual"), asr_model=job.get("asr", "whisper-ja-1.5B"),
            python=job.get("python") or sys.executable,
            gpu_managed=True, force=bool(job.get("force")), on_phase=report_phase,
        )
    elif stage == "asr_windows":
        # The queue's local slice executor: VAD prepare + one model child per alias + fusion, all in this runtime.
        from .pipeline import run_production_asr_windows

        documents = run_production_asr_windows(
            job["windows"], python=job.get("python") or sys.executable, on_phase=report_phase,
            **job.get("options", {}))
        runtime.atomic_write_json(Path(job["result"]), documents)
    elif stage == "asr_prepare":
        # CPU-only preparation of the turn's windows (the queue hides CUDA from this child): PCM + VAD span plans.
        from .pipeline import run_production_asr_windows

        run_production_asr_windows(
            job["windows"], python=job.get("python") or sys.executable, on_phase=report_phase,
            prepare_only=True, **job.get("options", {}))
    elif stage == "mt_session":
        run_mt_session(job)
    elif stage == "vram_probe":
        run_vram_probe(job)
        emit(type="done")
        return
    elif stage == "asr_model_windows":
        run_asr_model_windows(job, report_phase)
    elif stage == "asr_model":
        from .asr import run_asr

        run_asr(
            job["source"], job["folder"], model=job["asr"], model_path=job["model_path"],
            cache_dir=job.get("cache_dir") or Path(job["folder"]) / "cache", segmentation="utterance", resume_partial=True,
            gpu_managed=bool(job.get("gpu_managed")), force=bool(job.get("force")),
            shared_plan=job.get("shared_plan"), quality_config=job.get("quality_config"),
            vad_threshold=job.get("vad_threshold"),
            vad_rescue_chunk_seconds=job.get("vad_rescue_chunk_seconds", 60.0),
            generate_overrides=job.get("generate_overrides"),
            on_phase=report_phase,
        )
    elif stage == "translate":
        from .cli import translate_file

        model_name = Path(job["model_path"]).name.lower()
        profile = ("sakura-galtransl-7b-v3.7" if "galtransl-7b-v3.7" in model_name else
                   "sakura-galtransl-14b-v3.8" if "galtransl-14b-v3.8" in model_name else "sakura-v1")
        translate_file(
            Path(job["folder"]) / "ja.json", job["folder"], model="sakura", model_path=job["model_path"],
            llama_server=job["server"], sentence_mode=True, show_interjections=job["show_interjections"],
            generic_glossary_only=True, force=bool(job.get("force")), gpu_managed=True,
            on_phase=report_phase,
            translate_model=profile,
        )
    elif stage == "mux":
        mux(job)
        return
    else:
        raise ValueError("Unknown subtitle stage")

    report_phase("asr" if stage in ("asr_model", "asr_model_windows", "asr_windows", "asr_prepare") else stage, 100, "階段完成")
    emit(type="done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit(type="error", message=str(exc))
        sys.exit(1)
