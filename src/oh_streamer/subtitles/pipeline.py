"""Production ASR: one shared span plan, isolated sequential models, CPU fusion."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import runtime
from .quality import load_quality_config, f7_low_confidence

PRIMARY_ASR = "whisper-ja-1.5B"
SECONDARY_ASR = "anime-whisper"
RAW_RECIPE = "production-shared-spans-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _span_identity(plan):
    return {key: plan.get(key) for key in
            ("plan_key", "settings", "audio_path", "audio_duration_seconds", "speech_intervals", "groups")}


def _raw_job(source, cache_root, plan, plan_path, models, alias, config, *,
             force, gpu_managed, vad_threshold, vad_rescue_chunk_seconds):
    from .asr import ASR_FEATURE_VERSION
    model_path = models / runtime.model_dir_name(alias)
    fingerprint = runtime.local_model_fingerprint(model_path)
    decode = {"no_repeat_ngram_size": 6} if alias == PRIMARY_ASR else {}
    _alias, model_spec = runtime.resolve_model(alias)
    decoding = {**model_spec["generate_kwargs"], **decode}
    key = _digest(dict(recipe=RAW_RECIPE, features=ASR_FEATURE_VERSION, plan=_span_identity(plan),
                       model=alias, fingerprint=fingerprint, decode=decoding))
    folder = cache_root / "raw" / alias / key[:16]
    document = None
    if not force:
        try:
            identity = json.loads((folder / "pipeline-raw.json").read_text("utf-8"))
            candidate = json.loads((folder / "ja.json").read_text("utf-8"))
            if identity.get("key") == key and isinstance(candidate.get("segments"), list):
                document = candidate
        except (OSError, ValueError, AttributeError):
            pass
    job = dict(stage="asr_model", source=os.fspath(source), folder=str(folder), asr=alias,
               model_path=str(model_path), shared_plan=str(plan_path),
               cache_dir=str(cache_root / "cache"), force=bool(force), gpu_managed=bool(gpu_managed),
               vad_threshold=vad_threshold, vad_rescue_chunk_seconds=vad_rescue_chunk_seconds,
               generate_overrides=decode, quality_config=config)
    return key, folder, document, job


WORKER_MODULE = "oh_streamer.subtitles.worker"
SRC_DIR = Path(__file__).resolve().parents[2]       # streamer/src (frozen: inside the bundle, harmless)


def child_env():
    """Environment of a subtitle runtime child (as queue/subtitle_jobs.py's stages): the runtime Python finds the
    oh_streamer package from the checkout, UTF-8 pipes, models strictly offline."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), *filter(None, [env.get("PYTHONPATH")])])
    env.update(PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("OH_STREAMER_SUBTITLE_IDLE_UI_PIDS", None)
    return env


def run_worker_job(job, job_path, log_path, on_phase=None, *, python=None, on_process=None, env_extra=None):
    """Run one subtitle worker stage in the subtitle runtime (environment.subtitle_command: `python -m worker`, or
    the frozen exe's --subtitle-worker route) and wait for it. Progress events go to on_phase; a failure raises
    RuntimeError with the child's error message (or its log tail). The child gets this process's pid and exits
    when we die, so killing the process given to on_process ends the whole chain. env_extra: extra environment
    variables of the child (e.g. CUDA_VISIBLE_DEVICES="" for CPU-only preparation)."""
    from . import environment
    job = dict(job, parent_pid=os.getpid())
    runtime.atomic_write_json(job_path, job)
    error = ""
    with Path(log_path).open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            environment.subtitle_command(python or sys.executable, WORKER_MODULE, "--job", str(job_path)),
            stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
            errors="replace", env={**child_env(), **(env_extra or {})},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as process:
            if callable(on_process):
                on_process(process)
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "subtitle_progress" and callable(on_phase):
                    on_phase(event.get("phase", "asr"), event.get("percent"), event.get("detail", ""))
                elif event.get("type") == "error":
                    error = str(event.get("message", ""))
            returncode = process.wait()
    if returncode:
        try:
            tail = Path(log_path).read_text("utf-8", errors="replace")[-1500:]
        except OSError:
            tail = ""
        detail = error or tail
        raise RuntimeError(detail or f"subtitle worker exited with status {returncode}")


def run_asr_child(job, on_phase=None, *, python=None, on_process=None):
    """Wait for a model process to exit before returning to the coordinator.

    on_process(process): optional, called once right after the model process starts, so an owner (the queue's
    local subtitle executor) can register it and kill it on cancel."""
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    run_worker_job(job, folder / "asr-job.json", folder / "asr-worker.log", on_phase,
                   python=python, on_process=on_process)


def run_production_asr_windows_child(windows, *, workdir, python=None, on_phase=None, on_process=None,
                                     **options):
    """run_production_asr_windows inside one subtitle runtime child (stage asr_windows): VAD prepare, the model
    grandchildren and fusion never touch the calling process (the queue's lada venv has no subtitle packages).
    options: run_production_asr_windows' keyword arguments (JSON values). Returns its documents."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result = workdir / "asr-windows-result.json"
    result.unlink(missing_ok=True)
    job = dict(stage="asr_windows", folder=str(workdir), result=str(result), python=python,
               windows=[{key: os.fspath(value) for key, value in window.items()} for window in windows],
               options={key: os.fspath(value) if isinstance(value, Path) else value
                        for key, value in options.items()})
    run_worker_job(job, workdir / "asr-windows-job.json", workdir / "asr-windows-worker.log", on_phase,
                   python=python, on_process=on_process)
    documents = json.loads(result.read_text("utf-8"))
    if not isinstance(documents, list) or len(documents) != len(windows):
        raise RuntimeError("ASR worker returned no documents for the windows")
    return documents


def run_production_asr_prepare_child(windows, *, workdir, python=None, on_phase=None, on_process=None,
                                     **options):
    """CPU-only half of run_production_asr_windows_child (stage asr_prepare): per-window PCM + VAD span plan are
    written to each window's cache (the real run then reads the cached plans). The child sees no CUDA device, so it
    can run while restoration still owns the GPU. options: run_production_asr_windows' keyword arguments."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    job = dict(stage="asr_prepare", folder=str(workdir), python=python,
               windows=[{key: os.fspath(value) for key, value in window.items()} for window in windows],
               options={key: os.fspath(value) if isinstance(value, Path) else value
                        for key, value in options.items()})
    run_worker_job(job, workdir / "asr-prepare-job.json", workdir / "asr-prepare-worker.log", on_phase,
                   python=python, on_process=on_process, env_extra={"CUDA_VISIBLE_DEVICES": ""})


def run_vram_probe_child(need_mib, *, workdir, wait_s=10.0, poll_s=1.0, python=None, on_phase=None, on_process=None):
    """Free VRAM check before a model load (worker stage vram_probe): a subtitle runtime child (it has torch; the
    queue's lada venv must not import it) waits up to wait_s for torch.cuda.mem_get_info to report need_mib free.
    Returns {"ok", "free_mib", "total_mib", "need_mib", "waited_s", "measured"[, "error"]}. The child sees the GPU
    (unlike asr_prepare), so its own CUDA context is counted as used: the answer is what a model child would get."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result = workdir / "vram-probe-result.json"
    result.unlink(missing_ok=True)
    job = dict(stage="vram_probe", folder=str(workdir), result=str(result), python=python,
               need_mib=float(need_mib), wait_s=float(wait_s), poll_s=float(poll_s))
    run_worker_job(job, workdir / "vram-probe-job.json", workdir / "vram-probe-worker.log", on_phase,
                   python=python, on_process=on_process)
    doc = json.loads(result.read_text("utf-8"))
    if not isinstance(doc, dict) or "ok" not in doc:
        raise RuntimeError("VRAM probe returned no result")
    return doc


def run_production_asr(source, out_dir, *, models_dir, asr_mode="dual", asr_model=PRIMARY_ASR, python=None,
                       force=False, gpu_managed=False, on_phase=None, quality_config=None,
                       vad_threshold=None, vad_rescue_chunk_seconds=60.0, child_runner=None,
                       prepare_plan=None, cache_dir=None):
    """Run A1 then A2 with explicit process boundaries; single runs only A1."""
    if asr_mode not in ("dual", "single"):
        raise ValueError("ASR mode must be dual or single")
    from .asr import prepare_shared_spans

    config = load_quality_config(quality_config)
    folder, models = Path(out_dir), Path(models_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else folder
    folder.mkdir(parents=True, exist_ok=True)
    report = on_phase if callable(on_phase) else lambda *_args: None
    plan_factory = prepare_plan or prepare_shared_spans
    plan = plan_factory(source, cache_root / "shared", cache_dir=cache_root / "cache", quality_config=config,
                        vad_threshold=vad_threshold, vad_rescue_chunk_seconds=vad_rescue_chunk_seconds,
                        on_phase=on_phase)
    plan_path = cache_root / "shared" / "asr-plan.json"
    runtime.atomic_write_json(plan_path, plan)
    # F7/hallucination rules are deliberately absent from raw model identities.
    raw_documents, outcomes = {}, {}
    runner = child_runner or (lambda job, callback: run_asr_child(job, callback, python=python))
    aliases = (PRIMARY_ASR, SECONDARY_ASR) if asr_mode == "dual" else (asr_model,)
    for index, alias in enumerate(aliases):
        def forward(phase, percent, detail):
            if phase == "asr" and isinstance(percent, (int, float)):
                percent = 90 * (index + max(0, min(100, percent)) / 100) / len(aliases)
            elif phase == "prepare_asr":
                percent = None
            report(phase, percent, f"{alias}：{detail}")

        try:
            raw_key, raw_folder, document, job = _raw_job(
                source, cache_root, plan, plan_path, models, alias, config,
                force=force, gpu_managed=gpu_managed, vad_threshold=vad_threshold,
                vad_rescue_chunk_seconds=vad_rescue_chunk_seconds)
            output, marker = raw_folder / "ja.json", raw_folder / "pipeline-raw.json"
            cached = document is not None
            if document is None:
                report("asr", 90 * index / len(aliases), f"辨識模型 {alias}（{index + 1}/{len(aliases)}）")
                runner(job, forward)
                document = json.loads(output.read_text("utf-8"))
                if not isinstance(document.get("segments"), list):
                    raise ValueError(f"{alias} did not produce ASR segments")
                runtime.atomic_write_json(marker, {"key": raw_key})
            raw_documents[alias] = document
            outcomes[alias] = dict(status="cached" if cached else "done", key=raw_key, error=None)
            report("asr", 90 * (index + 1) / len(aliases), f"{alias} 辨識完成")
        except Exception as exc:
            outcomes[alias] = dict(status="failed", key=None, error=str(exc)[-1500:])
            report("asr", None, f"{alias} 辨識失敗：{exc}")
            report("asr", 90 * (index + 1) / len(aliases), f"{alias} 已略過，使用其他可用模型")
    return _finish_asr(plan, plan_path, folder, cache_root, config, asr_mode,
                       raw_documents, outcomes, force=force, report=report)


def run_production_asr_windows(windows, *, models_dir, asr_mode="dual", asr_model=PRIMARY_ASR,
                               python=None, force=False, gpu_managed=False, on_phase=None,
                               quality_config=None, vad_threshold=None, vad_rescue_chunk_seconds=60.0,
                               child_runner=None, prepare_plan=None, on_process=None, prepare_only=False):
    """Recognize independent canonical windows while loading each model once.

    Each window is a dict with source, folder and optional cache_dir. Return
    documents in input order. A failed window returns an error string plus empty
    segments and asr_warnings; it does not discard successful neighboring windows.
    No audio, VAD spans or decoding context crosses
    window boundaries. The child runner receives one multi-job per model.
    """
    if asr_mode not in ("dual", "single"):
        raise ValueError("ASR mode must be dual or single")
    from .asr import prepare_shared_spans
    config = load_quality_config(quality_config)
    report = on_phase if callable(on_phase) else lambda *_args: None
    prepare = prepare_plan or prepare_shared_spans
    runner = child_runner or (lambda job, callback: run_asr_child(job, callback, python=python,
                                                                  on_process=on_process))

    def failed_window(error, outcomes=None):
        detail = str(error)
        return dict(error=detail, segments=[], raw_segments=[], speech_intervals=[],
                    asr_warnings=[detail], asr_models=outcomes or {})

    states = []
    for window in windows:
        source, folder = window["source"], Path(window["folder"])
        cache = Path(window.get("cache_dir") or folder)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            plan = prepare(source, cache / "shared", cache_dir=cache / "cache", quality_config=config,
                           vad_threshold=vad_threshold, vad_rescue_chunk_seconds=vad_rescue_chunk_seconds,
                           on_phase=on_phase)
            plan_path = cache / "shared" / "asr-plan.json"
            runtime.atomic_write_json(plan_path, plan)
            states.append(dict(source=source, folder=folder, cache=cache, plan=plan, plan_path=plan_path,
                               documents={}, outcomes={}))
        except Exception as exc:
            states.append(dict(error_result=failed_window(exc)))
    if prepare_only:          # CPU half only: the span plans are cached for the real run
        return []
    aliases = (PRIMARY_ASR, SECONDARY_ASR) if asr_mode == "dual" else (asr_model,)
    for model_index, alias in enumerate(aliases):
        pending = []
        for state in states:
            if "error_result" in state:
                continue
            try:
                key, raw_folder, document, job = _raw_job(
                    state["source"], state["cache"], state["plan"], state["plan_path"],
                    Path(models_dir), alias, config, force=force, gpu_managed=gpu_managed,
                    vad_threshold=vad_threshold, vad_rescue_chunk_seconds=vad_rescue_chunk_seconds)
                if document is not None:
                    state["documents"][alias] = document
                    state["outcomes"][alias] = dict(status="cached", key=key, error=None)
                else:
                    pending.append((state, key, raw_folder, job))
            except Exception as exc:
                state["outcomes"][alias] = dict(status="failed", key=None, error=str(exc)[-1500:])
        if not pending:
            continue
        # This process exits and releases its model before the next alias starts.
        batch_folder = pending[0][2] / "window-batch"
        batch_job = dict(stage="asr_model_windows", folder=str(batch_folder),
                         jobs=[entry[3] for entry in pending])
        def forward(phase, percent, detail):
            if phase == "asr" and isinstance(percent, (int, float)):
                percent = 90 * (model_index + max(0, min(100, percent)) / 100) / len(aliases)
            elif phase == "prepare_asr":
                percent = None
            report(phase, percent, f"{alias}：{detail}")
        process_error = None
        try:
            runner(batch_job, forward)
        except Exception as exc:
            process_error = str(exc)[-1500:]
        try:
            failures = json.loads((batch_folder / "window-errors.json").read_text("utf-8"))
        except (OSError, ValueError):
            failures = {}
        for state, key, raw_folder, job in pending:
            try:
                error = failures.get(job["folder"])
                if error or process_error:
                    raise RuntimeError(error or process_error)
                document = json.loads((raw_folder / "ja.json").read_text("utf-8"))
                if not isinstance(document.get("segments"), list):
                    raise ValueError(f"{alias} did not produce ASR segments")
                runtime.atomic_write_json(raw_folder / "pipeline-raw.json", {"key": key})
                state["documents"][alias] = document
                state["outcomes"][alias] = dict(status="done", key=key, error=None)
            except Exception as exc:
                state["outcomes"][alias] = dict(status="failed", key=None, error=str(exc)[-1500:])
        report("asr", 90 * (model_index + 1) / len(aliases), f"{alias} 辨識完成")
    results = []
    for state in states:
        if "error_result" in state:
            results.append(state["error_result"])
            continue
        try:
            results.append(_finish_asr(
                state["plan"], state["plan_path"], state["folder"], state["cache"], config,
                asr_mode, state["documents"], state["outcomes"], force=force, report=report))
        except Exception as exc:
            results.append(failed_window(exc, state["outcomes"]))
    return results


def _finish_asr(plan, plan_path, folder, cache_root, config, asr_mode,
                raw_documents, outcomes, *, force, report):
    """CPU-only fusion/postprocessing shared by single and multi-window runs."""
    from .asr import segment_from_record
    from . import fusion
    from .core import clean_segments
    from .postprocess import render_fixed_srt
    span_identity = _span_identity(plan)
    if not raw_documents:
        raise RuntimeError("ASR 模型皆失敗：" + "; ".join(f"{a}: {v['error']}" for a, v in outcomes.items()))

    fusion_key = _digest(dict(recipe=fusion.fusion_identity(config), mode=asr_mode, plan=span_identity,
                              models={alias: item.get("key") for alias, item in outcomes.items()}))
    fusion_path = cache_root / "fusion" / f"{fusion_key}.json"
    fused = None
    fusion_warnings = []
    if not force:
        try:
            cached_fusion = json.loads(fusion_path.read_text("utf-8"))
            if cached_fusion.get("cache_key") == fusion_key:
                fused = cached_fusion["segments"]
                fusion_warnings = cached_fusion.get("warnings", [])
        except (OSError, ValueError, KeyError):
            pass
    warnings = [f"{alias}: {item['error']}" for alias, item in outcomes.items() if item['status'] == "failed"]
    raw_rows = {alias: document.get("raw_segments", document["segments"])
                for alias, document in raw_documents.items()}
    if fused is None:
        if PRIMARY_ASR in raw_documents and SECONDARY_ASR in raw_documents:
            try:
                fused = fusion.fuse_segments(raw_rows[PRIMARY_ASR], raw_rows[SECONDARY_ASR], config=config)
            except Exception as exc:
                fusion_warnings.append(f"融合失敗，使用 {PRIMARY_ASR}：{exc}")
                fused = [dict(row, alt_text=None, asr_agreement=None, selected_model="A1")
                         for row in raw_rows[PRIMARY_ASR]]
        else:
            alias = PRIMARY_ASR if PRIMARY_ASR in raw_documents else next(iter(raw_documents))
            fused = [dict(row, alt_text=None, asr_agreement=None,
                          selected_model="A1" if alias == PRIMARY_ASR else "A2") for row in raw_rows[alias]]
        runtime.atomic_write_json(fusion_path, {"cache_key": fusion_key, "segments": fused, "warnings": fusion_warnings})
    warnings.extend(fusion_warnings)
    rows = []
    for row in fused:
        segment = segment_from_record(row)
        segment = replace(segment, low_confidence=f7_low_confidence(segment, config))
        rows.append({**row, **asdict(segment)})
    intervals = plan.get("speech_intervals", [])
    cleaned = clean_segments([segment_from_record(row) for row in rows], quality_config=config)
    document = dict(schema_version=runtime.SCHEMA_VERSION, cache_key=fusion_key, asr_mode=asr_mode,
                    model="A1+A2" if len(raw_documents) == 2 and not fusion_warnings else
                          PRIMARY_ASR if PRIMARY_ASR in raw_documents else next(iter(raw_documents)),
                    source=plan.get("source"),
                    shared_plan=str(plan_path), segments=rows, raw_segments=rows,
                    speech_intervals=intervals, asr_models=outcomes, asr_warnings=warnings,
                    metrics={"asr_mode": asr_mode, "fusion_version": fusion.FUSION_VERSION,
                             "segment_count": len(rows), "fallback": bool(warnings),
                             "models": {alias: d.get("metrics", {}) for alias, d in raw_documents.items()}})
    runtime.atomic_write_json(folder / "ja.json", document)
    runtime.atomic_write_text(folder / "ja.srt", render_fixed_srt(cleaned, japanese=True))
    for warning in warnings:
        report("asr", None, warning)
    report("asr", 100, "語音辨識與字幕融合完成")
    return document


class MtSessionChild:
    """A long-lived subtitle runtime child (worker stage mt_session) that translates with a local Sakura model:
    llama-server, OpenCC, sentence options and residue filtering all live in that process (the queue's lada venv
    has none of them). Requests go in as JSON lines on stdin, events come back on stdout. Killing `process` (what
    on_process receives) ends the session: translate() then raises RuntimeError at once; the worker's kill-on-close
    job object takes llama-server down with it."""

    def __init__(self, job, workdir, *, python=None, on_phase=None, on_process=None):
        self.job, self.workdir = dict(job), Path(workdir)
        self.python, self.on_phase, self.on_process = python, on_phase, on_process
        self.process = None
        self._events = None
        self._next_id = 0
        self._error = ""
        self._log = None

    def _start(self):
        import queue
        import threading
        from . import environment
        self.workdir.mkdir(parents=True, exist_ok=True)
        job_path = self.workdir / "mt-session-job.json"
        runtime.atomic_write_json(job_path, dict(self.job, stage="mt_session", parent_pid=os.getpid(),
                                                 python=self.python))
        self._log = (self.workdir / "mt-session-worker.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            environment.subtitle_command(self.python or sys.executable, WORKER_MODULE, "--job", str(job_path)),
            stdout=subprocess.PIPE, stderr=self._log, stdin=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", env=child_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._events = queue.Queue()
        threading.Thread(target=self._pump, args=(self.process, self._events), daemon=True).start()
        if callable(self.on_process):
            self.on_process(self.process)

    @staticmethod
    def _pump(process, events):
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    events.put(event)
        except (OSError, ValueError):
            pass
        events.put(None)

    def _tail(self):
        try:
            return (self.workdir / "mt-session-worker.log").read_text("utf-8", errors="replace")[-1500:]
        except OSError:
            return ""

    def translate(self, ja_doc, workdir, *, sentence=None, on_batch=None):
        """Translate one document; returns the translated rows. on_batch(rows): cumulative finalized rows."""
        import queue
        if self.process is None:
            self._start()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self._next_id += 1
        rid = self._next_id
        ja, out = workdir / "ja.json", workdir / "mt-result.json"
        runtime.atomic_write_json(ja, ja_doc)
        out.unlink(missing_ok=True)
        request = {"op": "translate", "id": rid, "ja": str(ja), "out": str(out), "sentence": sentence}
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise RuntimeError(self._error or self._tail() or "subtitle translation process is gone") from exc
        while True:
            try:
                event = self._events.get(timeout=0.2)
            except queue.Empty:
                if self.process.poll() is not None and self._events.empty():
                    event = None
                else:
                    continue
            if event is None:
                self._events.put(None)
                raise RuntimeError(self._error or self._tail()
                                   or f"subtitle translation process exited ({self.process.poll()})")
            kind = event.get("type")
            if kind == "subtitle_progress" and callable(self.on_phase):
                self.on_phase(event.get("phase", "translate"), event.get("percent"), event.get("detail", ""))
            elif kind == "error":
                self._error = str(event.get("message", ""))
            elif event.get("id") != rid:
                continue
            elif kind == "mt_rows" and callable(on_batch):
                try:
                    on_batch(event.get("rows") or [])
                except BaseException:
                    self.close(kill=True)          # cancelled / failed mid-translation: do not leave it running
                    raise
            elif kind == "mt_error":
                raise RuntimeError(str(event.get("message", "")) or "translation failed")
            elif kind == "mt_done":
                return list(json.loads(out.read_text("utf-8")).get("segments") or [])

    def close(self, *, kill=False):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if not kill and process.poll() is None:
                try:
                    process.stdin.write('{"op": "close"}\n')
                    process.stdin.flush()
                    process.wait(timeout=15)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    kill = True
            if kill and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                process.kill()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            for stream in (process.stdin, process.stdout, self._log):
                try:
                    stream.close()
                except (OSError, ValueError, AttributeError):
                    pass
