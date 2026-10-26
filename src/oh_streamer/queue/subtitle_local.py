"""Unified subtitle scheduler, local GPU executor.

NOT wired into the queue yet (D3: M7). Like the cloud executor (subtitle_scheduler.CloudSliceExecutor, whose plan,
cache, reconcile and publication code it reuses) it depends on no queue service object: the GPU worker / lease /
restoration / playback side is the `GpuHost` interface, the models are the injectable `LocalDeps`.

- Only local candidates: the executor is ``sp.Executor("local")`` and takes a unit only while the current
  fallback chain puts it on a local candidate (fallback.owner_function, M4); it shares the folder's PlanStore
  with the cloud executor. GPU busy is not a failure and never spills to the cloud (D1): the work stays pending.
  Local errors are retried (CandidateChain.decide, RETRY_WAIT_S * attempt), then the next candidate (e.g. the
  cloud backup) or failed(exhausted).
- ASR: run_production_asr (single / dual from the subtitle config) on the slice audio with the shared pre-roll and
  cue ownership (slice_plan.AsrWindow / owns_cue). MT: the Sakura adapter = translate_file with exactly the
  arguments blocks.run_block passes (official prompt, glossary, sentence splitting, seed), design 4.6.
  Results go through the same content-addressed cache and publication as the cloud executor.
- Batch: every GPU turn does N slices (first subtitle / urgent: 1); inside a batch all ASR first. Each model
  loads once and processes independent per-slice VAD windows, then all MT shares one llama-server.
- Resource mode: auto (default) / interleave / parallel (gpu_turns.choose_resource_mode). Interleave asks
  gpu_turns.decide_gpu_turn before every batch and stops / resumes restoration around its turns. Parallel holds
  a shared lease and is watched by gpu_turns.parallel_guard (CUDA OOM, restoration falling behind playback):
  a trip demotes to interleave for the rest of the run and, in auto, records it for this card (GpuModeMemory).
  Work running when an OOM hits goes back to pending (redone, not failed).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import shutil
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Sequence, Tuple

from oh_streamer.queue import gpu_turns as gt
from oh_streamer.queue import stream_subtitle_gpu as ssg
from oh_streamer.queue.subtitle_scheduler import (
    LOCAL_EXECUTOR_NAME, SYNTHETIC, Cancelled, CloudHost, CloudSliceExecutor, FolderState, RunResult,
    mt_cache_params,
)
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr

LOCAL_EXECUTOR = sp.Executor(LOCAL_EXECUTOR_NAME)
MAX_ATTEMPTS = 3                  # local errors (crash, bad output) retried this often before `failed`
RETRY_WAIT_S = 30.0
BUDGET_EVENT_EVERY_S = 30.0       # a refused turn is logged (subtitle_budget) at most this often
MAX_CUT_AHEAD = 12                # media turns: lazy cuts decided per turn while preparing
BATCH_GAP_S = 3.0                 # silence between non-contiguous windows concatenated for one ASR run
LOCAL_ASR_PIPELINE = "local-slice-asr-v2-isolated"
LOCAL_MT_DISPLAY = "blocks-display-v1"


class GpuOutOfMemory(RuntimeError):
    """CUDA ran out of memory in a local model (detected from the child's error text / log)."""


class _Demoted(Exception):
    """Internal: the parallel guard tripped in the middle of a batch."""


# --------------------------------------------------------------------------------------------------
# Host interface
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RestoreInfo:
    lead_s: float = math.inf          # restored seconds continuous from the play position (inf = finished)
    rate: Optional[float] = None      # media seconds restored per wall second, recent (None = unknown)
    running: bool = False
    has_work: bool = False            # False: restoration finished or waiting for the download
    oom: bool = False                 # the restoration worker died of CUDA OOM (parallel guard)


@dataclass(frozen=True)
class GpuInfo:
    name: str = ""
    total_mib: Optional[float] = None


class GpuHost(CloudHost):
    """What the local executor needs from the queue service (M7 writes the adapter). Inherits the CloudHost
    part (source state, media id, scheduler context = play position / watched, cancel, progress, publish).
    The defaults = standalone run: no restoration, nobody watching, the GPU always free."""

    def acquire_lease(self, exclusive: bool) -> Optional[Any]:
        """A GPU lease for subtitle work, or None when the GPU cannot be used now (player busy, performance
        check, another job). exclusive=True: interleave (restoration was stopped by stop_restore first);
        False: parallel (shared with a running restoration). None is never an error (D1)."""
        return object()

    def release_lease(self, lease: Any) -> None:
        pass

    def stop_restore(self) -> bool:
        """Hard-stop the restoration worker for a subtitle turn (today's `preempt`); True if one was stopped."""
        return False

    def resume_restore(self) -> None:
        """Start restoration again after a subtitle turn (it resumes at the first missing segment)."""

    def restore_info(self) -> RestoreInfo:
        return RestoreInfo()

    def gpu_info(self) -> GpuInfo:
        """Name and total VRAM of the GPU the work runs on (the auto mode estimate and the per-card record)."""
        return GpuInfo()

    def note_vram_short(self, need_mib: float, free_mib: Optional[float]) -> None:
        """A media turn was given up because the free VRAM after the restore pause was below the model's need
        (T4.3: counted toward the player's gpu_short hint). Not an error of the slice."""


# --------------------------------------------------------------------------------------------------
# Model dependencies
# --------------------------------------------------------------------------------------------------

class LoadProbe:
    """Model load seconds seen in the phase events of the subtitle children (T4.2/T4.3 second round, fix 2). A child
    says "啟動字幕程序" when it starts and "...已就緒" (prepare_asr / prepare_translate) when its model is loaded, so
    load = ready - the latest start. Dual ASR runs one child per model: the loads add up. The executor arms the
    probe (begin) around a stage and reads it (take) afterwards; events outside that window are ignored."""
    START_MARK = "啟動字幕程序"
    READY_MARK = "已就緒"

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self._armed = False
        self._start: Optional[float] = None
        self._total = 0.0
        self._seen = False

    def begin(self) -> None:
        with self._lock:
            self._armed, self._start, self._total, self._seen = True, self.clock(), 0.0, False

    def event(self, phase: Any, percent: Any, detail: Any) -> None:
        text = str(detail or "")
        with self._lock:
            if not self._armed:
                return
            if self.START_MARK in text:
                self._start = self.clock()
            elif str(phase).startswith("prepare_") and self.READY_MARK in text and self._start is not None:
                self._total += max(0.0, self.clock() - self._start)
                self._seen, self._start = True, None

    def take(self) -> Optional[float]:
        """Seconds of model load since begin(), or None when no model finished loading (all cached, or failed)."""
        with self._lock:
            seconds = self._total if self._seen else None
            self._armed, self._total, self._seen = False, 0.0, False
            return seconds


AsrBatchFn = Callable[[Sequence[Tuple[float, float]], Path], List[Dict[str, Any]]]
TranslateFn = Callable[..., List[Dict[str, Any]]]


@dataclass
class LocalDeps:
    """asr_batch(windows, workdir) -> one dict per window {"segments": [...absolute times...],
    "speech_intervals": [[a, b], ...absolute], "model": str, "warnings": [...]}; it runs ONE ASR pass over all
    windows (models load once). mt_session() -> context manager yielding translate(ja_doc, workdir) -> translated
    rows (translate_file rows, absolute times); one llama-server per session. Both raise GpuOutOfMemory on OOM."""
    asr_batch: AsrBatchFn
    mt_session: Callable[[], ContextManager[TranslateFn]]
    asr_identity: Dict[str, Any]
    mt_identity: Dict[str, Any]
    asr_params: Dict[str, Any] = field(default_factory=dict)
    mt_params: Dict[str, Any] = field(default_factory=dict)
    mt_prompt_version: str = "sakura-official-numbered"
    mt_glossary_version: str = "default"
    probe_duration: Optional[Callable[[Any], Optional[float]]] = None
    detect_silences: Optional[Callable[..., Any]] = None
    relaxed_detect: Optional[Callable[..., Any]] = None
    asr_model_id: str = "local-asr"
    mt_model_id: str = "local-sakura"
    pieces: Any = None                # unused (cloud only); keeps the CloudDeps attribute set
    mt_batch_callback: bool = False   # translate(..., on_batch=...) supports cumulative raw rows
    mt_sentence_option: bool = False  # translate(..., sentence={...}) overrides the config's sentence options (M6b)
    # M4b: deps for another local frozen-plan binding (e.g. the local backup of a cloud primary); None = these
    # deps serve every local candidate. ledger / credentials: shared fallback.Ledger for cooldown dispatch.
    for_row: Optional[Callable[[Dict[str, Any]], "LocalDeps"]] = None
    ledger: Any = None
    credentials: Optional[Callable[[Dict[str, Any]], str]] = None
    asr_run: Any = None               # unused (cloud only)
    translator_for: Any = None        # unused (cloud only)
    # T4.2 fix 2: asr_prepare(windows, workdir) does the CPU-only half of asr_batch (window PCM + VAD span plans,
    # in a CUDA-less child) so it can run while restoration still holds the GPU; a later asr_batch(windows, same
    # workdir) finds it and skips that work. None = no separate preparation.
    asr_prepare: Optional[Callable[[Sequence[Tuple[float, float]], Path], None]] = None
    # T4.2/T4.3 fix 4: vram_probe(need_mib, wait_s) -> {"ok", "free_mib", "total_mib", ...}: torch.cuda.mem_get_info
    # measured inside a subtitle runtime child, re-measured for up to wait_s. None = no pre-load check.
    vram_probe: Optional[Callable[[float, float], Dict[str, Any]]] = None
    # T4.2/T4.3 second round, fix 2: the model load time of a stage, from the children's phase events. None = unknown.
    load_probe: Optional[LoadProbe] = None


def _digest(payload: Any, length: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def sakura_profile(model_path: Any, translation: Optional[dict] = None) -> str:
    """Same profile rule as blocks.run_block."""
    name = Path(model_path).name.lower()
    return (translation or {}).get("profile") or (
        "sakura-galtransl-7b-v3.7" if "galtransl-7b-v3.7" in name else
        "sakura-galtransl-14b-v3.8" if "galtransl-14b-v3.8" in name else "sakura-v1")


def local_identities(config: dict, models_dir: Path | str, model_path: Path | str) -> Dict[str, Any]:
    """Cache-key parts of the local ASR / Sakura bindings of a subtitle config (no item id, no snapshot)."""
    from oh_streamer.subtitles import runtime
    from oh_streamer.subtitles.model_adapters import model_fingerprint
    from oh_streamer.subtitles.pipeline import PRIMARY_ASR, SECONDARY_ASR
    asr = config.get("asr") or {}
    mode = asr.get("mode", "dual")
    aliases = (PRIMARY_ASR, SECONDARY_ASR) if mode == "dual" else (asr.get("model", PRIMARY_ASR),)
    fingerprints = {}
    for alias in aliases:
        try:
            fingerprints[alias] = runtime.local_model_fingerprint(Path(models_dir) / runtime.model_dir_name(alias))
        except (FileNotFoundError, OSError, ValueError):
            fingerprints[alias] = None
    translation, sentence, glossary = config.get("translation") or {}, config.get("sentence") or {}, \
        config.get("glossary") or {}
    profile = sakura_profile(model_path, translation)
    gloss_parts = [(Path(__file__).resolve().parents[1] / "subtitles" / "glossary_default.json")]
    if glossary.get("path"):
        gloss_parts.append(Path(glossary["path"]))
    gloss_digest = _digest([p.read_text("utf-8") if p.is_file() else None for p in gloss_parts]
                           + [bool(glossary.get("generic_only", True))])
    return dict(
        asr_identity={"provider": "local", "asr_mode": mode, "models": fingerprints},
        asr_params={"pipeline": LOCAL_ASR_PIPELINE, "vad_threshold": asr.get("vad_threshold"),
                    "vad_rescue_chunk_seconds": asr.get("vad_rescue_chunk_seconds", 60.0)},
        mt_identity={"provider": "local", "backend": "sakura", "model": model_fingerprint(model_path),
                     "profile": profile},
        mt_prompt_version=f"sakura-{translation.get('prompt_mode', 'official-numbered')}",
        mt_glossary_version=gloss_digest,
        mt_params={"display": LOCAL_MT_DISPLAY, "batch_size": translation.get("batch_size", 30),
                   "sentence_mode": sentence.get("sentence_mode", True),
                   "show_interjections": sentence.get("show_interjections", False),
                   "short_threshold": sentence.get("short_threshold", 1.0),
                   "short_policy": sentence.get("short_policy", "whitelist"),
                   "context_size": translation.get("context_size", 8192),
                   "gpu_layers": translation.get("gpu_layers", "auto"), "seed": translation.get("seed")},
        asr_model_id=f"local:{mode}" if mode == "dual" else f"local:{mode}:{aliases[0]}", mt_model_id=f"local:{profile}",
    )


def _concat_wavs(parts: Sequence[Path], out: Path, gap_s: float) -> List[float]:
    """Concatenate 16 kHz mono PCM wavs with `gap_s` of silence between them; returns each part's offset."""
    offsets, pos = [], 0.0
    with wave.open(str(out), "wb") as dst:
        params = None
        for i, part in enumerate(parts):
            with wave.open(str(part), "rb") as src:
                if params is None:
                    params = src.getparams()
                    dst.setparams(params)
                if i:
                    silence = int(round(gap_s * params.framerate))
                    dst.writeframes(b"\x00" * silence * params.sampwidth * params.nchannels)
                    pos += silence / params.framerate
                offsets.append(pos)
                frames = src.readframes(src.getnframes())
                dst.writeframes(frames)
                pos += src.getnframes() / params.framerate
    return offsets


def _interval(row: Any) -> Tuple[float, float]:
    return (float(row["start"]), float(row["end"])) if isinstance(row, dict) else (float(row[0]), float(row[1]))


def split_batch_result(doc: Dict[str, Any], windows: Sequence[Tuple[float, float]],
                       offsets: Sequence[float]) -> List[Dict[str, Any]]:
    """Map one ASR document over concatenated windows back to source time, per window (by segment midpoint;
    segments in a gap are dropped)."""
    from oh_streamer.subtitles.slice_results import _shift_segment
    spans = [(off, off + (b - a), a) for off, (a, b) in zip(offsets, windows)]

    def which(mid: float) -> Optional[int]:
        for i, (lo, hi, _a) in enumerate(spans):
            if lo <= mid < hi + 1e-6:
                return i
        return None

    out = [{"segments": [], "speech_intervals": [], "model": doc.get("model"),
            "warnings": list(doc.get("asr_warnings") or [])} for _ in windows]
    for seg in doc.get("segments") or []:
        i = which((float(seg["start"]) + float(seg["end"])) / 2.0)
        if i is not None:
            out[i]["segments"].append(_shift_segment(dict(seg), spans[i][2] - spans[i][0]) if spans[i][2] != spans[i][0]
                                      else dict(seg))
    for row in doc.get("speech_intervals") or []:
        a, b = _interval(row)
        i = which((a + b) / 2.0)
        if i is not None:
            shift = spans[i][2] - spans[i][0]
            out[i]["speech_intervals"].append([round(a + shift, 4), round(b + shift, 4)])
    return out


def default_local_deps(folder: Path | str, source: Path | str, config: dict, *, settings: Optional[dict] = None,
                       runtime_paths: Optional[dict] = None, keep_work: bool = False,
                       on_phase: Optional[Callable[[str, Any, str], None]] = None,
                       on_process: Optional[Callable[[Any], None]] = None) -> LocalDeps:
    """Real local models for one primary subtitle config (queue snapshot_subtitle_settings(...)["subtitle_configs"]
    [0]). runtime_paths: environment.runtime_paths(settings) plus optionally "sakura_model".
    on_process(popen): called with every model subprocess (ASR child, llama-server) right after it starts, so the
    owner can kill it on cancel (M7: QueueGpuHost.register_process)."""
    from oh_streamer.subtitles import runtime
    from oh_streamer.subtitles.environment import runtime_paths as resolve_paths
    paths = dict(runtime_paths or resolve_paths(settings or {}))
    models = Path(paths["models"])
    translation = config.get("translation") or {}
    model_path = Path(paths.get("sakura_model") or models / translation["model_file"])
    server = paths.get("server")
    python = paths.get("python") or sys.executable
    asr = config.get("asr") or {}
    ids = local_identities(config, models, model_path)
    source = Path(source)
    load_probe = LoadProbe()
    outer_phase = on_phase

    def on_phase(phase: str, percent: Any, detail: str) -> None:        # noqa: F811 - the probe sees every event
        load_probe.event(phase, percent, detail)
        if callable(outer_phase):
            outer_phase(phase, percent, detail)

    def asr_options() -> Dict[str, Any]:
        return dict(models_dir=models, asr_mode=asr.get("mode", "dual"),
                    asr_model=asr.get("model", "whisper-ja-1.5B"), force=True, gpu_managed=True,
                    vad_threshold=asr.get("vad_threshold"),
                    vad_rescue_chunk_seconds=asr.get("vad_rescue_chunk_seconds", 60.0))

    def window_jobs(windows: Sequence[Tuple[float, float]], workdir: Path) -> List[Dict[str, Any]]:
        """Window PCM (ffmpeg, CPU) once per workdir; `prepared.json` says which windows are already there."""
        marker = workdir / "prepared.json"
        wanted = [[float(a), float(b)] for a, b in windows]
        try:
            ready = json.loads(marker.read_text("utf-8")) == wanted
        except (OSError, ValueError):
            ready = False
        parts = []
        for i, (a, b) in enumerate(windows):
            part = workdir / f"window_{i}.wav"
            if not ready:
                runtime.resample_window_to_wav(source, part, a, b, pad=False)
            parts.append(part)
        if not ready:
            marker.write_text(json.dumps(wanted), encoding="utf-8")
        return [{"source": audio, "folder": workdir / f"asr_{i}"} for i, audio in enumerate(parts)]

    def asr_prepare(windows: Sequence[Tuple[float, float]], workdir: Path) -> None:
        """PCM + VAD span plans of the windows; the VAD runs in a subtitle-runtime child that sees no CUDA."""
        from oh_streamer.subtitles.pipeline import run_production_asr_prepare_child
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        jobs = window_jobs(windows, workdir)
        run_production_asr_prepare_child(
            jobs, workdir=workdir / "asr_prepare", python=python, on_phase=on_phase, on_process=on_process,
            **asr_options())

    def vram_probe(need_mib: float, wait_s: float) -> Dict[str, Any]:
        """One subtitle-runtime child (stage vram_probe) with torch: the queue's lada venv never imports it. The
        same runtime python hosts the ASR and the Sakura (llama.cpp) children, so one probe serves both."""
        from oh_streamer.subtitles.pipeline import run_vram_probe_child
        workdir = Path(folder) / "cache" / "tmp" / f"vram_probe_{time.monotonic_ns()}"
        try:
            return run_vram_probe_child(need_mib, workdir=workdir, wait_s=wait_s, python=python,
                                        on_phase=on_phase, on_process=on_process)
        finally:
            if not keep_work:
                shutil.rmtree(workdir, ignore_errors=True)

    def asr_batch(windows: Sequence[Tuple[float, float]], workdir: Path) -> List[Dict[str, Any]]:
        from oh_streamer.subtitles.pipeline import run_production_asr_windows_child
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            jobs = window_jobs(windows, workdir)
            # Keep VAD/fusion independent per slice while loading each model once for the whole turn. The whole
            # run (VAD prepare, model children, fusion) is one subtitle-runtime child: the queue's venv has no
            # silero_vad / subtitle torch stack. on_process gets that child; killing it ends its model child too.
            try:
                docs = run_production_asr_windows_child(
                    jobs, workdir=workdir / "asr_windows", python=python, on_phase=on_phase, on_process=on_process,
                    **asr_options())
            except Exception as exc:
                if gt.is_cuda_oom(exc):
                    raise GpuOutOfMemory(str(exc)[-500:]) from exc
                raise
            results = []
            for doc, (a, b) in zip(docs, windows):
                oom = [w for w in doc.get("asr_warnings") or [] if gt.is_cuda_oom(w)]
                if oom:
                    raise GpuOutOfMemory(str(oom[0])[-500:])
                if doc.get("error"):
                    results.append({"error": str(doc["error"]), "segments": []})
                    continue
                results.extend(split_batch_result(doc, [(a, b)], [0.0]))
            return results
        finally:
            if not keep_work:
                shutil.rmtree(workdir, ignore_errors=True)

    @contextlib.contextmanager
    def mt_session() -> Iterator[TranslateFn]:
        """One subtitle-runtime child (worker stage mt_session) per session: llama-server, OpenCC, sentence
        options and residue filtering run there, never in this (lada venv) process. on_process gets the child;
        killing it ends the llama-server grandchild too (kill-on-close job object / tree kill)."""
        from oh_streamer.subtitles.pipeline import MtSessionChild
        glossary = config.get("glossary") or {}
        log = Path(folder) / "cache" / "tmp" / "sakura.server.log"
        session_dir = Path(folder) / "cache" / "tmp" / f"mt_session_{time.monotonic_ns()}"
        child = MtSessionChild(
            dict(model_path=str(model_path), server=None if server is None else str(server),
                 profile=sakura_profile(model_path, translation), translation=translation, glossary=glossary,
                 sentence=config.get("sentence") or {}, server_log=str(log)),
            session_dir, python=python, on_phase=on_phase, on_process=on_process)

        def translate(ja_doc: Dict[str, Any], workdir: Path, *, on_batch=None,
                      sentence: Optional[dict] = None) -> List[Dict[str, Any]]:
            workdir = Path(workdir)
            try:
                return child.translate(ja_doc, workdir, sentence=sentence, on_batch=on_batch)
            except Exception as exc:
                tail = ""
                try:
                    tail = log.read_text("utf-8", errors="replace")[-4000:]
                except OSError:
                    pass
                if gt.is_cuda_oom(exc) or gt.is_cuda_oom(tail):
                    raise GpuOutOfMemory(str(exc)[-300:]) from exc
                raise
            finally:
                if not keep_work:
                    shutil.rmtree(workdir, ignore_errors=True)

        try:
            yield translate
        finally:
            child.close()
            if not keep_work:
                shutil.rmtree(session_dir, ignore_errors=True)

    def probe(src):
        return runtime._probe_audio_duration(str(src), runtime.resolve_ffmpeg(), 15)

    def for_row(row: Dict[str, Any]) -> LocalDeps:
        """Deps for a local frozen-plan binding: a dual ASR row (model_path = models folder), a single built-in ASR row (recipe = alias, model_path = model folder) or a Sakura gguf
        translation row (model_path = .gguf, recipe = profile). Other local ASR recipes (custom decode, unknown models) are not supported by
        the slice executor yet (ProviderError 'unsupported' -> the chain moves on)."""
        from oh_streamer.subtitles.providers.http import ProviderError
        from oh_streamer.subtitles.environment import model_file
        cfg = json.loads(json.dumps(config))
        sub_paths = dict(paths)
        if row.get("kind") == "asr":
            file = model_file(row, settings, models=sub_paths.get("models"))
            if not file:
                raise ProviderError("unsupported", "local ASR binding not supported by the slice executor")
            if row.get("recipe") == "__dual__":
                cfg.setdefault("asr", {})["mode"] = "dual"
                sub_paths["models"] = file
            else:
                # Single built-in model (registry row: recipe = MODEL_SPECS alias, model_path = <models>/<dir>).
                alias = str(row.get("recipe") or "").strip().lower()
                mpath = Path(file)
                if alias not in runtime.MODEL_SPECS or mpath.name.lower() != runtime.model_dir_name(alias).lower():
                    raise ProviderError("unsupported", "local ASR binding not supported by the slice executor")
                asr_cfg = cfg.setdefault("asr", {})
                asr_cfg["mode"] = "single"
                asr_cfg["model"] = runtime.model_dir_name(alias)
                sub_paths["models"] = str(mpath.parent)
        else:
            file = model_file(row, settings, models=sub_paths.get("models"))
            if not file:
                raise ProviderError("unsupported", "local translation model missing")
            sub_paths["sakura_model"] = file
            if str(row.get("recipe") or "").startswith("sakura"):
                cfg.setdefault("translation", {})["profile"] = row["recipe"]
        return default_local_deps(folder, source, cfg, settings=settings, runtime_paths=sub_paths,
                                  keep_work=keep_work, on_phase=on_phase, on_process=on_process)

    from oh_streamer.subtitles.model_adapters import _binding_credentials
    return LocalDeps(asr_batch=asr_batch, asr_prepare=asr_prepare, vram_probe=vram_probe, load_probe=load_probe,
                     mt_session=mt_session, probe_duration=probe,
                     mt_batch_callback=True, mt_sentence_option=True, for_row=for_row, ledger=fb.Ledger(),
                     credentials=lambda row: _binding_credentials(row, settings or {}), **ids)


def display_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A translate_file row + the fields slice_results.publish reads, with the blocks display rule
    (postprocess.display_segments + tracks.render_tracks): failed / leftover-Japanese rows hidden, Taiwan wording,
    low-confidence and onomatopoeia cues only in the full track, low confidence marked there."""
    from oh_streamer.subtitles.postprocess import display_segments
    from oh_streamer.subtitles.quality import is_onomatopoeia
    shown = display_segments([row], "hide", "keep")
    zh = shown[0].text if shown else ""
    low = bool(row.get("low_confidence"))
    onomatopoeia = bool(row.get("onomatopoeia", is_onomatopoeia(row.get("text_ja", ""))))
    out = dict(row, zh=zh, mt_zh=zh, is_filtered=not zh, status="mt_ready",
               full_only=low or onomatopoeia, full_prefix="（低信心）" if low else "")
    if not zh:
        out["filter_reason"] = "translation_failed"
    return out


def _shift(rows: Sequence[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
    from oh_streamer.subtitles.slice_results import _shift_segment
    return [_shift_segment(dict(r), offset) if offset else dict(r) for r in rows]


# --------------------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------------------

class LocalSliceExecutor(CloudSliceExecutor):
    """Runs the local asr / mt work of a slice plan on the GPU, in batches, interleaved with (or parallel to)
    restoration. One instance per item and run. `run(wait=...)` loops; `step()` makes one decision (M7 can call
    it from the queue tick instead).

    M4: shares the folder's PlanStore with the cloud executor. It takes only units whose current candidate is
    a local one (fallback.owner_function), runs the local candidate the unit is on (a cloud primary's local
    backup too), and on local errors asks CandidateChain.decide (retries, then e.g. the cloud backup)."""

    EXECUTOR_NAME = LOCAL_EXECUTOR_NAME
    EXECUTOR = LOCAL_EXECUTOR
    PLAN_SLICE_SECONDS = sp.DEFAULT_SLICE_SECONDS    # local plans keep 300 s slices

    def __init__(self, folder: Path | str, source: Path | str, *, host: Optional[GpuHost] = None,
                 deps: Optional[LocalDeps] = None, config: Optional[dict] = None, settings: Optional[dict] = None,
                 resource_mode: str = gt.MODE_AUTO, turns: Optional[gt.TurnConfig] = None,
                 vram_requirements: Optional[Dict[str, float]] = None,
                 memory: Optional[gt.GpuModeMemory] = None,
                 speed_memory: Optional[ssg.SubtitleSpeedMemory] = None,
                 cost_memory: Optional[ssg.SwitchCostMemory] = None, poll_s: float = 2.0, **kwargs: Any):
        host = host or GpuHost()
        if deps is None:
            deps = default_local_deps(folder, source, config or {}, settings=settings)
        kwargs.setdefault("asr_window", sp.AsrWindow(context_grid_s=float(
            deps.asr_params.get("vad_rescue_chunk_seconds", 60.0))))
        super().__init__(folder, source, host=host, deps=deps, settings=settings, config=config,
                         poll_s=poll_s, **kwargs)
        self.host: GpuHost = host
        self.resource_setting = resource_mode
        self.turns = turns or gt.TurnConfig.from_settings(settings)
        self.vram_requirements = vram_requirements
        self.memory = memory
        self.speed_memory = speed_memory                   # learned subtitle speed per card (best effort)
        self.cost_memory = cost_memory                     # media_turns: learned model load / restart per card
        self.mode: Optional[str] = None
        self.mode_reason = ""
        self.idle_wake_at: Optional[float] = None          # step() "idle": earliest time work may come back here
        self.idle_waiting_executor: Optional[str] = None
        self._gpu_key = ""
        self._lease: Any = None
        self._stopped_restore = False
        self._turn_started: Optional[float] = None
        self.last_turn: Optional[gt.GpuTurn] = None
        self.events: List[Dict[str, Any]] = []        # timeline for tools / UI: switches, batches, demotions
        self.switches = 0
        self._seconds_per_media_second = self.turns.subtitle_seconds_per_media_second
        self._pass: Optional[Tuple[sp.SchedulerContext, Dict[str, Any], Any]] = None   # last _ctx() pass
        self._task_candidate: Dict[Tuple[int, str, str], Any] = {}
        self._row_deps: Dict[str, LocalDeps] = {}
        self._turn_s = self.turns.turn_first_s             # media_turns: size of the next turn (doubles per turn)
        self._prepared: Dict[Tuple[str, tuple], Path] = {}  # (candidate id, windows) -> workdir prepared before the lease
        self._vram_retry_at = 0.0                          # media_turns: no new pause before this (a short turn)
        self._asr_load_s = self._mt_load_s = self._restart_s = 0.0
        self._use_learned_costs()
        self._mt_spm = self.turns.mt_seconds_per_media_second   # translation seconds per media second (warm)
        self._turn_loads: Dict[str, float] = {}            # model load seconds measured in the running batch
        self._budget_event_at = -math.inf                  # last subtitle_budget event
        self._budget_info: Dict[str, Any] = {}             # numbers of the last refused plan

    # -- events ---------------------------------------------------------------------------------
    def _event(self, kind: str, **info: Any) -> None:
        row = {"t": self.clock(), "kind": kind, **info}
        self.events.append(row)
        self._progress(state=kind, **{k: v for k, v in info.items() if k not in ("state",)})

    # -- candidates (M4) ------------------------------------------------------------------------
    def _chain_kwargs(self) -> Dict[str, Any]:
        # local errors keep the M3 back-off (RETRY_WAIT_S * attempt) before moving to the next candidate
        return {"retry_delay": lambda _candidate, attempt: RETRY_WAIT_S * attempt}

    def _error_kind(self, exc: Exception) -> Optional[str]:
        if isinstance(exc, GpuOutOfMemory):
            return "oom"
        from oh_streamer.subtitles.providers.http import ProviderError
        return exc.kind if isinstance(exc, ProviderError) else "error"

    def _deps_for(self, candidate) -> LocalDeps:
        """Models of a local candidate: the configured deps, or deps.for_row(row) for a frozen-plan binding
        (e.g. the local backup of a cloud primary) when the deps can build them."""
        if candidate is None or not candidate.row or candidate.row.get(SYNTHETIC) or self.deps.for_row is None:
            return self.deps
        key = json.dumps(dict(candidate.row), sort_keys=True, default=str)
        if key not in self._row_deps:
            self._row_deps[key] = self.deps.for_row(dict(candidate.row))
        return self._row_deps[key]

    def _identity(self, stage: str, candidate) -> Dict[str, Any]:
        deps = self._deps_for(candidate)
        return dict(deps.asr_identity if stage == sp.STAGE_ASR else deps.mt_identity)

    def _model_id(self, stage: str, candidate) -> str:
        deps = self._deps_for(candidate)
        return deps.asr_model_id if stage == sp.STAGE_ASR else deps.mt_model_id

    def _candidate_of(self, task: sp.Task):
        """Candidate a claimed unit runs on (set by _claim; else from the stored state / the chain head)."""
        cand = self._task_candidate.get((task.slice_index, task.stage, task.track))
        if cand is not None:
            return cand
        chain = self._chain(self._frozen(task.track), task.stage)
        st = self.plan.slices[task.slice_index].get(task.stage, task.track)
        return chain.get(st.candidate) or next((c for c in chain if c.executor == self.EXECUTOR_NAME), None) \
            or next(iter(chain), None)

    def _claim_task(self, task: sp.Task) -> bool:
        """Under the plan lock, with the pass from the last _ctx(): claim one selected unit."""
        if self._pass is None:
            self._ctx(None)
        ctx, plans, cooldown = self._pass
        cand, _changed = self._claim(task, plans, cooldown, self.clock())
        if cand is None:
            return False
        self._task_candidate[(task.slice_index, task.stage, task.track)] = cand
        return True

    def _load_partials(self) -> None:
        # Keep already published batches visible while a restarted model rebuilds its translation memo.
        # Partial output never makes the task done and is used only under the exact current MT key.
        super()._load_partials()
        for track in self._ready_tracks():
            chain = self._chain(self._frozen(track), sp.STAGE_MT)
            home = next((c for c in chain if c.executor == self.EXECUTOR_NAME), None)
            if home is None:
                continue
            done, partial = self._rows(track), self._partials(track)
            for s in self.plan.slices:
                if s.index in done or s.index in partial or s.is_done(sp.STAGE_MT, track):
                    continue
                key, doc = self._asr_doc(s, track)
                if doc:
                    st = s.get(sp.STAGE_MT, track)
                    cand = chain.get(st.candidate) if st.candidate else home
                    if cand is None or cand.executor != self.EXECUTOR_NAME:
                        continue
                    rows = sr.read_partial(self.folder,
                                           self._mt_key(self._digest_of(key, doc["segments"]), cand, track))
                    if rows:
                        partial[s.index] = rows

    def _mt_params(self, candidate, track: str = sp.PRIMARY_TRACK) -> Dict[str, Any]:
        """Primary: the deps' params (keys unchanged). A comparison row translates with its own sentence options
        (short-sentence threshold, interjections ...) and its own translation model, so both are in its key."""
        params = self._deps_for(candidate).mt_params
        if track == self.plan.primary:
            return params
        wanted = ("sentence_mode", "short_threshold", "short_policy", "show_interjections")
        own = mt_cache_params(self._track_config(track), self.settings)
        extra: Dict[str, Any] = {"track_sentence": {k: own[k] for k in wanted}}
        row = ((self._frozen(track).get("models") or {}).get("translation"))
        if row and not row.get(SYNTHETIC):
            extra["track_mt"] = self._row_identity(sp.STAGE_MT, fb.Candidate(index=0, id=str(row.get("id", "")),
                                                                              row=dict(row), executor=""))
        return {**params, **extra}

    # -- mode -----------------------------------------------------------------------------------
    def _resolve_mode(self) -> None:
        info = self.host.gpu_info() or GpuInfo()
        self._gpu_key = gt.gpu_key(info.name, info.total_mib)
        self._use_learned_speed()
        self._use_learned_costs()
        remembered = self.memory.get(self._gpu_key) if self.memory is not None else None
        self.mode, self.mode_reason = gt.choose_resource_mode(
            self.resource_setting, info.total_mib, asr_mode=str(self.deps.asr_identity.get("asr_mode", "dual")),
            requirements=self.vram_requirements, remembered=remembered)
        self._event("mode", mode=self.mode, reason=self.mode_reason, gpu=self._gpu_key)

    def _use_learned_speed(self) -> None:
        """A speed learned on this card replaces the built-in default of the turn budget (never below 0.2)."""
        if self.speed_memory is None:
            return
        try:
            learned = self.speed_memory.get(self._gpu_key)
        except Exception:
            return
        if learned is not None:
            self._seconds_per_media_second = max(learned, 0.2)

    def _use_learned_costs(self) -> None:
        """Model load (ASR, translation) and restoration restart of the turn budget: the values learned on this card
        (media turns only), else the config's (subtitle_load_s split by mt_load_s, restore_restart_s)."""
        asr, mt = ssg.default_load_split(self.turns)
        restart = float(self.turns.restore_restart_s)
        if self.turns.media_turns and self.cost_memory is not None and self._gpu_key:
            learned = ssg.learned_costs(self._gpu_key, self.cost_memory, self.turns)
            asr, mt, restart = learned["asr_load_s"], learned["mt_load_s"], learned["restart_s"]
        self._asr_load_s, self._mt_load_s, self._restart_s = asr, mt, restart

    def _load_total(self) -> float:
        return self._asr_load_s + self._mt_load_s

    def _learn_load(self, name: str, seconds: Optional[float]) -> None:
        """A stage's measured model load: kept for this batch (speed) and, for media turns, remembered per card."""
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            return
        self._turn_loads[name] = self._turn_loads.get(name, 0.0) + seconds
        self._event("load_measured", stage=name, seconds=round(seconds, 2))
        if self.turns.media_turns and self.cost_memory is not None and self._gpu_key:
            try:
                self.cost_memory.record(self._gpu_key, ssg.COST_ASR_LOAD if name == "asr" else ssg.COST_MT_LOAD, seconds)
            except Exception:
                pass

    def _record_speed(self, picked: Sequence[sp.Task], wall_s: float) -> None:
        """Best effort: wall seconds of a successful batch per media second of the slices it finished."""
        if self.speed_memory is None or not self._gpu_key:
            return
        try:
            done = {(t.slice_index, t.track) for t in picked}
            media_s = 0.0
            with self._lock:
                for index, track in done:
                    s = self.plan.slices[index]
                    if not all(s.is_done(stage, track) for stage in sp.STAGES):
                        return              # a unit failed / was deferred: the wall time says nothing
                    media_s += s.end - s.start
            if media_s > 0:
                self.speed_memory.record(self._gpu_key, wall_s, media_s)
        except Exception:
            pass

    def _demote(self, reason: str, detail: str = "") -> None:
        """Parallel -> interleave for the rest of the run; remembered for this card when the setting is auto."""
        self.mode, self.mode_reason = gt.MODE_INTERLEAVE, f"demoted:{reason}"
        if self.memory is not None and (self.resource_setting or gt.MODE_AUTO) == gt.MODE_AUTO:
            self.memory.remember(self._gpu_key, gt.MODE_INTERLEAVE, reason, detail)
        self._end_gpu_turn()
        self._event("demote", reason=reason, detail=str(detail)[:200], gpu=self._gpu_key)

    def _guard(self) -> Optional[str]:
        if self.mode != gt.MODE_PARALLEL:
            return None
        info = self.host.restore_info()
        ctx = self.host.scheduler_context()
        return gt.parallel_guard(gt.GuardInput(oom=info.oom, watched=ctx.watched, restore_rate=info.rate,
                                               restore_lead_s=info.lead_s), self.turns)

    def _guard_point(self) -> None:
        reason = self._guard()
        if reason:
            raise _Demoted(reason)

    # -- GPU turns ------------------------------------------------------------------------------
    def _turn_state(self, ctx: sp.SchedulerContext) -> gt.TurnState:
        info = self.host.restore_info()
        sub_lead = max(0.0, sp.covered_end(self.plan, ctx.play_pos) - ctx.play_pos)
        return gt.TurnState(sub_lead_s=sub_lead, restore_lead_s=info.lead_s, watched=ctx.watched,
                            subtitle_work=True, restore_work=info.has_work)

    def _begin_gpu_turn(self, exclusive: bool) -> bool:
        if self._lease is not None:
            return True
        t0 = self.clock()
        stopped = self.host.stop_restore() if exclusive else False
        lease = self.host.acquire_lease(exclusive)
        if lease is None:
            if stopped:
                self.host.resume_restore()
            self._event("gpu_busy", exclusive=exclusive)
            return False
        self._lease, self._stopped_restore, self._turn_started = lease, stopped, self.clock()
        if exclusive:
            self.switches += 1
        self._event("gpu_acquired", exclusive=exclusive, restore_stopped=stopped,
                    stop_s=round(self._turn_started - t0, 3))
        return True

    def _end_gpu_turn(self) -> None:
        if self._lease is None:
            return
        lease, self._lease = self._lease, None
        try:
            self.host.release_lease(lease)
        finally:
            held = None if self._turn_started is None else round(self.clock() - self._turn_started, 3)
            resumed = self._stopped_restore
            self._stopped_restore = False
            if resumed:
                self.host.resume_restore()
            self._event("gpu_released", held_s=held, restore_resumed=resumed)

    # -- main loop ------------------------------------------------------------------------------
    def _ctx(self, limit: Optional[float]) -> sp.SchedulerContext:
        """Scheduling context of one pass, with the M4 owner (only units on a local candidate are ours)."""
        ctx, frozen, cooldown = self._dispatch(limit)
        self._pass = (ctx, frozen, cooldown)
        return ctx

    def _owner_ok(self, stage: str, track: str, st: sp.TaskState) -> bool:
        if self._pass is None:
            self._ctx(None)
        owner = self._pass[0].owner(stage, track, st)
        return not owner or owner == self.EXECUTOR_NAME

    def step(self, limit: Optional[float] = None) -> str:
        """One decision: "batch" (ran one), "restore_turn" (interleave: restoration keeps the GPU), "gpu_busy"
        (no lease now; work stays pending, D1: never handed to the cloud for that), "demoted" (parallel guard
        tripped), "idle" (no local task can run now: see _idle_reason)."""
        if self.host.seek_pending():
            self._turn_s = self.turns.turn_first_s      # interrupted: start small again
            self._vram_retry_at = 0.0
            return "idle"
        with self._lock:
            if sp.select_next_task(self.plan, self._ctx(limit), self.EXECUTOR) is None:
                self._end_gpu_turn()
                idle = True
            else:
                idle = False
        if idle:
            # Wake-up hint for the queue tick: when work now owned by another executor may come back here.
            self.idle_wake_at = None
            try:
                _reason, until, self.idle_waiting_executor = self._idle_reason(limit)
                self.idle_wake_at = until
            except Exception:
                self.idle_waiting_executor = None
            return "idle"
        if self.mode == gt.MODE_PARALLEL:
            reason = self._guard()
            if reason:
                self._demote(reason)
                return "demoted"
            if not self._begin_gpu_turn(exclusive=False):
                return "gpu_busy"
            n = 1 if not self._mt_rows else max(1, int(self.turns.batch_slices))
            urgent = False
        else:
            ctx = self._ctx(limit)
            turn = gt.decide_gpu_turn(self._turn_state(ctx), self.turns)
            # Publish the short head before filling a normal two-slice batch. Otherwise the viewer can
            # start while both ASR windows are running and wait several minutes for the first subtitle.
            head = next((s for s in self.plan.slices if s.start == 0), None)
            if (not ctx.watched and head is not None and not self._mt_rows
                    and not head.is_done(sp.STAGE_MT, self.plan.primary)):
                stage = sp.STAGE_ASR if not head.is_done(sp.STAGE_ASR, self.plan.primary) else sp.STAGE_MT
                if self._take(head, stage, self.plan.primary) and (limit is None or head.end <= limit + 0.5):
                    turn = gt.GpuTurn(gt.SUBTITLES, 1, "first_subtitle")
            self.last_turn = turn
            if turn.kind == gt.RESTORE:
                self._end_gpu_turn()
                return "restore_turn"
            if self.turns.media_turns:
                return self._media_turn(ctx, limit, turn)
            n, urgent = turn.slices, turn.urgent
            if ctx.watched and self.host.restore_info().has_work:
                # Fixed watermarks alone are unsafe: a 150 s lead cannot cover a 183 s subtitle turn
                # plus a 26 s restore restart. Budget the actual candidate windows before stopping it.
                lead = self.host.restore_info().lead_s
                while n > 0:
                    candidates = self._preview_batch(n, limit)
                    work_s = sum(self.plan.slices[t.slice_index].end
                                 - (self.asr_window.audio_start(self.plan.slices[t.slice_index])
                                    if t.stage == sp.STAGE_ASR else self.plan.slices[t.slice_index].start)
                                 for t in candidates)
                    required = (work_s * self._seconds_per_media_second + self.turns.subtitle_load_s
                                + self.turns.restore_restart_s + self.turns.playback_reserve_s)
                    if candidates and lead >= required:
                        break
                    n -= 1
                if n == 0:
                    self._end_gpu_turn()
                    self.last_turn = gt.GpuTurn(gt.RESTORE, reason="subtitle_budget")
                    self._budget_event(lead_s=lead, work_s=work_s, target_s=None,
                                       load_s=self.turns.subtitle_load_s, restart_s=self.turns.restore_restart_s,
                                       floor_s=self.turns.playback_reserve_s)
                    return "restore_turn"
            if not self._begin_gpu_turn(exclusive=True):
                return "gpu_busy"
            self._event("turn", reason=turn.reason, slices=n)
        try:
            self._run_batch(n, limit, urgent=urgent)
        except _Demoted as exc:
            self._release_mine()
            self._demote(str(exc))
            return "demoted"
        return "batch"

    # -- media-second turns (stream items, T4.2 fixes 1-3) ------------------------------------------
    def _batch_work_s(self, tasks: Sequence[sp.Task]) -> float:
        """Media seconds the GPU works on for these units (ASR includes the pre-roll)."""
        return sum(self.plan.slices[t.slice_index].end
                   - (self.asr_window.audio_start(self.plan.slices[t.slice_index])
                      if t.stage == sp.STAGE_ASR else self.plan.slices[t.slice_index].start)
                   for t in tasks)

    def _lead_after(self, lead_s: float, work_s: float) -> float:
        """Restore lead left once `work_s` media seconds of subtitles are done: the lead now, minus the GPU time of
        the work, the model loads (learned per card) and the restoration restart (learned per card)."""
        return lead_s - work_s * self._seconds_per_media_second - self._load_total() - self._restart_s

    def _lead_floor_s(self) -> float:
        return max(self.turns.urgent_min_restore_s, self.turns.playback_reserve_s)

    def _early_end(self, picked: Sequence[sp.Task]) -> bool:
        """Between ASR and MT (second round, fix 3): the translation is finished by default. End the turn only when
        the restore lead after it would drop below turns.mt_min_restore_s. Estimate = MT load (the translation
        model only: the ASR load is already paid) + the translation work of the slices still to translate; the lead
        is the live one, so the time ASR just took is already out of it. A batch holding the slice that contains the
        play position never defers its translation."""
        ctx = self.host.scheduler_context()
        info = self.host.restore_info()
        if not (ctx.watched and info.has_work):
            return False
        with self._lock:
            here = sp.slice_containing(self.plan, ctx.play_pos)
            media_s = sum(self.plan.slices[i].end - self.plan.slices[i].start
                          for i in {t.slice_index for t in picked
                                    if not self.plan.slices[t.slice_index].is_done(sp.STAGE_MT, t.track)})
            if here is not None and any(t.slice_index == here.index for t in picked):
                return False
        expect = self._mt_load_s + media_s * self._mt_spm
        return (info.lead_s - expect - self._restart_s) < self.turns.mt_min_restore_s

    _preview_oversize = False       # the last media-turn _preview_batch stopped before a slice that is too long

    def _covered_s(self, units: Sequence[sp.Task]) -> float:
        return sum(self.plan.slices[i].end - self.plan.slices[i].start for i in {t.slice_index for t in units})

    def _cut_ahead(self, limit: Optional[float], target_s: float) -> None:
        """Second round, fix 1: cut lazy slices forward from the play position (the silence search is CPU / IO work,
        done while restoration still runs) until the units a turn would take cover `target_s` media seconds, so a
        batch holds several slices. At most MAX_CUT_AHEAD cuts; stops when nothing can be cut or a seek is pending."""
        for _ in range(MAX_CUT_AHEAD):
            self._check()
            if self.host.seek_pending():
                return
            if self._covered_s(self._preview_batch(0, limit, target_s)) >= target_s:
                return
            if self._preview_oversize:
                return                     # an existing slice too long for this turn is next: cutting beyond it is useless
            if not self._extend_one_cut(limit, slice_seconds=sp.SHORT_SLICE_SECONDS):
                return

    def _budget_event(self, **info: Any) -> None:
        """Log a turn the budget refused (event subtitle_budget: lead, work estimate, costs, target size), at most
        once per BUDGET_EVENT_EVERY_S so a long refusal does not flood the log."""
        now = self.clock()
        if now - self._budget_event_at < BUDGET_EVENT_EVERY_S:
            return
        self._budget_event_at = now

        def num(v: Any) -> Any:
            return round(float(v), 2) if isinstance(v, (int, float)) and math.isfinite(v) else None
        out = {k: num(v) for k, v in info.items()}
        out["s"] = round(self._seconds_per_media_second, 3)
        self._event("subtitle_budget", **out)

    def _plan_media_turn(self, ctx: sp.SchedulerContext, limit: Optional[float]) -> Tuple[List[sp.Task], float]:
        """(units, size): units for `self._turn_s` media seconds, shrunk (halving, not below turn_min_s) until the
        restore lead after the work stays above the floor. ([], size) = does not fit even at the minimum
        (self._budget_info then holds the numbers)."""
        size = max(1.0, float(self._turn_s))
        info = self.host.restore_info()
        budget = ctx.watched and info.has_work
        while True:
            units = self._preview_batch(0, limit, size)
            work_s = self._batch_work_s(units)
            if (not units or not budget
                    or self._lead_after(info.lead_s, work_s) >= self._lead_floor_s()):
                return units, size
            if size <= self.turns.turn_min_s:
                self._budget_info = dict(lead_s=info.lead_s, work_s=work_s, target_s=float(self._turn_s), min_s=size,
                                         load_s=self._load_total(), restart_s=self._restart_s,
                                         floor_s=self._lead_floor_s())
                return [], size
            size = max(self.turns.turn_min_s, size / 2.0)

    def _media_turn(self, ctx: sp.SchedulerContext, limit: Optional[float], turn: gt.GpuTurn) -> str:
        """One turn of `_turn_s` media seconds (consecutive lazy slices in one batch). The CPU preparation (cutting
        ahead, audio, VAD) runs before the restoration is paused; the lease ends with the batch so the next turn is
        prepared while restoration runs again."""
        self._use_learned_costs()                     # the host may have learned a restart since the last turn
        if self._lease is None:
            self._cut_ahead(limit, float(self._turn_s))
            ctx = self._ctx(limit)
        units, size = self._plan_media_turn(ctx, limit)
        if not units:
            self._end_gpu_turn()
            self.last_turn = gt.GpuTurn(gt.RESTORE, reason="subtitle_budget")
            self._budget_event(**self._budget_info)
            return "restore_turn"
        if self._lease is None:
            self._prepare_turn(units)
            if self.host.seek_pending():
                return "idle"
            units, size = self._plan_media_turn(self._ctx(limit), limit)    # the lead moved while preparing
            if not units:
                self.last_turn = gt.GpuTurn(gt.RESTORE, reason="subtitle_budget")
                self._budget_event(**self._budget_info)
                return "restore_turn"
        if self.clock() < self._vram_retry_at:
            self.last_turn = gt.GpuTurn(gt.RESTORE, reason="vram_short")
            return "restore_turn"
        if not self._begin_gpu_turn(exclusive=True):
            return "gpu_busy"
        if not self._vram_ready(units):
            self._end_gpu_turn()                   # gives the GPU back: restoration resumes; the units stay pending
            self._vram_retry_at = self.clock() + max(0.0, float(self.turns.vram_retry_s))
            self.last_turn = gt.GpuTurn(gt.RESTORE, reason="vram_short")
            return "restore_turn"
        self._event("turn", reason=turn.reason, seconds=round(size, 1))
        try:
            done = self._run_batch(0, limit, urgent=turn.urgent, target_s=size)
        except _Demoted as exc:
            self._release_mine()
            self._demote(str(exc))
            return "demoted"
        self._end_gpu_turn()
        if done:
            self._turn_s = min(self.turns.turn_max_s, size * max(1.0, self.turns.turn_growth))
        return "batch"

    def _vram_need_mib(self, units: Sequence[sp.Task]) -> float:
        """Free VRAM the turn's models need beside the paused restoration: the largest need among the stages in the
        turn (ASR and MT load one after the other, never together) plus the margin."""
        req = dict(gt.VRAM_REQUIREMENTS_MIB)
        req.update(self.vram_requirements or {})
        needs = []
        if any(t.stage == sp.STAGE_ASR for t in units):
            mode = str(self.deps.asr_identity.get("asr_mode", "dual"))
            needs.append(float(req.get(f"asr_{mode}", req.get("asr_dual", 0.0))))
        if any(t.stage == sp.STAGE_MT for t in units):
            needs.append(float(req["mt_sakura_7b"]))
        return (max(needs) if needs else 0.0) + float(self.turns.vram_margin_mib)

    def _vram_ready(self, units: Sequence[sp.Task]) -> bool:
        """After the pause is confirmed, before any model loads: is enough VRAM free? The probe (a subtitle child with
        torch) re-measures for turns.vram_wait_s. A short result tells the host (gpu_short count) and ends the turn;
        a probe that cannot run never blocks the turn (the OOM guard still protects)."""
        probe = self.deps.vram_probe
        if probe is None or not units:
            return True
        need = self._vram_need_mib(units)
        try:
            res = probe(need, float(self.turns.vram_wait_s))
        except Exception as exc:  # noqa: BLE001
            self._check()              # cancelled (the probe child was killed): not a failure
            self._event("vram_probe_failed", error=str(exc)[-200:])
            return True
        if not isinstance(res, dict) or res.get("ok", True) is not False:
            self._event("vram_ok", need_mib=round(need), free_mib=(res or {}).get("free_mib") if isinstance(res, dict) else None)
            return True
        free = res.get("free_mib")
        self._event("vram_short", need_mib=round(need), free_mib=free, waited_s=res.get("waited_s"))
        try:
            self.host.note_vram_short(need, free if isinstance(free, (int, float)) else None)
        except Exception as exc:  # noqa: BLE001 - a hint only
            self._event("vram_note_failed", error=str(exc)[-200:])
        return False

    def _prepare_turn(self, units: Sequence[sp.Task]) -> None:
        """CPU-only preparation (window audio, VAD) of the turn's ASR windows while restoration still runs.
        Failures only mean the batch prepares them itself later."""
        wanted: Dict[Tuple[str, tuple], Tuple[LocalDeps, List[Tuple[float, float]]]] = {}
        asr = [t for t in units if t.stage == sp.STAGE_ASR]
        for cand, group in self._group_by_candidate(asr, self._candidate_of):
            if cand is None:
                continue
            deps = self._deps_for(cand)
            if deps.asr_prepare is None:
                continue
            todo = [t for t in group
                    if sr.read_asr(self.folder, self._asr_key(self.plan.slices[t.slice_index], cand)) is None]
            if not todo:
                continue
            todo.sort(key=lambda t: self.plan.slices[t.slice_index].start)
            w = self.asr_window
            windows = [(w.audio_start(self.plan.slices[t.slice_index]), self.plan.slices[t.slice_index].end)
                       for t in todo]
            wanted[(cand.id, tuple(windows))] = (deps, windows)
        for key in [k for k in self._prepared if k not in wanted]:
            shutil.rmtree(self._prepared.pop(key), ignore_errors=True)
        for key, (deps, windows) in wanted.items():
            if key in self._prepared:
                continue
            self._check()
            workdir = (self.folder / "cache" / "tmp"
                       / f"asr_prep_{int(self.clock() * 1000) % 10**9}_{len(self._prepared)}")
            t0 = self.clock()
            try:
                self._progress(state="asr", stage=sp.STAGE_ASR, detail="準備辨識音訊中")
                deps.asr_prepare(windows, workdir)
            except Exception as exc:  # noqa: BLE001
                self._check()          # cancelled (its child was killed): not a failure
                shutil.rmtree(workdir, ignore_errors=True)
                self._event("prepare_failed", error=str(exc)[-200:])
                continue
            self._prepared[key] = workdir
            self._event("prepared", windows=len(windows), wall_s=round(self.clock() - t0, 3))

    def _drop_prepared(self) -> None:
        for key in list(self._prepared):
            shutil.rmtree(self._prepared.pop(key), ignore_errors=True)

    def run(self, *, wait: bool = False) -> RunResult:
        """Work until the plan is complete (wait=True: also sleeps through restore turns, GPU busy, download and
        retry waits), or until nothing can run right now (wait=False; RunResult.reason says why: restore_turn,
        gpu_busy, download, quota (= local retry wait), stuck, idle)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        try:
            self._check()
            try:
                if not self._prepare():
                    return self._result("source_unavailable")
            except Exception as exc:  # noqa: BLE001
                if self._source_gone(exc):
                    return self._result("source_unavailable")
                raise
            self._usage_run_start()
            self._publish()
            self._publish_tracks()
            self._resolve_mode()
            spins = 0
            while True:
                self._check()
                if self.host.seek_pending():
                    self._turn_s = self.turns.turn_first_s
                    self.sleep(min(self.poll_s, 0.1))
                    continue
                usable, limit, _total = self.host.source_state()
                if not usable:
                    return self._result("source_unavailable", limit=limit)
                try:
                    advanced = self._advance_cuts(limit)
                except Exception as exc:  # noqa: BLE001
                    if not self._source_gone(exc):
                        raise
                    return self._result("source_unavailable", limit=limit)
                if self._refresh.is_set():      # soft drain: the previous batch finished, start none
                    return self._result("refresh", limit=limit)
                outcome = self.step(limit)
                if outcome in ("batch", "demoted"):
                    spins = 0
                    continue
                if advanced and self._cuts_paced and outcome not in ("restore_turn", "gpu_busy"):
                    continue          # one cut per round: nothing to run yet, decide the next cut right away
                if outcome in ("restore_turn", "gpu_busy"):
                    self._usage_wait_started(outcome, limit)
                    if wait:
                        self.sleep(self.poll_s)
                        continue
                    return self._result(outcome, limit=limit)
                reason, until, other = self._idle_reason(limit)
                if reason == "complete":
                    self._publish()
                    self._publish_tracks()
                    return self._result("complete", limit=limit)
                spins = spins + 1 if (reason == "idle" and other is None and until is not None) else 0
                if spins > 5:
                    return self._result(reason, limit=limit, wait_until=until, waiting_executor=other)
                if wait and (reason in ("quota", "download") or (reason == "idle" and until is not None)):
                    self._progress(state="waiting_" + ("handoff" if reason == "idle" else reason), detail=reason)
                    pause = self.download_poll_s
                    if until is not None:
                        pause = max(0.01, min(until - self.clock(), self.download_poll_s))
                    self.sleep(pause)
                    continue
                return self._result(reason, limit=limit, wait_until=until, waiting_executor=other)
        except Cancelled:
            self._stop.set()
            self._release_mine()
            return self._result("cancelled")
        finally:
            self._end_gpu_turn()
            self._drop_prepared()
            self._release_store()
            self._usage_run_end()

    def _usage_wait_started(self, outcome: str, limit: Optional[float]) -> None:
        """Usage log, once per wait: the unit the selector would take next was handed to the local executor and
        has to wait for the GPU (gpu_busy: no lease; restore_turn: restoration keeps the GPU)."""
        try:
            from oh_streamer.subtitles import usage_log
            with self._lock:
                task = sp.select_next_task(self.plan, self._ctx(limit), self.EXECUTOR)
                st = self.plan.slices[task.slice_index].get(task.stage, task.track) if task else None
                if st is None or st.wait_executor != self.EXECUTOR_NAME:
                    return
                key = (task.slice_index, task.stage, task.track)
                waiting = FolderState.of(self.folder).waiting
                if key in waiting:
                    return
                waiting[key] = self.clock()
            usage_log.event(self._usage_id(), f"{self.EXECUTOR_NAME}_wait_started", slice=task.slice_index,
                            stage=task.stage, track=task.track, reason=outcome, candidate=st.candidate,
                            mode=self.mode)
        except Exception:
            pass

    def _release_mine(self) -> None:
        with self._lock:
            if self.plan is not None and self.plan.release_running(self.EXECUTOR_NAME):
                self._save()

    # -- batch ----------------------------------------------------------------------------------
    def _take(self, s: sp.Slice, stage: str, track: str) -> bool:
        st = s.get(stage, track)
        return (st.status == sp.PENDING and self._owner_ok(stage, track, st)
                and (st.retry_at is None or st.retry_at <= self.clock()))

    def _blocking_task(self, ctx: sp.SchedulerContext) -> Optional[sp.Task]:
        """The first unfinished primary slice from the play position, when its next stage can be taken now.
        The selector sweeps all ASR before any MT (right for the cloud, where ASR is the bottleneck); a GPU
        turn should rather extend the continuous coverage first (e.g. a translation left over by a
        demotion), because that is what sub_lead and playback wait for."""
        plan, track = self.plan, self.plan.primary
        for s in sp.slices_from_position(plan, ctx.play_pos):
            if s.all_done(track):
                continue
            stage = sp.STAGE_ASR if not s.is_done(sp.STAGE_ASR, track) else sp.STAGE_MT
            readable = ctx.max_available is None or s.end <= ctx.max_available + 0.5
            if self._take(s, stage, track) and (stage == sp.STAGE_MT or readable):
                return sp.Task(s.index, stage, track, False, "contiguous")
            return None
        return None

    def _preview_batch(self, n: int, limit: Optional[float], target_s: Optional[float] = None) -> List[sp.Task]:
        """Select without claiming work; used both for playback budgeting and the actual batch. target_s (media
        turns): keep adding units until the picked slices cover that many media seconds, instead of n units."""
        def enough() -> bool:
            if target_s is None:
                return len(picked) >= n
            return sum(self.plan.slices[i].end - self.plan.slices[i].start
                       for i in {t.slice_index for t in picked}) >= target_s

        shadow = sp.SlicePlan.from_dict(self.plan.to_dict())
        picked = []
        self._preview_oversize = False
        ctx = self._ctx(limit)
        first = self._blocking_task(ctx)
        if first is not None:
            shadow.mark_running(first.slice_index, first.stage, self.EXECUTOR_NAME, first.track)
            picked.append(first)
        if target_s is not None:
            # translations left over by an earlier (early-ended) turn go before more ASR: that is what the
            # viewer's subtitle lead waits for, and the model for them has to be loaded anyway
            track = self.plan.primary
            for sl_ in sp.slices_from_position(self.plan, ctx.play_pos):
                if enough():
                    break
                if (sl_.is_done(sp.STAGE_ASR, track) and not sl_.is_done(sp.STAGE_MT, track)
                        and all(t.slice_index != sl_.index for t in picked)
                        and self._take(sl_, sp.STAGE_MT, track)):
                    task = sp.Task(sl_.index, sp.STAGE_MT, track, False, "leftover")
                    shadow.mark_running(task.slice_index, task.stage, self.EXECUTOR_NAME, task.track)
                    picked.append(task)
        while not enough():
            task = sp.select_next_task(shadow, ctx, self.EXECUTOR)
            if task is None:
                break
            if target_s is not None and picked and all(t.slice_index != task.slice_index for t in picked):
                # fix I: a turn never takes a slice that pushes its work past the target by more than one normal
                # (short) slice; the slice waits for a turn of its own
                s_ = self.plan.slices[task.slice_index]
                have = sum(self.plan.slices[i].end - self.plan.slices[i].start for i in {t.slice_index for t in picked})
                if have + (s_.end - s_.start) > target_s + sp.SHORT_SLICE_SECONDS:
                    self._preview_oversize = True
                    break
            shadow.mark_running(task.slice_index, task.stage, self.EXECUTOR_NAME, task.track)
            picked.append(task)
        return picked

    def _run_batch(self, n: int, limit: Optional[float], *, urgent: bool = False,
                   target_s: Optional[float] = None) -> bool:
        """One batch. True = everything it took was finished (False: nothing to do, a failure, or the turn ended
        early because the restore lead would run too low - the translation then waits for the next turn)."""
        picked: List[sp.Task] = []
        self._turn_loads = {}
        with self._lock:
            for task in self._preview_batch(n, limit, target_s):
                if self._claim_task(task):        # candidate chosen + running (or deferred / failed)
                    picked.append(task)
            self._save()
        if not picked:
            return False
        for _ in picked:
            self._bump("launched")
        t0 = self.clock()
        self._event("batch_start", slices=[t.slice_index for t in picked], stages=[t.stage for t in picked],
                    urgent=urgent, mode=self.mode,
                    candidates=[getattr(self._candidate_of(t), "id", None) for t in picked])
        asr_s = mt_s = 0.0
        completed = False
        try:
            asr_tasks = [t for t in picked if t.stage == sp.STAGE_ASR]
            if asr_tasks:
                self._guard_point()
                ta = self.clock()
                self._run_asr_batch(asr_tasks)
                asr_s = self.clock() - ta
            mt_tasks: List[sp.Task] = []
            if (asr_tasks and self.turns.media_turns and self.mode == gt.MODE_INTERLEAVE
                    and self._early_end(picked)):
                # Finished ASR stays saved; the translation goes back to pending for the next turn.
                self._release_mine()
                self._event("turn_early_end", restore_lead_s=self.host.restore_info().lead_s)
                return False
            with self._lock:
                self._ctx(limit)
                for t in picked:
                    s = self.plan.slices[t.slice_index]
                    if t.stage == sp.STAGE_MT:
                        mt_tasks.append(t)
                    elif s.is_done(sp.STAGE_ASR, t.track) and self._take(s, sp.STAGE_MT, t.track):
                        unit = sp.Task(s.index, sp.STAGE_MT, t.track, t.urgent, t.reason)
                        if self._claim_task(unit):
                            mt_tasks.append(unit)
                            self._bump("launched")
                self._save()
            if mt_tasks:
                self._guard_point()
                tm = self.clock()
                self._run_mt_batch(mt_tasks)
                mt_s = self.clock() - tm
                mt_media = sum(self.plan.slices[i].end - self.plan.slices[i].start
                               for i in {t.slice_index for t in mt_tasks})
                if self.turns.media_turns and mt_media >= 10.0 and "mt" in self._turn_loads:
                    self._mt_spm = max(0.05, (mt_s - self._turn_loads["mt"]) / mt_media)   # warm translation speed
            completed = True
        except GpuOutOfMemory as exc:
            self._on_oom(picked, exc)
        finally:
            work_s = sum(self.plan.slices[t.slice_index].end - self.plan.slices[t.slice_index].start for t in picked)
            if work_s > 0 and self.mode == gt.MODE_INTERLEAVE:
                self._seconds_per_media_second = max(self._seconds_per_media_second,
                                                     max(0.0, self.clock() - t0 - self._load_total()) / work_s)
            if completed:
                wall = self.clock() - t0
                loaded = sum(self._turn_loads.values())
                # the budget adds the measured loads itself: the learned speed is the per-media-second work only
                self._record_speed(picked, max(wall - loaded, 0.2 * wall) if loaded > 0 else wall)
            self._event("batch_end", slices=[t.slice_index for t in picked], asr_s=round(asr_s, 3),
                        mt_s=round(mt_s, 3), wall_s=round(self.clock() - t0, 3))
        return completed

    def _on_oom(self, picked: Sequence[sp.Task], exc: GpuOutOfMemory) -> None:
        """Parallel: the running work goes back to pending (redone after the switch) and the mode is demoted.
        Interleave (alone on the GPU, e.g. a game holds VRAM): a counted retry like any local error."""
        if self.mode == gt.MODE_PARALLEL:
            self._release_mine()
            raise _Demoted("oom") from exc
        with self._lock:
            running = []
            for t in picked:
                for stage in sp.STAGES:
                    st = self.plan.slices[t.slice_index].get(stage, t.track)
                    unit = sp.Task(t.slice_index, stage, t.track)
                    if st.status == sp.RUNNING and st.executor == self.EXECUTOR_NAME and unit not in running:
                        running.append(unit)
        for unit in running:
            self._failed(unit, exc)

    @staticmethod
    def _group_by_candidate(tasks: Sequence[sp.Task], cand_of) -> List[Tuple[Any, List[sp.Task]]]:
        groups: Dict[str, Tuple[Any, List[sp.Task]]] = {}
        for t in tasks:
            cand = cand_of(t)
            cid = getattr(cand, "id", "")
            groups.setdefault(cid, (cand, []))[1].append(t)
        return list(groups.values())

    def _run_asr_batch(self, tasks: Sequence[sp.Task]) -> None:
        for cand, group in self._group_by_candidate(tasks, self._candidate_of):
            self._run_asr_group(cand, group)

    def _run_asr_group(self, cand, tasks: Sequence[sp.Task]) -> None:
        """One ASR pass (models load once) over the windows of the units on one local candidate."""
        deps = self._deps_for(cand)
        todo: List[sp.Task] = []
        for t in tasks:
            s = self.plan.slices[t.slice_index]
            key = self._asr_key(s, cand)
            doc = sr.read_asr(self.folder, key)
            if doc is not None:
                with self._lock:
                    self.plan.mark_done(s.index, sp.STAGE_ASR, str(doc.get("model_id") or ""), t.track,
                                        candidate=cand.id, result_key=key)
                    self._save()
                self._usage_unit_done(t)
            else:
                todo.append(t)
        if not todo:
            return
        todo.sort(key=lambda t: self.plan.slices[t.slice_index].start)
        groups = [[t] for t in todo]        # canonical per-slice VAD, independent of batch size / seek order
        w = self.asr_window
        windows = [(w.audio_start(self.plan.slices[g[0].slice_index]), self.plan.slices[g[-1].slice_index].end)
                   for g in groups]
        self._check()
        self._progress(state="asr", stage=sp.STAGE_ASR, slice=[t.slice_index for t in todo],
                       detail="載入辨識模型中", candidate=cand.id)
        workdir = (self._prepared.pop((cand.id, tuple(windows)), None)       # prepared before the lease (CPU part done)
                   or self.folder / "cache" / "tmp" / f"asr_batch_{todo[0].slice_index:04d}_{int(self.clock() * 1000) % 10**9}")
        if deps.load_probe is not None:
            deps.load_probe.begin()
        try:
            results = deps.asr_batch(windows, workdir)
            if (len(results) != len(windows)
                    or any(not isinstance(r, dict) or not isinstance(r.get("segments"), list) for r in results)):
                raise RuntimeError("ASR batch returned incomplete window results")
        except (GpuOutOfMemory, InterruptedError):
            raise
        except Exception as exc:
            self._check()          # cancelled (its model process was killed): not a counted failure
            for t in todo:
                self._failed(t, exc)
            return
        self._check()
        if deps.load_probe is not None:
            self._learn_load("asr", deps.load_probe.take())
        for g, (a, b), res in zip(groups, windows, results):
            if res.get("error"):
                for t in g:
                    self._failed(t, RuntimeError(str(res["error"])))
                continue
            for j, t in enumerate(g):
                s = self.plan.slices[t.slice_index]
                lead = w.lead_keep_s if j == 0 else w.tail_guard_s
                owned = sp.own_segments(res.get("segments") or [], s.start, s.end, first=s.start <= 0.0,
                                        last=sp.is_last_slice(self.plan, s), lead_keep_s=lead,
                                        tail_guard_s=w.tail_guard_s)
                lo = min([s.start] + [float(x["start"]) for x in owned])
                speech = [[round(x - s.start, 4), round(y - s.start, 4)]
                          for x, y in (_interval(r) for r in res.get("speech_intervals") or []) if y > lo and x < s.end]
                key = self._asr_key(s, cand)
                doc = {"media_id": self._media_id, "start": s.start, "end": s.end,
                       "model": self._identity(sp.STAGE_ASR, cand), **self._result_meta(sp.STAGE_ASR, cand),
                       "model_id": res.get("model") or deps.asr_model_id, "segments": _shift(owned, -s.start),
                       "speech_intervals": speech, "warnings": res.get("warnings") or [], "audio_start": a,
                       "batch_window": [a, b], "window": w.cache_params()}
                sr.write_asr(self.folder, key, doc)
                with self._lock:
                    self.plan.mark_done(s.index, sp.STAGE_ASR, str(doc["model_id"]), t.track, candidate=cand.id,
                                        result_key=key)
                    self._save()
                self._usage_unit_done(t)
                self._progress(state="done", slice=s.index, stage=sp.STAGE_ASR, candidate=cand.id)

    def _run_mt_batch(self, tasks: Sequence[sp.Task]) -> None:
        for cand, group in self._group_by_candidate(tasks, self._candidate_of):
            probe = self._deps_for(cand).load_probe
            if probe is not None:
                probe.begin()
            with self._deps_for(cand).mt_session() as translate:     # one llama-server per candidate
                for t in group:
                    self._check()
                    self._guard_point()
                    try:
                        self._run_mt_local(t, translate, cand)
                        self._usage_unit_done(t)
                        self._progress(state="done", slice=t.slice_index, stage=sp.STAGE_MT, candidate=cand.id)
                    except (GpuOutOfMemory, InterruptedError, _Demoted):
                        raise
                    except Exception as exc:
                        self._check()      # cancelled (llama-server was killed): not a counted failure
                        self._failed(t, exc)
                if probe is not None:
                    self._learn_load("mt", probe.take())

    def _run_mt_local(self, task: sp.Task, translate: TranslateFn, cand) -> None:
        s = self.plan.slices[task.slice_index]
        deps = self._deps_for(cand)
        asr_key, asr_doc = self._asr_doc(s, task.track)
        if asr_doc is None:
            raise RuntimeError(f"ASR result missing for slice {s.index}")
        digest = self._digest_of(asr_key, asr_doc["segments"])
        done_rows, partial_rows = self._rows(task.track), self._partials(task.track)
        key = self._mt_key(digest, cand, task.track)
        doc = sr.read_mt(self.folder, key)
        if doc is not None:
            rows, model_id = list(doc["segments"]), str(doc.get("model_id") or "")
        else:
            self._progress(state="mt", slice=s.index, stage=sp.STAGE_MT, detail="翻譯中", candidate=cand.id)
            ja_rows = _shift(asr_doc["segments"], s.start)                     # absolute for translate_file
            speech = [[x + s.start, y + s.start] for x, y in (_interval(r) for r in asr_doc.get("speech_intervals") or [])]
            ja_doc = {"segments": ja_rows, "raw_segments": ja_rows, "speech_intervals": speech,
                      "model": asr_doc.get("model_id")}

            def publish_batch(out_rows):
                self._check()
                self._guard_point()
                partial = _shift([display_row(r) for r in out_rows], -s.start)
                sr.write_partial(self.folder, key, partial)
                with self._lock:
                    partial_rows[s.index] = partial
                self._publish_for(task.track)

            kwargs = {"on_batch": publish_batch} if deps.mt_batch_callback else {}
            if task.track != self.plan.primary and deps.mt_sentence_option:
                kwargs["sentence"] = dict(self._track_config(task.track).get("sentence") or {})    # the row's own
            out_rows = translate(ja_doc, self.folder / "cache" / "tmp" / f"mt_{key[:16]}", **kwargs)
            rows = _shift([display_row(r) for r in out_rows], -s.start)     # display rule on absolute times
            model_id = deps.mt_model_id
            source = self._row_identity(sp.STAGE_MT, cand)
            for r in rows:
                r.setdefault("translation_source", dict(source))
            sr.write_mt(self.folder, key, {"media_id": self._media_id, "start": s.start, "end": s.end,
                                           "model": self._identity(sp.STAGE_MT, cand),
                                           **self._result_meta(sp.STAGE_MT, cand), "asr_digest": digest,
                                           "model_id": model_id, "models_used": [model_id], "segments": rows})
        with self._lock:
            done_rows[s.index] = rows
            partial_rows.pop(s.index, None)
            self.plan.mark_done(s.index, sp.STAGE_MT, model_id, task.track, candidate=cand.id, result_key=key)
            self._save()
        sr.clear_partial(self.folder, key)
        self._publish_for(task.track)

    # failure policy: CloudSliceExecutor._failed (CandidateChain.decide with the local back-off and kinds
    # error / oom). Local errors are retried RETRY_WAIT_S * attempt up to MAX_ATTEMPTS, then the next
    # candidate (e.g. the cloud backup: wait_executor "cloud") or failed(exhausted). A busy GPU never gets
    # here (D1).


# --------------------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------------------

def run_local_subtitles(folder: Path | str, source: Path | str, *, config: dict, settings: Optional[dict] = None,
                        host: Optional[GpuHost] = None, deps: Optional[LocalDeps] = None, wait: bool = True,
                        **kwargs: Any) -> Tuple[RunResult, LocalSliceExecutor]:
    """Run the local executor on one item folder (``<item>/subtitles``) until done (wait=True) or until nothing
    can run now. Blocks; the caller picks the thread. `config` = the primary subtitle config."""
    executor = LocalSliceExecutor(folder, source, host=host, deps=deps, config=config, settings=settings, **kwargs)
    return executor.run(wait=wait), executor
