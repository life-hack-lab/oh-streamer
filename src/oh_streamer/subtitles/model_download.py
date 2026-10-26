"""On-demand download of the local-subtitle models (stdlib only, no UI).

The catalog pins every file (URL at a fixed revision, size, SHA-256). The layout under the models directory is what
``subtitles/environment.py detect()`` expects::

    <models>/<asr-alias>/config.json + model.safetensors + tokenizer files
    <models>/sakura/<file>.gguf
    <models>/llama-cpp/**/llama-server.exe

Each file streams to ``<dest>.part`` (HTTP Range resume), is verified (size + SHA-256) and only then published with
an atomic ``os.replace``. Zip items are downloaded to ``<models>/.downloads/``, verified, extracted into a temp dir
inside the models dir (members escaping it are rejected) and moved in. ``<models>/downloads.json`` records the item
ids and revisions that finished. Nothing here runs at import time or connects to the network by itself.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

USER_AGENT = "oh-streamer-model-download/1"
RECORD_NAME = "downloads.json"
DOWNLOAD_DIR = ".downloads"          # verified zips wait here until extracted
CHUNK = 1 << 20
HTTP_TIMEOUT_S = 30.0
MAX_ATTEMPTS = 4
BACKOFF_S = 2.0
SPEED_WINDOW_S = 8.0
MIN_MARGIN = 512 * 2**20             # free space kept on top of what a plan needs
MARGIN_FRACTION = 0.02

_HF = "https://huggingface.co"
_LLAMA = "https://github.com/ggml-org/llama.cpp/releases/download/b11617"


# ------------------------------------------------------------------------------------------------ catalog types

@dataclass(frozen=True)
class FileSpec:
    url: str
    dest: str                         # relative to the models dir, posix separators
    size: int
    sha256: str
    extract_to: str = ""              # non-empty: dest is a zip, extracted into this models-relative folder
    unpacked: int = 0                 # zip only: estimated bytes after extraction (for the space check)


@dataclass(frozen=True)
class CatalogItem:
    id: str
    label: str                        # zh-TW
    revision: str
    default: bool
    files: tuple[FileSpec, ...]
    marker: str = ""                  # zip items: glob (relative to models dir) that must match once extracted
    note: str = ""
    category: str = ""                # "asr" | "translation" | "runtime"
    description: str = ""             # short zh-TW hint shown in the download card
    license_id: str = ""              # SPDX-ish id, or "unspecified" when the model card declares none
    license_label: str = ""           # zh-TW, shown before the user accepts the download
    card_url: str = ""                # https model card / project page, opened in the system browser

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


class DownloadError(Exception):
    """Carries a zh-TW message fit to show the user."""


class InsufficientSpace(DownloadError):
    pass


class _Cancelled(Exception):
    pass


def _hf(repo, rev, names):
    return [(f"{_HF}/{repo}/resolve/{rev}/{n}", n) for n in names]


def _specs(repo, rev, prefix, rows):
    """rows: (path, size, sha256)."""
    return tuple(FileSpec(f"{_HF}/{repo}/resolve/{rev}/{p}", f"{prefix}/{p}", size, sha) for p, size, sha in rows)


_WHISPER_15B = _specs("efwkjn/whisper-ja-1.5B", "72c13fb7b6cc2a831780cdd79cd1a829e77bab02", "whisper-ja-1.5B", [
    ("added_tokens.json", 34648, "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"),
    ("config.json", 1272, "ad0e8d1e46f4d01f7861a21509e5d0f977d6cc1f367a370603c92541d819807b"),
    ("generation_config.json", 3870, "c062cb67fa928171919cf0d05d42830dbf61eff2d442c70c21d6cd4232709145"),
    ("merges.txt", 493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"),
    ("normalizer.json", 52666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"),
    ("preprocessor_config.json", 340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
    ("tokenizer.json", 2480617, "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca"),
    ("tokenizer_config.json", 282843, "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389"),
    ("vocab.json", 1036558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"),
    ("model.safetensors", 4900174632, "0bf92cf794ebb2cef2699c24dc71184be6a1ac133cc3c64f228ea08173d0155f"),
])

_ANIME_WHISPER = _specs("litagin/anime-whisper", "22e2008a8182b357da3922a6308d095008f72973", "anime-whisper", [
    ("added_tokens.json", 34648, "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"),
    ("config.json", 1268, "1b0bf54206d648763dbadbab3fc7ff0577b0f80224f002dbbe8d8c2caa147360"),
    ("generation_config.json", 3898, "4e5bccc702f2322fcb497967cff54f890dcb3053c2993884e44313655fa951e0"),
    ("merges.txt", 493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"),
    ("normalizer.json", 52666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"),
    ("preprocessor_config.json", 340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
    ("special_tokens_map.json", 2186, "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74"),
    ("tokenizer_config.json", 282863, "c86c7398ba7f81f784569ea0ae165e252c4a006acc56906f04e5ee3300cf043e"),
    ("vocab.json", 1036558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"),
    ("model.safetensors", 3025686376, "15c672f0bf687b1c67aa14325f9c382c6919f1fce976f990513618b464a3c626"),
])

_REV_G7 = "759d76b1745f4428308de6564c5ab710d4358b2e"
_REV_G14 = "10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e"
_REV_Q7 = "24393225dfab4750c6fb9ec2468fb76ad3d2dd1d"

CATALOG: tuple[CatalogItem, ...] = (
    CatalogItem("whisper-ja-1.5B", "日文語音辨識模型 whisper-ja-1.5B", "72c13fb7b6cc2a831780cdd79cd1a829e77bab02", True,
                _WHISPER_15B, category="asr",
                description="日文語音辨識，通用、穩定。",
                license_id="unspecified", license_label="模型卡未標示授權，請自行確認使用條件",
                card_url="https://huggingface.co/efwkjn/whisper-ja-1.5B"),
    CatalogItem("anime-whisper", "日文語音辨識模型 anime-whisper", "22e2008a8182b357da3922a6308d095008f72973", True,
                _ANIME_WHISPER, category="asr",
                description="動畫／口語風格的日文語音辨識。",
                license_id="MIT", license_label="MIT",
                card_url="https://huggingface.co/litagin/anime-whisper"),
    CatalogItem("sakura-galtransl-7b-v3.7", "日翻中模型 Sakura-GalTransl 7B v3.7（建議）", _REV_G7, True, (
        FileSpec(f"{_HF}/SakuraLLM/Sakura-GalTransl-7B-v3.7/resolve/{_REV_G7}/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
                 "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf", 4250296192,
                 "8f515bf4769f279a7fcf43e57446455a9d4de7f65b1bc9eddee76717e1ff7919"),), category="translation",
                description="預設翻譯模型，速度與品質平衡。",
                license_id="CC-BY-NC-SA-4.0", license_label="CC BY-NC-SA 4.0（僅限非商業用途）；模型卡另禁止付費翻譯服務與付費字幕／補丁",
                card_url="https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7"),
    CatalogItem("llama-server", "翻譯執行檔 llama-server（NVIDIA CUDA 12.4，Windows）", "b11617", True, (
        FileSpec(f"{_LLAMA}/cudart-llama-bin-win-cuda-12.4-x64.zip", f"{DOWNLOAD_DIR}/cudart-llama-bin-win-cuda-12.4-x64.zip",
                 391443627, "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
                 extract_to="llama-cpp", unpacked=574138880),
        FileSpec(f"{_LLAMA}/llama-b11617-bin-win-cuda-12.4-x64.zip", f"{DOWNLOAD_DIR}/llama-b11617-bin-win-cuda-12.4-x64.zip",
                 265266505, "46d94e8676933a0eb0ad8c80ba1935001c193ccd745ae8372a90678675366dfb",
                 extract_to="llama-cpp", unpacked=622256413)), marker="llama-cpp/llama-server.exe", category="runtime",
                description="執行翻譯模型用的程式，必要。",
                license_id="MIT", license_label="MIT（llama.cpp）；內含 NVIDIA CUDA 執行階段，適用 NVIDIA 授權條款",
                card_url="https://github.com/ggml-org/llama.cpp"),
    CatalogItem("sakura-galtransl-14b-v3.8", "日翻中模型 Sakura-GalTransl 14B v3.8（選用，品質較高、需要更多顯示記憶體）", _REV_G14, False, (
        FileSpec(f"{_HF}/SakuraLLM/Sakura-GalTransl-14B-v3.8/resolve/{_REV_G14}/Sakura-Galtransl-14B-v3.8-IQ4_XS.gguf",
                 "sakura/Sakura-Galtransl-14B-v3.8-IQ4_XS.gguf", 8186193216,
                 "4762c2600d3b358dc913ef94b71f6f6b24010a2b73fb82786d21c72263c74573"),), category="translation",
                description="較準，需要較多顯示記憶體。",
                license_id="CC-BY-NC-SA-4.0", license_label="CC BY-NC-SA 4.0（僅限非商業用途）",
                card_url="https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8"),
    CatalogItem("sakura-7b-qwen2.5-v1.0", "日翻中模型 Sakura 7B Qwen2.5 v1.0（選用）", _REV_Q7, False, (
        FileSpec(f"{_HF}/SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF/resolve/{_REV_Q7}/sakura-7b-qwen2.5-v1.0-iq4xs.gguf",
                 "sakura/sakura-7b-qwen2.5-v1.0-iq4xs.gguf", 4250298208,
                 "30e82ca13bc1e321f4f7b77459d533d483d46200522dc5c31eef9d2f2a6335d6"),), category="translation",
                description="舊版。",
                license_id="CC-BY-NC-SA-4.0", license_label="CC BY-NC-SA 4.0（僅限非商業用途）",
                card_url="https://huggingface.co/SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF"),
)


def catalog() -> tuple[CatalogItem, ...]:
    return CATALOG


def default_item_ids(items: Sequence[CatalogItem] | None = None) -> list[str]:
    return [i.id for i in (items or CATALOG) if i.default]


def _lookup(item_ids: Iterable[str], items: Sequence[CatalogItem] | None) -> list[CatalogItem]:
    table = {i.id: i for i in (items or CATALOG)}
    out = []
    for item_id in item_ids:
        if item_id not in table:
            raise DownloadError(f"未知的下載項目：{item_id}")
        if table[item_id] not in out:
            out.append(table[item_id])
    return out


def _safe_rel(models: Path, rel: str) -> Path:
    pure = PurePosixPath(rel)
    if not rel or pure.is_absolute() or ".." in pure.parts or ":" in rel.split("/")[0]:
        raise DownloadError(f"不安全的目的路徑：{rel}")
    return models.joinpath(*pure.parts)


# -------------------------------------------------------------------------------------------------- record file

_record_lock = threading.Lock()


def _read_record(models: Path) -> dict:
    try:
        data = json.loads((models / RECORD_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("items"), dict) else {"version": 1, "items": {}}
    except (OSError, ValueError):
        return {"version": 1, "items": {}}


def _write_record(models: Path, record: dict) -> None:
    tmp = models / (RECORD_NAME + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, models / RECORD_NAME)


def _mark_installed(models: Path, item: CatalogItem) -> None:
    with _record_lock:
        record = _read_record(models)
        record["version"] = 1
        record["items"][item.id] = {"revision": item.revision, "installed_at": int(time.time())}
        (record.get("partial") or {}).pop(item.id, None)       # the half-way record is obsolete once installed
        _write_record(models, record)


def _extracted_dests(models: Path, item: CatalogItem) -> set[str]:
    """Zip dests of ``item`` already extracted by an interrupted run (same revision only)."""
    rec = (_read_record(models).get("partial") or {}).get(item.id)
    if not isinstance(rec, dict) or rec.get("revision") != item.revision:
        return set()
    done = rec.get("extracted")
    return {d for d in done if isinstance(d, str)} if isinstance(done, list) else set()


def _mark_extracted(models: Path, item: CatalogItem, spec: FileSpec) -> None:
    with _record_lock:
        record = _read_record(models)
        record["version"] = 1
        partial = record.get("partial")
        if not isinstance(partial, dict):
            partial = record["partial"] = {}
        rec = partial.get(item.id)
        if not isinstance(rec, dict) or rec.get("revision") != item.revision:
            rec = partial[item.id] = {"revision": item.revision, "extracted": []}
        if spec.dest not in rec["extracted"]:
            rec["extracted"].append(spec.dest)
        _write_record(models, record)


def _file_ok(models: Path, spec: FileSpec) -> bool:
    try:
        return _safe_rel(models, spec.dest).stat().st_size == spec.size
    except (OSError, DownloadError):
        return False


def _item_installed(models: Path, item: CatalogItem) -> bool:
    if item.marker:   # zip item: the zips are deleted after extraction, so the record plus the marker decide
        rec = _read_record(models)["items"].get(item.id)
        return bool(rec and rec.get("revision") == item.revision and (models / item.marker).is_file())
    return all(_file_ok(models, f) for f in item.files)


def _part_size(models: Path, spec: FileSpec) -> int:
    try:
        return min(Path(str(_safe_rel(models, spec.dest)) + ".part").stat().st_size, spec.size)
    except (OSError, DownloadError):
        return 0


def _item_remaining(models: Path, item: CatalogItem) -> int:
    if _item_installed(models, item):
        return 0
    done = _extracted_dests(models, item)
    return sum(0 if (f.dest in done or _file_ok(models, f)) else f.size - _part_size(models, f) for f in item.files)


def status(models_dir, items: Sequence[CatalogItem] | None = None) -> list[dict]:
    models = Path(models_dir)
    rows = []
    for item in (items or CATALOG):
        installed = _item_installed(models, item)
        done = set() if installed else _extracted_dests(models, item)
        partial = 0 if installed else sum(f.size if (f.dest in done or _file_ok(models, f)) else _part_size(models, f)
                                          for f in item.files)
        rows.append(dict(id=item.id, label=item.label, revision=item.revision, default=item.default, size=item.size,
                         installed=installed, partial_bytes=partial, remaining_bytes=_item_remaining(models, item),
                         note=item.note, category=item.category, description=item.description,
                         license_id=item.license_id, license_label=item.license_label, card_url=item.card_url))
    return rows


# ------------------------------------------------------------------------------------------------------- plan

def _fmt_bytes(n: int) -> str:
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def plan(models_dir, item_ids: Iterable[str], items: Sequence[CatalogItem] | None = None,
         free_bytes: int | None = None, margin: int | None = None) -> dict:
    """Bytes still to fetch, extra space for zip extraction, free disk; ``ok`` False when it would not fit."""
    models = Path(models_dir)
    chosen = _lookup(item_ids, items)
    download = sum(_item_remaining(models, i) for i in chosen)
    extract = sum(f.unpacked for i in chosen if not _item_installed(models, i)
                  for f in i.files if f.extract_to and f.dest not in _extracted_dests(models, i))
    needed = download + extract
    free = _free_bytes(models) if free_bytes is None else free_bytes
    margin = max(MIN_MARGIN, int(needed * MARGIN_FRACTION)) if margin is None else margin
    ok = free >= needed + margin
    message = "" if ok else (f"磁碟空間不足：需要約 {_fmt_bytes(needed + margin)}（含解壓縮與保留空間），"
                             f"目前只剩 {_fmt_bytes(free)}。請清出空間後再試。")
    return dict(items=[i.id for i in chosen], download_bytes=download, extract_bytes=extract, needed_bytes=needed,
                margin_bytes=margin, free_bytes=free, ok=ok, message=message)


def check_selection(models_dir, item_ids: Iterable[str], items: Sequence[CatalogItem] | None = None) -> dict:
    """Selection rules: >= 1 ASR and >= 1 translation model (selected or already installed); the runtime
    (llama-server) is mandatory and added automatically when missing. Returns ``ids`` (with the runtime added),
    ``ok`` and a zh-TW ``message`` for the first broken rule."""
    models = Path(models_dir)
    table = list(items or CATALOG)
    chosen = _lookup(item_ids, table)
    ids = [i.id for i in chosen]
    for item in table:
        if item.category == "runtime" and item.id not in ids and not _item_installed(models, item):
            ids.append(item.id)
    covered = {i.category for i in table if i.id in ids or _item_installed(models, i)}
    message = ""
    if "asr" not in covered:
        message = "請至少選擇一個語音辨識模型。"
    elif "translation" not in covered:
        message = "請至少選擇一個翻譯模型。"
    return dict(ids=ids, ok=not message, message=message)


# --------------------------------------------------------------------------------------------- download engine

def _sha256_file(path: Path, cancel: threading.Event, progress=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel.is_set():
                raise _Cancelled()
            block = f.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)
            if progress:
                progress(len(block))


class _Transient(Exception):
    pass


def _fetch_into(spec: FileSpec, part: Path, cancel: threading.Event, on_bytes) -> None:
    """One HTTP attempt: resume from ``part`` if the server honours Range. Raises _Transient on network trouble."""
    have = part.stat().st_size if part.exists() else 0
    if have > spec.size:
        part.unlink()
        have = 0
    if have == spec.size:
        return
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(spec.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:       # our .part does not fit the file: start over
            part.unlink(missing_ok=True)
            on_bytes(-have)
            raise _Transient("range rejected")
        if exc.code in (408, 425, 429) or exc.code >= 500:
            raise _Transient(f"HTTP {exc.code}")
        raise DownloadError(f"伺服器拒絕下載（HTTP {exc.code}）：{spec.dest}")
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise _Transient(str(exc))
    with response:
        if have and response.status != 206:    # server ignored Range: restart from zero
            on_bytes(-have)
            have = 0
        mode = "ab" if have else "wb"
        try:
            with open(part, mode) as out:
                while True:
                    if cancel.is_set():
                        raise _Cancelled()
                    block = response.read1(1 << 16)
                    if not block:
                        break
                    out.write(block)
                    on_bytes(len(block))
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
                raise DownloadError("磁碟空間不足，下載中斷。")
            raise _Transient(str(exc))


def _safe_extract(zip_path: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename
            pure = PurePosixPath(name.replace("\\", "/"))
            target = (dest / Path(*pure.parts)).resolve() if pure.parts else root
            if pure.is_absolute() or ".." in pure.parts or ":" in name or not (target == root or root in target.parents):
                raise DownloadError(f"壓縮檔含有不安全的路徑，已拒絕解壓縮：{name}")
        archive.extractall(dest)


def _move_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if path.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, dst / rel)


class DownloadJob:
    """Downloads the chosen catalog items on a worker thread; poll ``snapshot()`` for progress."""

    def __init__(self, models_dir, item_ids: Iterable[str], items: Sequence[CatalogItem] | None = None,
                 free_bytes: int | None = None, backoff: float | None = None, attempts: int = MAX_ATTEMPTS):
        self.models = Path(models_dir)
        self._items = _lookup(item_ids, items)
        self._free_bytes = free_bytes
        self._backoff = BACKOFF_S if backoff is None else backoff
        self._attempts = attempts
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state = "queued"
        self._current = ""
        self._current_label = ""
        self._error = ""
        self._done = 0
        self._total = sum(i.size for i in self._items)
        self._samples: deque = deque()
        self._started = 0.0

    # -- public
    def start(self) -> "DownloadJob":
        if self._thread:
            return self
        check = plan(self.models, [i.id for i in self._items], self._items, free_bytes=self._free_bytes)
        if not check["ok"]:
            self._set(state="failed", error=check["message"])
            return self
        self._thread = threading.Thread(target=self._run, name="subtitle-model-download", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            while self._samples and now - self._samples[0][0] > SPEED_WINDOW_S:
                self._samples.popleft()
            speed = 0.0
            if len(self._samples) >= 2 and self._samples[-1][0] > self._samples[0][0]:
                speed = (self._samples[-1][1] - self._samples[0][1]) / (self._samples[-1][0] - self._samples[0][0])
            speed = max(speed, 0.0) if self._state == "running" else 0.0
            remaining = max(self._total - self._done, 0)
            eta = remaining / speed if speed > 1 else None
            return dict(state=self._state, item=self._current, item_label=self._current_label,
                        bytes_done=self._done, bytes_total=self._total, speed=speed, eta=eta, error=self._error)

    # -- internals
    def _set(self, **kw):
        with self._lock:
            for key, value in kw.items():
                setattr(self, "_" + key, value)

    def _add(self, n: int):
        with self._lock:
            self._done = max(self._done + n, 0)
            self._samples.append((time.monotonic(), self._done))

    def _run(self) -> None:
        try:
            self.models.mkdir(parents=True, exist_ok=True)
            self._set(state="running")
            for item in self._items:
                self._install(item)
            self._set(state="done", current="", current_label="")
        except _Cancelled:
            self._set(state="cancelled")
        except DownloadError as exc:
            self._set(state="failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected as a failed job
            self._set(state="failed", error=f"下載失敗：{exc}")

    def _install(self, item: CatalogItem) -> None:
        self._set(current=item.id, current_label=item.label)
        if _item_installed(self.models, item):
            self._add(item.size)
            return
        done = _extracted_dests(self.models, item)
        for spec in item.files:
            if spec.extract_to and spec.dest in done:      # extracted by an interrupted run: no download, no extract
                self._add(spec.size)
                continue
            self._ensure_file(spec)
            if spec.extract_to:
                self._set(state="verifying")
                self._extract(item, [spec])
                self._set(state="running")
        _mark_installed(self.models, item)

    def _ensure_file(self, spec: FileSpec) -> None:
        final = _safe_rel(self.models, spec.dest)
        if final.is_file() and final.stat().st_size == spec.size:
            self._add(spec.size)      # only ever published after size + SHA-256 passed
            return
        part = final.with_name(final.name + ".part")
        final.parent.mkdir(parents=True, exist_ok=True)
        counted = part.stat().st_size if part.exists() else 0
        self._add(min(counted, spec.size))
        on_bytes = self._add

        for attempt in range(1, self._attempts + 1):
            if self._cancel.is_set():
                raise _Cancelled()
            try:
                _fetch_into(spec, part, self._cancel, on_bytes)
                break
            except _Transient as exc:
                if attempt == self._attempts:
                    raise DownloadError(f"網路連線失敗，已重試 {self._attempts} 次（{exc}）。已下載的部分會保留，稍後可從中斷處繼續。")
                waited = 0.0
                while waited < self._backoff * attempt:
                    if self._cancel.is_set():
                        raise _Cancelled()
                    time.sleep(min(0.1, self._backoff * attempt))
                    waited += 0.1
        self._set(state="verifying")
        size = part.stat().st_size
        if size != spec.size:
            part.unlink(missing_ok=True)
            raise DownloadError(f"檔案大小不符（{spec.dest}），已捨棄，請重新下載。")
        digest = _sha256_file(part, self._cancel)
        if digest != spec.sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(f"檔案校驗碼不符（{spec.dest}），已捨棄損毀的檔案，請重新下載。")
        os.replace(part, final)
        self._set(state="running")

    def _extract(self, item: CatalogItem, zips: list[FileSpec]) -> None:
        for spec in zips:
            if self._cancel.is_set():
                raise _Cancelled()
            archive = _safe_rel(self.models, spec.dest)
            if not archive.is_file():
                continue          # already extracted in an earlier run
            try:
                if zipfile.ZipFile(archive).testzip() is not None:
                    raise zipfile.BadZipFile("crc")
            except zipfile.BadZipFile:
                archive.unlink(missing_ok=True)
                raise DownloadError(f"壓縮檔損毀（{archive.name}），已捨棄，請重新下載。")
            tmp = Path(tempfile.mkdtemp(prefix=".extract-", dir=self.models))
            try:
                _safe_extract(archive, tmp)
                dest = _safe_rel(self.models, spec.extract_to)
                dest.mkdir(parents=True, exist_ok=True)
                _move_tree(tmp, dest)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            archive.unlink(missing_ok=True)
            _mark_extracted(self.models, item, spec)
        try:
            (self.models / DOWNLOAD_DIR).rmdir()
        except OSError:
            pass
