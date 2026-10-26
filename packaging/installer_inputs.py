"""Installer input manifest: download, verify and stage the external resources of the installer.

Reads ``packaging/installer_inputs.json`` and fills a cache folder (default
``<worktree>/build-installer/cache``)::

    cache/archives/<filename>        downloaded archives (.zip / .tar.gz / .7z / .exe bundle), SHA-256 verified
    cache/files/<dest>               every single file, laid out like the install root, SHA-256 verified
    cache/trees/<id>/                whole-archive trees (embedded Python), with a .complete marker

Sources of a manifest file entry:
    url      downloaded directly
    archive  one member extracted from an entry of ``archives`` (zip; 7z via Windows tar.exe or 7-Zip;
             'burn' = cabinets inside a WiX Burn bundle such as vc_redist.x64.exe, member '<container>/<name>',
             opened with Windows expand.exe without running the bundle)
    local    copied from a development-machine path (used when no official archive reproduces the hash)

Standard library only.

    python packaging/installer_inputs.py fetch [--cache DIR] [--local-dir DIR]
    python packaging/installer_inputs.py verify [--cache DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "packaging" / "installer_inputs.json"
DEFAULT_CACHE = ROOT / "build-installer" / "cache"
MANIFEST_FORMAT = "oh-streamer-installer-inputs-v1"
LOCAL_ENV = "OH_STREAMER_LOCAL_INPUTS"
USER_AGENT = "oh-streamer-installer-inputs/1"
COMPLETE_MARKER = ".complete"

Opener = Callable[[str], Any]


class InputsError(Exception):
    """Any manifest, download, hash or extraction problem."""


# --------------------------------------------------------------------------- hashing / paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_root() -> Path:
    """Directory that holds the main checkout's siblings (``testdata`` etc.): parent of ``worktrees``."""
    parent = ROOT.parent
    return parent.parent if parent.name == "worktrees" else parent


def safe_relpath(name: str) -> PurePosixPath:
    """Normalise an archive member / destination name; reject absolute paths, drives and ``..``."""
    text = str(name).replace("\\", "/")
    if not text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise InputsError(f"unsafe path (absolute or drive): {name!r}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." or ":" in part for part in parts):
        raise InputsError(f"unsafe path (traversal): {name!r}")
    return PurePosixPath(*parts)


def _join_safe(base: Path, rel: PurePosixPath) -> Path:
    target = (base / Path(*rel.parts)).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise InputsError(f"path escapes target folder: {rel}")
    return target


# --------------------------------------------------------------------------- manifest


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputsError(f"cannot read manifest {path}: {exc}") from exc
    problems = validate_manifest(data)
    if problems:
        raise InputsError("manifest invalid:\n  " + "\n  ".join(problems))
    return data


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(data: Any) -> list[str]:
    """Return a list of problems (empty when the manifest is well formed)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["manifest must be an object"]
    if data.get("format") != MANIFEST_FORMAT:
        problems.append(f"format must be {MANIFEST_FORMAT!r}")
    archives = data.get("archives", [])
    files = data.get("files", [])
    trees = data.get("trees", [])
    for key, value in (("archives", archives), ("files", files), ("trees", trees)):
        if not isinstance(value, list):
            problems.append(f"{key} must be a list")
    if problems:
        return problems

    ids: set[str] = set()
    archive_ids: set[str] = set()
    filenames: set[str] = set()

    def check_id(entry: dict, kind: str) -> None:
        ident = entry.get("id")
        if not isinstance(ident, str) or not ident:
            problems.append(f"{kind}: missing id")
        elif ident in ids:
            problems.append(f"{kind} {ident}: duplicate id")
        else:
            ids.add(ident)

    for entry in archives:
        check_id(entry, "archive")
        label = f"archive {entry.get('id')}"
        if not _is_sha256(entry.get("sha256")):
            problems.append(f"{label}: sha256 missing or malformed")
        if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
            problems.append(f"{label}: size missing")
        if not str(entry.get("url", "")).startswith("https://"):
            problems.append(f"{label}: url must be https")
        if archive_format(entry) not in ARCHIVE_FORMATS:
            problems.append(f"{label}: unknown archive format (use zip, tar, 7z or burn)")
        name = entry.get("filename")
        try:
            if not isinstance(name, str) or len(safe_relpath(name).parts) != 1:
                raise InputsError("filename must be a plain file name")
        except InputsError as exc:
            problems.append(f"{label}: {exc}")
        else:
            if name in filenames:
                problems.append(f"{label}: duplicate filename {name}")
            filenames.add(name)
        archive_ids.add(str(entry.get("id")))

    # files may be placed inside a tree's folder (e.g. extra DLLs next to python.exe, copied after the
    # tree); files must not overlap each other and trees must not overlap each other.
    dests: dict[str, set[str]] = {"file": set(), "tree": set()}

    def check_dest(entry: dict, kind: str) -> None:
        label = f"{kind} {entry.get('id')}"
        try:
            rel = safe_relpath(entry.get("dest", ""))
        except InputsError as exc:
            problems.append(f"{label}: dest {exc}")
            return
        key = str(rel).lower()
        for other in dests[kind]:
            if key == other or key.startswith(other + "/") or other.startswith(key + "/"):
                problems.append(f"{label}: dest {rel} overlaps another entry")
                break
        dests[kind].add(key)

    for entry in files:
        check_id(entry, "file")
        label = f"file {entry.get('id')}"
        check_dest(entry, "file")
        if not _is_sha256(entry.get("sha256")):
            problems.append(f"{label}: sha256 missing or malformed")
        if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
            problems.append(f"{label}: size missing")
        source = entry.get("source")
        kind = source.get("type") if isinstance(source, dict) else None
        if kind == "url":
            if not str(source.get("url", "")).startswith("https://"):
                problems.append(f"{label}: url must be https")
        elif kind == "archive":
            if source.get("archive") not in archive_ids:
                problems.append(f"{label}: unknown archive {source.get('archive')!r}")
            try:
                safe_relpath(source.get("member", ""))
            except InputsError as exc:
                problems.append(f"{label}: member {exc}")
        elif kind == "local":
            cands = source.get("candidates")
            if not isinstance(cands, list) or not cands or not all(isinstance(c, str) for c in cands):
                problems.append(f"{label}: local source needs candidates")
        else:
            problems.append(f"{label}: source.type must be url, archive or local")

    for entry in trees:
        check_id(entry, "tree")
        label = f"tree {entry.get('id')}"
        check_dest(entry, "tree")
        source = entry.get("source")
        if not isinstance(source, dict) or source.get("archive") not in archive_ids:
            problems.append(f"{label}: source.archive unknown")
        elif source.get("strip") is not None and not isinstance(source["strip"], str):
            problems.append(f"{label}: strip must be a string")
    return problems


# --------------------------------------------------------------------------- download


def default_opener(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def download_verified(
    url: str,
    dest: Path,
    sha256: str,
    *,
    opener: Opener | None = None,
    log: Callable[[str], None] = print,
    retries: int = 2,
) -> bool:
    """Download ``url`` to ``dest`` unless ``dest`` already has the right hash.

    Returns True when a download happened, False for a cache hit. The data goes to
    ``<dest>.part`` first; a hash mismatch deletes the temp file (and a bad existing
    file) and raises InputsError.
    """
    dest = Path(dest)
    if dest.is_file():
        if sha256_file(dest) == sha256:
            return False
        log(f"  cached file has the wrong hash, deleting: {dest.name}")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    opener = opener or default_opener
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        digest = hashlib.sha256()
        try:
            with opener(url) as response, part.open("wb") as out:
                total = int(getattr(response, "headers", {}).get("Content-Length") or 0) if hasattr(response, "headers") else 0
                done = 0
                shown = time.monotonic()
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    done += len(block)
                    if time.monotonic() - shown > 5:
                        shown = time.monotonic()
                        suffix = f"/{total / 1e6:.0f}" if total else ""
                        log(f"  {dest.name}: {done / 1e6:.0f}{suffix} MB")
        except OSError as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 5))
                continue
            raise InputsError(f"download failed for {url}: {exc}") from exc
        break
    actual = digest.hexdigest()
    if actual != sha256:
        part.unlink(missing_ok=True)
        raise InputsError(f"SHA-256 mismatch for {url}\n  expected {sha256}\n  actual   {actual}")
    os.replace(part, dest)
    return True


# --------------------------------------------------------------------------- extraction


def extract_zip_member(zip_path: Path, member: str, dest: Path) -> None:
    """Extract exactly one member of a zip to ``dest`` (the member name is validated)."""
    safe_relpath(member)
    with zipfile.ZipFile(zip_path) as archive:
        wanted = member.replace("\\", "/")
        names = {info.filename.replace("\\", "/"): info for info in archive.infolist()}
        info = names.get(wanted)
        if info is None or info.is_dir():
            raise InputsError(f"{Path(zip_path).name}: no file member {member!r}")
        safe_relpath(info.filename)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with archive.open(info) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
        os.replace(tmp, dest)


def extract_zip_all(zip_path: Path, dest_dir: Path) -> None:
    """Extract a whole zip, rejecting traversal members."""
    dest_dir = Path(dest_dir)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            rel = safe_relpath(info.filename)
            target = _join_safe(dest_dir, rel)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)


def extract_tar_tree(tar_path: Path, dest_dir: Path, strip: str = "") -> int:
    """Extract a tar(.gz) into ``dest_dir`` (removed first), dropping the ``strip`` prefix.

    Only regular files and folders are accepted; links and traversal names raise
    InputsError. Returns the number of files written.
    """
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    count = 0
    with tarfile.open(tar_path, "r:*") as archive:
        prefix = PurePosixPath(strip.strip("/")) if strip else None
        for info in archive:
            rel = safe_relpath(info.name)
            if prefix is not None:
                try:
                    rel = rel.relative_to(prefix)
                except ValueError:
                    raise InputsError(f"tar member outside {strip!r}: {info.name}") from None
            if not rel.parts:
                continue
            if not (info.isfile() or info.isdir()):
                raise InputsError(f"unsupported tar member type (link/device): {info.name}")
            target = _join_safe(dest_dir, rel)
            if info.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = archive.extractfile(info)
            if src is None:
                raise InputsError(f"cannot read tar member {info.name}")
            with src, target.open("wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
            count += 1
    return count


# --------------------------------------------------------------------------- 7z and Burn bundles (external tools)


ARCHIVE_FORMATS = ("zip", "tar", "7z", "burn")


def archive_format(entry: dict) -> str:
    """Archive kind: explicit ``format`` field, else guessed from the file name."""
    fmt = entry.get("format")
    if fmt:
        return fmt
    name = str(entry.get("filename", "")).lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    if name.endswith(".7z"):
        return "7z"
    if name.endswith(".exe"):
        return "burn"
    return ""


def _system32_tool(name: str) -> Path | None:
    root = os.environ.get("SystemRoot")
    if root:
        candidate = Path(root) / "System32" / name
        if candidate.is_file():
            return candidate
    return None


def find_7z_tools() -> list[list[str]]:
    """Command prefixes able to extract 7z: Windows bsdtar (libarchive) first, then 7-Zip."""
    tools: list[list[str]] = []
    bsdtar = _system32_tool("tar.exe")
    if bsdtar:
        tools.append([str(bsdtar), "-xf"])
    for name in ("7z", "7zz", "7za"):
        found = shutil.which(name)
        if found:
            tools.append([found, "x", "-y", "-bd"])
    for pf in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if pf and (Path(pf) / "7-Zip" / "7z.exe").is_file():
            tools.append([str(Path(pf) / "7-Zip" / "7z.exe"), "x", "-y", "-bd"])
    return tools


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, stdin=subprocess.DEVNULL, timeout=1800)


def _commit_extracted(found: Path, base: Path, dest: Path, label: str) -> None:
    """Copy an extracted file to ``dest`` after checking it is a real file inside ``base``."""
    if found.is_symlink() or not found.is_file():
        raise InputsError(f"{label}: extracted member is not a regular file")
    if base.resolve() not in found.resolve().parents:
        raise InputsError(f"{label}: extracted member escapes the temp folder")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    shutil.copyfile(found, tmp)
    os.replace(tmp, dest)


def extract_7z_member(archive: Path, member: str, dest: Path, *, tools: list[list[str]] | None = None) -> None:
    """Extract exactly one member of a 7z with an external tool (Windows tar.exe, else 7-Zip)."""
    rel = safe_relpath(member)
    tools = find_7z_tools() if tools is None else tools
    if not tools:
        raise InputsError("no tool to read .7z found (need Windows System32\\tar.exe or 7-Zip)")
    errors: list[str] = []
    for prefix in tools:
        with tempfile.TemporaryDirectory(prefix="ohinputs7z-") as tmp_name:
            tmp = Path(tmp_name)
            if Path(prefix[0]).name.lower().startswith("tar"):
                cmd = prefix + [str(archive), "-C", str(tmp), str(rel)]
            else:
                cmd = prefix + [f"-o{tmp}", str(archive), str(rel)]
            try:
                result = _run(cmd, tmp)
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{prefix[0]}: {exc}")
                continue
            found = _join_safe(tmp, rel)
            if result.returncode == 0 and found.is_file():
                _commit_extracted(found, tmp, dest, f"{Path(archive).name}:{member}")
                return
            errors.append(f"{prefix[0]}: exit {result.returncode} {result.stderr.decode('utf-8', 'replace').strip()[:200]}")
    raise InputsError(f"{Path(archive).name}: cannot extract {member!r}: " + "; ".join(errors))


def find_embedded_cabs(data: bytes) -> Iterator[tuple[int, int]]:
    """Yield (offset, size) of every Microsoft cabinet header (``MSCF``) embedded in ``data``."""
    pos = 0
    while True:
        pos = data.find(b"MSCF\0\0\0\0", pos)
        if pos < 0:
            return
        if pos + 36 <= len(data):
            size = int.from_bytes(data[pos + 8:pos + 12], "little")
            if data[pos + 24:pos + 26] == b"\x03\x01" and 36 < size <= len(data) - pos:
                yield pos, size
        pos += 4


def extract_burn_member(exe: Path, member: str, dest: Path, *, expand: str | None = None) -> None:
    """Extract ``<container>/<name>`` from a WiX Burn bundle (e.g. vc_redist.x64.exe) without running it.

    The bundle carries cabinets appended to the exe; one holds payload containers (a12 ...),
    each itself a cabinet. Both layers are opened with Windows' own ``expand.exe``.
    """
    rel = safe_relpath(member)
    if len(rel.parts) != 2:
        raise InputsError(f"burn member must be '<container>/<name>': {member!r}")
    container, name = rel.parts
    expand = expand or (str(_system32_tool("expand.exe") or "") or shutil.which("expand.exe") or "")
    if not expand:
        raise InputsError("expand.exe not found (Burn bundles can only be opened on Windows)")
    data = Path(exe).read_bytes()
    for offset, size in find_embedded_cabs(data):
        with tempfile.TemporaryDirectory(prefix="ohinputs-burn-") as tmp_name:
            tmp = Path(tmp_name)
            outer = tmp / "outer.cab"
            outer.write_bytes(data[offset:offset + size])
            first = tmp / "l1"
            first.mkdir()
            result = _run([expand, f"-F:{container}", str(outer), str(first)], tmp)
            inner = first / container
            if result.returncode != 0 or not inner.is_file():
                continue
            second = tmp / "l2"
            second.mkdir()
            result = _run([expand, f"-F:{name}", str(inner), str(second)], tmp)
            found = second / name
            if result.returncode != 0 or not found.is_file():
                raise InputsError(f"{Path(exe).name}: container {container} has no file {name!r}")
            _commit_extracted(found, second, dest, f"{Path(exe).name}:{member}")
            return
    raise InputsError(f"{Path(exe).name}: no embedded container {container!r} found")


def extract_archive_member(entry: dict, archive: Path, member: str, dest: Path) -> None:
    fmt = archive_format(entry)
    if fmt == "zip":
        extract_zip_member(archive, member, dest)
    elif fmt == "7z":
        extract_7z_member(archive, member, dest)
    elif fmt == "burn":
        extract_burn_member(archive, member, dest)
    else:
        raise InputsError(f"archive {entry.get('id')}: cannot extract single members from format {fmt!r}")


# --------------------------------------------------------------------------- manifest operations


def archive_path(entry: dict, cache: Path) -> Path:
    return Path(cache) / "archives" / entry["filename"]


def staged_file_path(entry: dict, cache: Path) -> Path:
    return _join_safe(Path(cache) / "files", safe_relpath(entry["dest"]))


def staged_tree_path(entry: dict, cache: Path) -> Path:
    return Path(cache) / "trees" / entry["id"]


def _by_id(items: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in items}


def resolve_local(entry: dict, local_dirs: list[Path]) -> Path:
    """Find a local source file whose SHA-256 matches the entry."""
    candidates: list[Path] = []
    for text in entry["source"]["candidates"]:
        candidates.append(Path(text.replace("{project}", str(project_root()))))
    env_dir = os.environ.get(LOCAL_ENV)
    for base in ([Path(env_dir)] if env_dir else []) + list(local_dirs):
        candidates.insert(0, base / Path(*safe_relpath(entry["dest"]).parts))
    mismatched: list[str] = []
    for candidate in candidates:
        if candidate.is_file():
            if sha256_file(candidate) == entry["sha256"]:
                return candidate
            mismatched.append(str(candidate))
    detail = "; ".join(f"wrong hash: {m}" for m in mismatched) or "none of the candidate paths exists"
    raise InputsError(
        f"{entry['id']}: no local source with SHA-256 {entry['sha256'][:12]}... ({detail}). "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def fetch_all(
    manifest: dict[str, Any],
    cache: Path,
    *,
    opener: Opener | None = None,
    local_dirs: list[Path] | None = None,
    log: Callable[[str], None] = print,
) -> None:
    cache = Path(cache)
    local_dirs = local_dirs or []
    archives = _by_id(manifest.get("archives", []))
    used = {
        e["source"]["archive"]
        for e in manifest.get("files", [])
        if e["source"]["type"] == "archive"
    } | {t["source"]["archive"] for t in manifest.get("trees", [])}

    for ident, entry in archives.items():
        if ident not in used:
            continue
        path = archive_path(entry, cache)
        log(f"archive {ident}")
        if download_verified(entry["url"], path, entry["sha256"], opener=opener, log=log):
            log(f"  downloaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            log("  cache hit")

    for entry in manifest.get("files", []):
        target = staged_file_path(entry, cache)
        log(f"file {entry['id']} -> {entry['dest']}")
        if target.is_file() and sha256_file(target) == entry["sha256"]:
            log("  cache hit")
            continue
        target.unlink(missing_ok=True)
        source = entry["source"]
        if source["type"] == "url":
            download_verified(source["url"], target, entry["sha256"], opener=opener, log=log)
        elif source["type"] == "archive":
            arch = archives[source["archive"]]
            extract_archive_member(arch, archive_path(arch, cache), source["member"], target)
            actual = sha256_file(target)
            if actual != entry["sha256"]:
                target.unlink(missing_ok=True)
                raise InputsError(
                    f"{entry['id']}: SHA-256 mismatch after extraction\n  expected {entry['sha256']}\n  actual   {actual}"
                )
        else:
            src = resolve_local(entry, local_dirs)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            shutil.copyfile(src, tmp)
            if sha256_file(tmp) != entry["sha256"]:
                tmp.unlink(missing_ok=True)
                raise InputsError(f"{entry['id']}: copy of {src} does not match SHA-256")
            os.replace(tmp, target)
            log(f"  copied from {src}")

    for entry in manifest.get("trees", []):
        arch = archives[entry["source"]["archive"]]
        tree = staged_tree_path(entry, cache)
        marker = tree / COMPLETE_MARKER
        log(f"tree {entry['id']} -> {entry['dest']}")
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == arch["sha256"]:
            log("  cache hit")
            continue
        count = extract_tar_tree(archive_path(arch, cache), tree, entry["source"].get("strip", ""))
        marker.write_text(arch["sha256"] + "\n", encoding="utf-8")
        log(f"  extracted {count} files")


def verify_all(manifest: dict[str, Any], cache: Path) -> list[str]:
    """Check everything in the cache against the manifest. Returns a list of problems."""
    cache = Path(cache)
    problems: list[str] = []
    archives = _by_id(manifest.get("archives", []))
    used = {
        e["source"]["archive"]
        for e in manifest.get("files", [])
        if e["source"]["type"] == "archive"
    } | {t["source"]["archive"] for t in manifest.get("trees", [])}
    for ident in sorted(used):
        path = archive_path(archives[ident], cache)
        if not path.is_file():
            problems.append(f"archive {ident}: missing {path}")
        elif sha256_file(path) != archives[ident]["sha256"]:
            problems.append(f"archive {ident}: SHA-256 mismatch")
    for entry in manifest.get("files", []):
        path = staged_file_path(entry, cache)
        if not path.is_file():
            problems.append(f"file {entry['id']}: missing {path}")
        elif path.stat().st_size != entry["size"]:
            problems.append(f"file {entry['id']}: size {path.stat().st_size} != {entry['size']}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"file {entry['id']}: SHA-256 mismatch")
    for entry in manifest.get("trees", []):
        marker = staged_tree_path(entry, cache) / COMPLETE_MARKER
        want = archives[entry["source"]["archive"]]["sha256"]
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != want:
            problems.append(f"tree {entry['id']}: not extracted from the pinned archive")
    return problems


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="cache folder (default: build-installer/cache)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--local-dir", type=Path, action="append", default=[], help="extra folder searched for local-source files (laid out like the install root)")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "fetch":
            fetch_all(manifest, args.cache, local_dirs=args.local_dir)
        problems = verify_all(manifest, args.cache)
    except InputsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    print(f"OK: {len(manifest.get('files', []))} files, {len(manifest.get('trees', []))} trees verified in {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
