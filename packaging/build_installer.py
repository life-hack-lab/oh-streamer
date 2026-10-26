"""Assemble the installer staging folder and its build manifest (standard library only).

    python packaging/build_installer.py stage [--out DIR] [--cache DIR] [--rife-dir DIR] [--local-dir DIR] [--skip-fetch]
    python packaging/build_installer.py verify-stage [DIR]
    python packaging/build_installer.py installer [--skip-stage] [--out DIR] [--stage-dir DIR]   (compile the Inno Setup script)

``stage`` fetches/verifies the external resources (packaging/installer_inputs.py), then lays out the future install root:

    <stage>/app/...                 program files (whitelist below)  + dest files of the inputs manifest
    <stage>/python/                 embedded Python (a tree of the inputs manifest)
    <stage>/LICENSE, THIRD_PARTY_NOTICES.md   project licence + third-party notices (installed in the install folder root)
    <stage>/uv.exe                  (a file of the inputs manifest)
    <stage>/licenses/               licence texts collected from the program, the lada wheel, Python, uv, ffmpeg, RIFE
    <stage>/oh-streamer-build-manifest.json   every file with size + SHA-256, plus version / uv.lock / git / input ids

The Inno script packs the whole stage folder. ``verify-stage`` re-checks the folder against that manifest.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import importlib.util
import json
import re
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_provenance  # noqa: E402  (git_identity)
import installer_inputs as ii  # noqa: E402

BUILD_MANIFEST_NAME = "oh-streamer-build-manifest.json"
BUILD_MANIFEST_FORMAT = "oh-streamer-build-manifest-v1"
DEFAULT_OUT = ROOT / "build-installer" / "stage"

# ---------------------------------------------------------------------------------------------------------------
# What goes into <stage>/app besides the inputs-manifest files (model weights, ffmpeg, mpv, RIFE weights, ...).
# launch_env sets PYTHONPATH=<app>/src and OH_STREAMER_PROJECT_ROOT=<app>, so upstream_root() == app/.
# ---------------------------------------------------------------------------------------------------------------

# Single files, project-relative (source) -> app-relative (destination); must exist.
APP_FILES = [
    ("launch.pyw", "launch.pyw"),                      # shortcut target: pythonw launch.pyw
    ("pyproject.toml", "pyproject.toml"),              # `uv sync --no-install-project` reads dependencies + extras
    ("uv.lock", "uv.lock"),                            # locked versions; its hash is also the env id (layout.lock_id)
]
# Files placed at the stage root (= the install folder root, next to python/ and uv.exe): project licence + third-party notices.
ROOT_FILES = ["LICENSE", "THIRD_PARTY_NOTICES.md"]
# Glob patterns, project-relative, each must match at least one file; the relative path is kept under app/.
APP_GLOBS = [
    "packaging/wheels/*.whl",                          # pyproject [tool.uv.sources] lada = path wheel
]
# Whole folders, project-relative -> app-relative, minus APP_EXCLUDE.
APP_TREES = [
    ("src/oh_streamer", "src/oh_streamer"),            # the program (python, lua, ui, json resources, icons)
]
# Excluded anywhere below a copied tree (fnmatch on each path component / file name).
APP_EXCLUDE = ["__pycache__", "*.pyc", "*.pyo", "*.bak", "*.bak-*", "*.orig", "*.rej", "*.log", ".DS_Store", "Thumbs.db"]
# RIFE code from the repo's own third_party/rife; weights come from the inputs manifest (never from the repo).
RIFE_APP_DEST = "third_party/rife"
RIFE_GLOBS = ["vsrife_models/*.py", "vsrife_models/LICENSE", "README.md"]
# Deliberately NOT shipped (see the A3 log): tests, docs, benchmarks, testdata, tools/nvencc (optional dev-machine NVEncC),
# root-level legacy scripts, scripts/, artifacts/, build outputs.

# Licence files collected into <stage>/licenses/ (name inside licenses/ -> how to find it). Missing ones are reported.
LICENSE_PROGRAM = "oh-streamer-LICENSE.txt"


class BuildError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- small helpers

def sha256_of(path: Path) -> str:
    return ii.sha256_file(path)


def _excluded(parts: tuple[str, ...], patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(part, pat) for part in parts for pat in patterns)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _clear(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def read_version(project: Path) -> str:
    text = (project / "src" / "oh_streamer" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        raise BuildError("version.py has no __version__")
    return m.group(1)


def _load_layout(project: Path):
    path = project / "src" / "oh_streamer" / "runtime_env" / "layout.py"
    spec = importlib.util.spec_from_file_location("_stage_layout", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- app files

def app_sources(project: Path, rife_dir: Path | None) -> dict[str, Path]:
    """Return {app-relative posix path: source file} for the whitelist. Raises BuildError for a missing required item."""
    found: dict[str, Path] = {}

    def add(rel: str, src: Path) -> None:
        if rel in found and found[rel] != src:
            raise BuildError(f"two sources for app/{rel}")
        found[rel] = src

    for src_rel, dst_rel in APP_FILES:
        src = project / src_rel
        if not src.is_file():
            raise BuildError(f"required app file missing: {src_rel}")
        add(dst_rel, src)
    for name in ROOT_FILES:
        if not (project / name).is_file():
            raise BuildError(f"required file missing: {name}")
    for pattern in APP_GLOBS:
        matches = sorted(p for p in project.glob(pattern) if p.is_file())
        if not matches:
            raise BuildError(f"no file matches {pattern}")
        for p in matches:
            add(p.relative_to(project).as_posix(), p)
    for src_rel, dst_rel in APP_TREES:
        base = project / src_rel
        if not base.is_dir():
            raise BuildError(f"required app folder missing: {src_rel}")
        count = 0
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if not p.is_file() or _excluded(rel.parts, APP_EXCLUDE):
                continue
            add((Path(dst_rel) / rel).as_posix(), p)
            count += 1
        if not count:
            raise BuildError(f"{src_rel} contains no files")
    if rife_dir is not None:
        if not rife_dir.is_dir():
            raise BuildError(f"RIFE code folder not found: {rife_dir}")
        for pattern in RIFE_GLOBS:
            matches = sorted(p for p in rife_dir.glob(pattern) if p.is_file())
            if not matches:
                raise BuildError(f"no RIFE file matches {pattern} in {rife_dir}")
            for p in matches:
                add((Path(RIFE_APP_DEST) / p.relative_to(rife_dir)).as_posix(), p)
    return found


# ----------------------------------------------------------------------------------------------- licences

def collect_licenses(stage: Path, project: Path, rife_dir: Path | None, inputs: dict[str, Any], cache: Path) -> tuple[list[str], list[str]]:
    """Copy licence texts into <stage>/licenses. Returns (collected names, missing descriptions)."""
    out = stage / "licenses"
    out.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    missing: list[str] = []

    def put(name: str, src: Path | None, what: str) -> None:
        if src is not None and src.is_file():
            _copy(src, out / name)
            got.append(name)
        else:
            missing.append(what)

    put(LICENSE_PROGRAM, project / "LICENSE", "Oh Streamer (project LICENSE)")

    # lada (AGPL-3.0): the licence files shipped inside the wheel
    wheels = sorted((stage / "app" / "packaging" / "wheels").glob("lada-*.whl"))
    lada_found = False
    if wheels:
        with zipfile.ZipFile(wheels[0]) as z:
            for info in z.infolist():
                m = re.search(r"\.dist-info/licenses/(.+)$", info.filename)
                if m and not info.is_dir():
                    name = "lada/" + m.group(1)
                    dst = out / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(z.read(info))
                    got.append(name)
                    lada_found = True
    if not lada_found:
        missing.append("lada (no licence files inside the wheel)")

    put("python-LICENSE.txt", stage / "python" / "LICENSE.txt", "Python (python/LICENSE.txt from python-build-standalone)")

    # uv: not inside the zip that A1 extracts; look in the archive anyway
    uv_done = False
    for arch in inputs.get("archives", []):
        if arch["id"].startswith("uv"):
            zp = ii.archive_path(arch, cache)
            if zp.is_file():
                with zipfile.ZipFile(zp) as z:
                    for n in z.namelist():
                        if re.search(r"(^|/)(LICENSE|COPYING)[^/]*$", n, re.I):
                            (out / "uv-LICENSE.txt").write_bytes(z.read(n))
                            got.append("uv-LICENSE.txt")
                            uv_done = True
                            break
    if not uv_done and not any(e["id"].startswith("uv-license") for e in inputs.get("files", [])):
        missing.append("uv (the release zip contains only the executables; MIT/Apache-2.0 text must be added by hand)")

    # ffmpeg / mpv (and anything else): licence files the inputs manifest placed (id contains "license"), except model licences
    for entry in inputs.get("files", []):
        if "license" in entry["id"].lower() and entry.get("role") != "model-license":
            if Path(entry["dest"]).parent == Path("licenses"):      # already placed in licenses/ by the inputs manifest
                got.append(Path(entry["dest"]).name)
                continue
            put(Path(entry["dest"]).name, stage / entry["dest"], entry["id"])
    if not any(n.startswith("ffmpeg") for n in got):
        missing.append("ffmpeg (no licence entry in the inputs manifest)")
    if not any("mpv" in n.lower() for n in got):
        missing.append("mpv (GPL-2.0-or-later; the inputs manifest has no licence entry for it)")

    if rife_dir is not None:
        put("rife-LICENSE.txt", rife_dir / "vsrife_models" / "LICENSE", "RIFE code (vs-rife MIT)")
        put("rife-README.md", rife_dir / "README.md", "RIFE README")
    else:
        missing.append("RIFE (no code folder)")
    if not any(n.startswith("LICENSE-Practical-RIFE") for n in got):
        missing.append("RIFE weights flownet_v4.6.pkl (no Practical-RIFE licence entry in the inputs manifest)")
    return got, missing


# ----------------------------------------------------------------------------------------------- stage

def _git_info(project: Path) -> dict[str, Any]:
    try:
        info = build_provenance.git_identity(project)
    except Exception:
        info = {}
    return {"revision": info.get("revision"), "dirty": info.get("dirty")}


def list_stage_files(stage: Path) -> list[Path]:
    return sorted((p for p in stage.rglob("*") if p.is_file() and p.name != BUILD_MANIFEST_NAME),
                  key=lambda p: p.relative_to(stage).as_posix())


def file_records(stage: Path) -> list[dict[str, Any]]:
    return [{"path": p.relative_to(stage).as_posix(), "size": p.stat().st_size, "sha256": sha256_of(p)}
            for p in list_stage_files(stage)]


def stage_build(
    *,
    project: Path = ROOT,
    out: Path = DEFAULT_OUT,
    cache: Path = ii.DEFAULT_CACHE,
    inputs: dict[str, Any] | None = None,
    rife_dir: Path | None = None,
    local_dirs: list[Path] | None = None,
    fetch: bool = True,
    log=print,
) -> dict[str, Any]:
    project, out, cache = Path(project), Path(out), Path(cache)
    inputs = inputs if inputs is not None else ii.load_manifest()
    if fetch:
        ii.fetch_all(inputs, cache, local_dirs=local_dirs or [], log=log)
    problems = ii.verify_all(inputs, cache)
    if problems:
        raise BuildError("external resources failed verification:\n  " + "\n  ".join(problems))

    sources = app_sources(project, rife_dir)          # fail before touching the output folder

    _clear(out)
    out.mkdir(parents=True)
    log(f"stage -> {out}")

    for entry in inputs.get("files", []):
        _copy(ii.staged_file_path(entry, cache), out / ii.safe_relpath(entry["dest"]).as_posix())
    for entry in inputs.get("trees", []):
        base = ii.staged_tree_path(entry, cache)
        dest = out / ii.safe_relpath(entry["dest"]).as_posix()
        for p in sorted(base.rglob("*")):
            if p.is_file() and not (p.parent == base and p.name == ii.COMPLETE_MARKER):
                _copy(p, dest / p.relative_to(base))
    for name in ROOT_FILES:
        if (out / name).exists():
            raise BuildError(f"{name} collides with a file from the inputs manifest")
        _copy(project / name, out / name)
    for rel, src in sources.items():
        dst = out / "app" / rel
        if dst.exists():
            raise BuildError(f"app/{rel} collides with a file from the inputs manifest")
        _copy(src, dst)

    got, missing = collect_licenses(out, project, rife_dir, inputs, cache)
    for m in missing:
        log(f"MISSING LICENCE: {m}")

    lock_path = project / "uv.lock"
    layout = _load_layout(project)
    lock_sha = layout.lock_sha256(lock_path)
    records = file_records(out)
    manifest = {
        "format": BUILD_MANIFEST_FORMAT,
        "version": read_version(project),
        "uv_lock": {"sha256": lock_sha, "lock_id": layout.lock_id(lock_sha)},
        "git": _git_info(project),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "archives": [{"id": a["id"], "filename": a["filename"], "url": a["url"], "sha256": a["sha256"]}
                         for a in inputs.get("archives", [])],
            "files": [{"id": f["id"], "dest": f["dest"], "sha256": f["sha256"], "size": f["size"], "source": f["source"]}
                      for f in inputs.get("files", [])],
            "trees": [{"id": t["id"], "dest": t["dest"], "source": t["source"]} for t in inputs.get("trees", [])],
        },
        "licenses": sorted(got),
        "missing_licenses": missing,
        "files": records,
    }
    (out / BUILD_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(r["size"] for r in records)
    log(f"staged {len(records)} files, {total / 1e6:.1f} MB")
    return manifest


# ----------------------------------------------------------------------------------------------- verify

def verify_stage(stage: Path) -> list[str]:
    stage = Path(stage)
    mpath = stage / BUILD_MANIFEST_NAME
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {mpath}: {exc}"]
    if manifest.get("format") != BUILD_MANIFEST_FORMAT:
        return [f"unknown manifest format {manifest.get('format')!r}"]
    problems: list[str] = []
    expected = {r["path"]: r for r in manifest.get("files", [])}
    actual = {p.relative_to(stage).as_posix(): p for p in list_stage_files(stage)}
    for rel in sorted(expected.keys() - actual.keys()):
        problems.append(f"missing: {rel}")
    for rel in sorted(actual.keys() - expected.keys()):
        problems.append(f"unexpected: {rel}")
    for rel in sorted(expected.keys() & actual.keys()):
        rec, p = expected[rel], actual[rel]
        if p.stat().st_size != rec["size"]:
            problems.append(f"size differs: {rel}")
        elif sha256_of(p) != rec["sha256"]:
            problems.append(f"sha256 differs: {rel}")
    return problems


# ----------------------------------------------------------------------------------------------- Inno Setup

ISS_PATH = HERE / "installer" / "oh_streamer.iss"
DEFAULT_INSTALLER_OUT = ROOT / "build-installer" / "out"
BRANDING_JS = ROOT / "src" / "oh_streamer" / "resources" / "queue_ui" / "branding.js"
ISCC_ENV = "OH_STREAMER_ISCC"
ISCC_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
]


def read_brand_name(branding_js: Path = BRANDING_JS) -> str:
    text = branding_js.read_text(encoding="utf-8")
    m = re.search(r"window\.BRAND\s*=\s*(\{.*?\})\s*;", text, re.S)
    if not m:
        raise BuildError(f"{branding_js.name}: window.BRAND object not found")
    try:
        name = json.loads(m.group(1)).get("name")
    except json.JSONDecodeError as exc:
        raise BuildError(f"{branding_js.name}: BRAND is not strict JSON: {exc}") from exc
    if not isinstance(name, str) or not name.strip():
        raise BuildError(f"{branding_js.name}: BRAND.name is empty")
    return name.strip()


def installer_basename(app_name: str, version: str) -> str:
    """'Oh Streamer' + '0.1.0' -> 'OhStreamer-Setup-0.1.0' (spaces and odd characters dropped)."""
    compact = re.sub(r"[^0-9A-Za-z_.\-]", "", app_name) or "Setup"
    return f"{compact}-Setup-{version}"


def find_iscc(env: dict[str, str] | None = None, candidates: list[Path] | None = None) -> Path:
    env = os.environ if env is None else env
    custom = env.get(ISCC_ENV)
    if custom:
        p = Path(custom)
        if not p.is_file():
            raise BuildError(f"{ISCC_ENV} points to a missing file: {p}")
        return p
    for p in (candidates if candidates is not None else ISCC_CANDIDATES):
        if p.is_file():
            return p
    raise BuildError(f"ISCC.exe not found (install Inno Setup 6 or set {ISCC_ENV})")


def iscc_command(iscc: Path, iss: Path, *, stage: Path, out: Path, app_name: str, version: str) -> list[str]:
    return [
        str(iscc), "/Q",
        f"/DAppName={app_name}",
        f"/DAppVersion={version}",
        f"/DStageDir={Path(stage).resolve()}",
        f"/DOutDir={Path(out).resolve()}",
        f"/DOutputBaseName={installer_basename(app_name, version)}",
        str(iss),
    ]


def build_installer(*, stage: Path = DEFAULT_OUT, out: Path = DEFAULT_INSTALLER_OUT, log=print) -> Path:
    stage = Path(stage)
    try:
        manifest = json.loads((stage / BUILD_MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read the stage manifest in {stage}: {exc}") from exc
    version = manifest.get("version")
    if not version:
        raise BuildError("stage manifest has no version")
    app_name = read_brand_name()
    iscc = find_iscc()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = iscc_command(iscc, ISS_PATH, stage=stage, out=out, app_name=app_name, version=version)
    log("ISCC: " + " ".join(cmd))
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(ISS_PATH.parent))
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise BuildError(f"ISCC failed with exit code {proc.returncode}")
    exe = out / (installer_basename(app_name, version) + ".exe")
    if not exe.is_file():
        raise BuildError(f"ISCC finished but {exe} does not exist")
    log(f"installer: {exe} ({exe.stat().st_size / 1e6:.1f} MB, compiled in {elapsed:.0f} s)")
    return exe


def check_iss(*, log=print) -> None:
    """Parse/compile the .iss against a tiny fake stage with ISCC's /O- (no installer file is written).

    Catches syntax errors, missing #define values and bad directives without downloading the 1 GB of inputs."""
    import tempfile
    app_name = read_brand_name()
    iscc = find_iscc()
    icon_rel = Path("app") / "src" / "oh_streamer" / "resources" / "icons" / "app.ico"
    with tempfile.TemporaryDirectory(prefix="oh_iss_check_") as tmp:
        stage = Path(tmp) / "stage"
        icon_dst = stage / icon_rel
        icon_dst.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "src" / "oh_streamer" / "resources" / "icons" / "app.ico", icon_dst)
        (stage / "placeholder.txt").write_text("iss check", encoding="utf-8")
        (stage / "redist").mkdir()
        (stage / "redist" / "MicrosoftEdgeWebview2Setup.exe").write_bytes(b"MZ placeholder")   # embedded with dontcopy
        cmd = iscc_command(iscc, ISS_PATH, stage=stage, out=Path(tmp) / "out", app_name=app_name, version="0.0.0")
        cmd.insert(2, "/O-")
        log("ISCC: " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(ISS_PATH.parent))
        if proc.returncode != 0:
            raise BuildError(f"ISCC failed to compile the .iss (exit code {proc.returncode})")
    log("oh_streamer.iss compiles")


# ----------------------------------------------------------------------------------------------- CLI

def _default_rife_dir() -> Path:
    return ii.ROOT / "third_party" / "rife"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--out", type=Path, default=DEFAULT_OUT)
    st.add_argument("--cache", type=Path, default=ii.DEFAULT_CACHE)
    st.add_argument("--rife-dir", type=Path, default=None, help="RIFE code folder (default: <repo>/third_party/rife)")
    st.add_argument("--local-dir", type=Path, action="append", default=[])
    st.add_argument("--skip-fetch", action="store_true", help="only verify the cache, do not download")
    vs = sub.add_parser("verify-stage")
    vs.add_argument("dir", nargs="?", type=Path, default=DEFAULT_OUT)
    ins = sub.add_parser("installer", help="compile the Inno Setup installer (runs stage + verify-stage first unless --skip-stage)")
    ins.add_argument("--skip-stage", action="store_true", help="reuse the existing stage folder (still runs verify-stage)")
    ins.add_argument("--stage-dir", type=Path, default=DEFAULT_OUT)
    ins.add_argument("--out", type=Path, default=DEFAULT_INSTALLER_OUT)
    ins.add_argument("--cache", type=Path, default=ii.DEFAULT_CACHE)
    ins.add_argument("--rife-dir", type=Path, default=None)
    ins.add_argument("--local-dir", type=Path, action="append", default=[])
    sub.add_parser("check-iss", help="compile the .iss against a fake tiny stage without writing an installer (CI)")
    args = parser.parse_args(argv)
    try:
        problems: list[str] = []
        if args.command == "check-iss":
            check_iss()
        elif args.command == "installer":
            if not args.skip_stage:
                stage_build(out=args.stage_dir, cache=args.cache, rife_dir=args.rife_dir or _default_rife_dir(),
                            local_dirs=args.local_dir)
            problems = verify_stage(args.stage_dir)
            if not problems:
                build_installer(stage=args.stage_dir, out=args.out)
        elif args.command == "stage":
            stage_build(out=args.out, cache=args.cache, rife_dir=args.rife_dir or _default_rife_dir(),
                        local_dirs=args.local_dir, fetch=not args.skip_fetch)
            problems = verify_stage(args.out)
        else:
            problems = verify_stage(args.dir)
    except (BuildError, ii.InputsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
