"""Privacy-conscious source-only Mac test bundle with a verified payload manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

MANIFEST = "manifest.json"
# Observed read-only from the original Lada checkout on 2026-10-10. This is an
# expected baseline, not a claim about whatever upstream the recipient has.
UPSTREAM = {
    "observed_date": "2026-10-10", "version": "0.11.1-dev",
    "revision": "20cb34a20a83c72c87a991d2c949032c70085b16",
    "public_repository": "https://github.com/ladaapp/lada.git",
    "provenance": "original local Lada checkout, read-only observation; no destination checkout inferred",
    "cpu_extra": {"torch": "2.9.1", "torchvision": "0.24.1"},
    "patched_packages": {"ultralytics": "8.4.4", "mmengine": "0.10.7"},
    "baseline_sha256": {
        "pyproject.toml": "2f04e963672034460e04d90ae3692353754dd917c1f02cbac8afda26f4b6d5b8",
        "uv.lock": "66d1d054be1545a232aa74f59e1de358279b5433df8734537a4f97e0c49a2871",
        "docs/macOS_install.md": "2dda9b8d7f41a2a5ceed7b41dc4c1663a2c776364fd325ec831c5a3c3075ba88",
    },
    "patches": {
        "increase_mms_time_limit.patch": "ed9d42b66b13883bc96cc05a6da14900526ad7acce33b4c649658ff3294a08d9",
        "remove_ultralytics_telemetry.patch": "ada9554a087d01f1f550e042e8dec304208d5c13f378f0d38d189cff75dc2ffe",
        "fix_loading_mmengine_weights_on_torch26_and_higher.diff": "31a453e785a72a51cfcb294954d8b575a8f3c80d7c62a77ae538c3523e060a9b",
    },
}
MODELS = [
    {"filename": "lada_mosaic_detection_model_v2.pt", "sha256": "056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5"},
    {"filename": "lada_mosaic_detection_model_v4_accurate.pt", "sha256": "c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f"},
    {"filename": "lada_mosaic_detection_model_v4_fast.pt", "sha256": "9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4"},
    {"filename": "lada_mosaic_restoration_model_generic_v1.2.pth", "sha256": "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee"},
]
ROOT_FILES = {"LICENSE", "README.md", "README_EN.md", "requirements-streamer.txt", "pyproject.toml",
              "Oh Player.command", "Oh Streamer.command", "setup_macos.command"}
RESOURCE_ASSETS = {
    "src/oh_streamer/resources/oh_seek.lua",
    "src/oh_streamer/resources/oh_compare_wipe.lua",
    "src/oh_streamer/resources/oh_sub_combos.lua",
    "src/oh_streamer/resources/oh_subtitle_preferences.lua",
    "src/oh_streamer/resources/resolvers.default.json",
    "src/oh_streamer/resources/resolvers.example.json",
    "src/oh_streamer/resources/queue_ui/index.html",
    "src/oh_streamer/resources/queue_ui/app.js",
    "src/oh_streamer/resources/queue_ui/app.css",
    "src/oh_streamer/resources/queue_ui/app-modern.css",
    "src/oh_streamer/resources/queue_ui/branding.js",
    "src/oh_streamer/resources/icons/app.png",
    "src/oh_streamer/resources/icons/app.icns",
}
REQUIRED = {"oh_player.py", "queue_app.py", "resolvers/resolve_browser.py",
            "requirements-streamer.txt",
            "README.md", "Oh Player.command", "Oh Streamer.command", "setup_macos.command",
            "portability_fixtures.py", "borrow_mac_bundle.py", "portability_doctor.py",
            "portability_acceptance.py", "benchmarks/bench_portability.py", "docs/docs_borrow_mac.md",
            "docs/docs_macos_setup.md", *RESOURCE_ASSETS}
PACKAGE_MODULES = {
    "player": ("oh_player player_runtime controller buffer_progress info_panel mpv_ipc open_input opening pipeline "
               "player_settings progressive_stream queue_bridge seek_tail single_instance thumbnails"),
    "queue": "queue_app queue_download queue_ipc queue_service queue_store queue_thumbs queue_worker",
    "processing": "compare_stack detect_gate device_backend encoding_backend rife_interp",
    "sources": "qualities source_resolve hls_cache http_transport",
    "platform": "platform_support player_branding windows_identity airplay_cast hls_cast pyatv_compat",
}
REQUIRED.update({"pyproject.toml", "src/oh_streamer/__init__.py",
                 "src/oh_streamer/__main__.py", "src/oh_streamer/paths.py",
                 "src/oh_streamer/branding.py",
                 "src/oh_streamer/user_data.py",
                 "src/oh_streamer/sources/resolvers/__init__.py",
                  "src/oh_streamer/sources/resolvers/resolve_browser.py",
                  "src/oh_streamer/sources/resolvers/browser.py",
                  "src/oh_streamer/sources/resolvers/covers.py"})
for _group, _modules in PACKAGE_MODULES.items():
    REQUIRED.add(f"src/oh_streamer/{_group}/__init__.py")
    REQUIRED.update(f"src/oh_streamer/{_group}/{name}.py" for name in _modules.split())
BLOCKED_PARTS = {".git", ".venv", "venv", "__pycache__", ".test-tmp", ".test_temp",
                 ".port_baseline", "testdata", "model_weights", "logs", "credentials",
                 "settings", "node_modules", ".aws", ".codex", ".agents"}


def source_allowed(relative: str, include_tests=False) -> bool:
    """Only designated source extensions/locations; never recursive directory copy."""
    path = PurePosixPath(relative)
    if (not relative or "\\" in relative or path.is_absolute() or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        return False
    if any(part.lower() in BLOCKED_PARTS or part.startswith(".") or part.endswith((".", " "))
           or part.split(".")[0].lower() in {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
           for part in path.parts):
        return False
    name = path.name.lower()
    if any(token in name for token in (".bak", "credential", "secret", "private", "settings.json")):
        return False
    if len(path.parts) == 1:
        if relative in ROOT_FILES:
            return True
        if relative.startswith("docs_") and path.suffix == ".md":
            return True
        if path.suffix == ".spec" and relative == "streamer_macos.spec":
            return True
        if path.suffix == ".py":
            if name.startswith("test_"):
                return bool(include_tests)
            if name.startswith("bench_"):
                return relative == "bench_portability.py"
            return True
    if path.parts[0] == "docs":
        return len(path.parts) == 2 and name.startswith("docs_") and path.suffix == ".md"
    if path.parts[0] == "benchmarks":
        return relative == "benchmarks/bench_portability.py"
    if path.parts[0] == "tests":
        return bool(include_tests) and len(path.parts) == 2 and path.suffix == ".py" and (name.startswith("test_") or name == "__init__.py")
    if path.parts[:3] == ("src", "oh_streamer", "resources"):
        return relative in RESOURCE_ASSETS
    if path.parts[:2] == ("src", "oh_streamer"):
        return path.suffix == ".py" and len(path.parts) in {3, 4, 5}
    # These directories contain public application assets/modules only.
    if path.parts[0] == "resolvers":
        return path.suffix == ".py" and len(path.parts) == 2 and not name.startswith("test_")
    if path.parts[0] == "queue_ui":
        return len(path.parts) == 2 and path.suffix in {".html", ".css", ".js"}
    if path.parts[0] == "packaging":
        return path.suffix in {".py", ".spec", ".md", ".command", ".sh"}
    return False


def inventory(root: Path, include_tests=False) -> list[Path]:
    root = root.resolve()
    result = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if source_allowed(relative, include_tests) and path.is_file():
            # Reject symlinks in any ancestor, including links resolving inside root.
            if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
                raise ValueError(f"Source symlinks are not permitted: {relative}")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Source escapes root: {relative}")
            result.append(path)
    found = {path.relative_to(root).as_posix() for path in result}
    missing = REQUIRED - found
    if missing:
        raise ValueError("Incomplete source bundle; missing: " + ", ".join(sorted(missing)))
    return sorted(result)


def create_bundle(output: Path, project_root: Path | None = None, include_tests=False) -> dict:
    root = Path(project_root or Path(__file__).parent).resolve()
    if (root / "streamer" / "oh_player.py").is_file():
        root = root / "streamer"
    paths = inventory(root, include_tests)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "bundle_type": "source-only Mac preflight",
                "layout": "Extract streamer/ beside a separately prepared lada/ checkout",
                "include_tests": bool(include_tests), "upstream_expected": UPSTREAM,
                "models_expected": MODELS, "model_weights_included": False,
                "fixture": "Rebuild via portability_fixtures.py; synthetic transport/audio only",
                "manifest_hash_policy": "SHA256 for every payload member; manifest itself excluded to avoid self-reference",
                "members": {}}
    with tempfile.TemporaryDirectory(prefix="bundle-", dir=output.parent) as tmp:
        temporary = Path(tmp) / "bundle.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                name = "streamer/" + path.relative_to(root).as_posix()
                data = path.read_bytes()
                manifest["members"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if path.suffix in {".command", ".sh"} else 0o644
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(info, data)
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_bundle(temporary)
        temporary.replace(output)
    return manifest


def verify_bundle(archive_path: Path) -> dict:
    """Verify without extraction: reject paths, duplicate/extra/missing files, bad hashes."""
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError("Duplicate archive members")
        for info in infos:
            name = info.filename
            if name != MANIFEST and (not name.startswith("streamer/") or not source_allowed(name[9:], True)):
                raise ValueError(f"Forbidden or unsafe archive path: {name}")
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Directory or symlink archive member: {name}")
        if MANIFEST not in names:
            raise ValueError("Missing manifest")
        manifest = json.loads(archive.read(MANIFEST))
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or not isinstance(manifest.get("members"), dict)
                or not isinstance(manifest.get("include_tests"), bool)):
            raise ValueError("Unsupported manifest")
        members = manifest["members"]
        if set(names) != set(members) | {MANIFEST}:
            raise ValueError("Manifest has missing or extra archive members")
        if not {"streamer/" + item for item in REQUIRED}.issubset(members):
            raise ValueError("Manifest omits required source files")
        for name, expected in members.items():
            if not source_allowed(name[9:], manifest.get("include_tests", False)):
                raise ValueError(f"Member violates manifest source policy: {name}")
            if (not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
                    or type(expected.get("size")) is not int or expected["size"] < 0):
                raise ValueError(f"Invalid member digest: {name}")
            data = archive.read(name)
            if len(data) != expected.get("size") or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"SHA256/size mismatch: {name}")
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args()
    try:
        manifest = verify_bundle(args.verify) if args.verify else create_bundle(args.output, args.project_root, args.include_tests)
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        parser.exit(1, f"Bundle failed: {exc}\n")
    print(json.dumps({"status": "verified", "members": len(manifest["members"]),
                      "archive": str(args.verify or args.output), "upstream_revision": UPSTREAM["revision"]}, indent=2))


if __name__ == "__main__":
    main()
