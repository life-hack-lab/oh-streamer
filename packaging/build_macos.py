"""macOS-only PyInstaller build. No dependency installation or network access."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


STREAMER_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_RESOURCE_PREFIX = "src/oh_streamer/resources/"
RESOURCE_FILES = (
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
    # Frozen execution routes this CLI shim to the packaged browser resolver.
    "resolvers/resolve_browser.py",
)


def resource_inventory(root: Path = STREAMER_ROOT) -> list[tuple[str, str]]:
    """Exact resource destinations under the frozen root; never mutable user files."""
    names = [*RESOURCE_FILES, *(p.relative_to(root).as_posix() for p in sorted((root / "docs").glob("docs_*.md"))),
             "packaging/README_macos.md"]
    entries = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing packaging resource: {path}")
        if name.startswith(PACKAGED_RESOURCE_PREFIX):
            package_relative = name[len(PACKAGED_RESOURCE_PREFIX):]
            destination = Path("resources") / Path(package_relative).parent
        else:
            destination = Path(name).parent
        entries.append((str(path), destination.as_posix()))
    return entries


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "darwin":
        print("Mac app builds require macOS; cross-building from this OS is unsupported.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lada-source", type=Path, default=STREAMER_ROOT.parent / "lada",
                        help="Prepared upstream checkout root containing lada/__init__.py")
    parser.add_argument("--dist", type=Path, default=STREAMER_ROOT / "dist-macos")
    parser.add_argument("--work", type=Path, default=STREAMER_ROOT / "build-macos")
    parser.add_argument("--mpv", type=Path, required=True,
                        help="Prepared native mpv Mach-O executable; its linked dependencies are bundled")
    parser.add_argument("--mpv-notices", type=Path, required=True,
                        help="Directory of redistribution notices/source offers for the exact mpv dependency build")
    parser.add_argument("--mpv-vulkan-icd", type=Path,
                        help="MoltenVK_icd.json with a resolvable driver library; required when mpv links Vulkan")
    args = parser.parse_args(argv)
    source = args.lada_source.resolve()
    if not (source / "lada" / "__init__.py").is_file():
        parser.error(f"Prepared Lada source is missing: {source / 'lada' / '__init__.py'}")
    if not args.mpv.is_file():
        parser.error(f"Prepared mpv executable is missing: {args.mpv}")
    if not args.mpv_notices.is_dir() or not any(path.is_file() for path in args.mpv_notices.rglob("*")):
        parser.error("--mpv-notices must be a nonempty directory of redistribution notices")
    if args.mpv_vulkan_icd is not None and not args.mpv_vulkan_icd.is_file():
        parser.error(f"Vulkan ICD manifest is missing: {args.mpv_vulkan_icd}")
    resource_inventory()
    env = {**os.environ, "OH_STREAMER_PACKAGING_LADA_SOURCE": str(source),
           "OH_STREAMER_PACKAGING_MPV": str(args.mpv.resolve()),
           "OH_STREAMER_PACKAGING_MPV_NOTICES": str(args.mpv_notices.resolve())}
    # Do not reuse stale configuration inherited from a previous build shell.
    env.pop("OH_STREAMER_PACKAGING_MPV_VULKAN_ICD", None)
    if args.mpv_vulkan_icd is not None:
        env["OH_STREAMER_PACKAGING_MPV_VULKAN_ICD"] = str(args.mpv_vulkan_icd.resolve())
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(args.dist.resolve()),
           "--workpath", str(args.work.resolve()), str(STREAMER_ROOT / "streamer_macos.spec")]
    return subprocess.run(cmd, cwd=STREAMER_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
