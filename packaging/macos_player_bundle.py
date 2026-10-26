"""Collect a standalone, native mpv helper without changing the source installation.

Linked Mach-O dependencies and an explicitly supplied Vulkan ICD are collected.
Third-party framework resources require a separate bundler and are rejected.
"""
from __future__ import annotations

from collections import namedtuple
import json
import os
from pathlib import Path
import platform
import plistlib
import posixpath
import re
import shutil
import subprocess


PLAYER_APP_RELATIVE = Path("Contents/Helpers/oh-player.app")
PLAYER_EXECUTABLE_RELATIVE = PLAYER_APP_RELATIVE / "Contents/MacOS/mpv"
PLAYER_BUNDLE_ID = "local.oh.streamer.player"
MachO = namedtuple("MachO", "dependencies rpaths dylib_id")
LOAD_COMMANDS = {"LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB", "LC_LOAD_UPWARD_DYLIB"}


def command(*args: str, env: dict[str, str] | None = None) -> str:
    try:
        return subprocess.run(list(args), check=True, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=env, timeout=30).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"mpv packaging command failed: {list(args)!r}: {(error.stderr or '').strip()}") from error


def parse_load_commands(output: str) -> MachO:
    dependencies, rpaths, dylib_id = [], [], None
    current = None
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        key, value = fields
        if key == "cmd":
            current = value
        elif key in {"name", "path"}:
            value = re.sub(r" \(offset \d+\)$", "", value)
            if current in LOAD_COMMANDS and key == "name":
                dependencies.append(value)
            elif current == "LC_ID_DYLIB" and key == "name":
                dylib_id = value
            elif current == "LC_RPATH" and key == "path":
                rpaths.append(value)
    return MachO(tuple(dict.fromkeys(dependencies)), tuple(dict.fromkeys(rpaths)), dylib_id)


def system_library(path: str | Path) -> bool:
    return posixpath.normpath(str(path).replace("\\", "/")).startswith(("/System/Library/", "/usr/lib/"))


def expand_path(value: str, loader: Path, executable: Path) -> Path:
    if value == "@loader_path":
        return loader.parent
    if value == "@executable_path":
        return executable.parent
    if value.startswith("@loader_path/"):
        return loader.parent / value[len("@loader_path/"):]
    if value.startswith("@executable_path/"):
        return executable.parent / value[len("@executable_path/"):]
    return Path(value)


def resolve_dependency(name: str, loader: Path, executable: Path, rpaths: tuple[Path, ...]) -> Path:
    if name.startswith("@rpath/"):
        candidates = [path / name[len("@rpath/"):] for path in rpaths]
    else:
        candidates = [expand_path(name, loader, executable)]
    for candidate in candidates:
        # System dylibs can live only in the dyld shared cache, without a file.
        if system_library(candidate):
            return candidate
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"Unresolved mpv dependency {name!r} in {loader}; prepare a complete mpv build")


def dependency_inventory(executable: Path, architecture: str, *,
                         runtime_executable: Path | None = None) -> dict[Path, tuple[MachO, dict[str, Path]]]:
    """Resolve LC_LOAD_* transitively using each loader's and its ancestors' rpaths."""
    inventory = {}
    runtime_executable = runtime_executable or executable

    def visit(path: Path, inherited_rpaths: tuple[Path, ...]) -> None:
        if path in inventory:
            return
        if ".framework" in path.parts or any(part.endswith(".framework") for part in path.parts):
            raise RuntimeError(f"External framework resources are unsupported in the mpv payload: {path}")
        architectures = command("/usr/bin/lipo", "-archs", str(path)).strip().split()
        if architecture not in architectures:
            raise RuntimeError(f"mpv payload {path} lacks required architecture {architecture}: {architectures}")
        info = parse_load_commands(command("/usr/bin/otool", "-arch", architecture, "-l", str(path)))
        paths = tuple(expand_path(value, path, runtime_executable) for value in info.rpaths) + inherited_rpaths
        dependencies = {}
        inventory[path] = (info, dependencies)
        for name in info.dependencies:
            resolved = resolve_dependency(name, path, runtime_executable, paths)
            dependencies[name] = resolved
            if not system_library(resolved):
                visit(resolved, paths)

    visit(executable.resolve(), ())
    return inventory


def copy_native(source: Path, destination: Path, architecture: str) -> None:
    architectures = command("/usr/bin/lipo", "-archs", str(source)).strip().split()
    if len(architectures) > 1:
        command("/usr/bin/lipo", str(source), "-thin", architecture, "-output", str(destination))
    else:
        shutil.copy2(source, destination)
    destination.chmod(0o755)
    # install_name_tool invalidates existing signatures; remove only copied ones.
    if command("/usr/bin/otool", "-l", str(destination)).find("LC_CODE_SIGNATURE") >= 0:
        command("/usr/bin/codesign", "--remove-signature", str(destination))


def player_plist(app_name: str) -> dict:
    return {"CFBundleName": f"{app_name} Player", "CFBundleDisplayName": f"{app_name} Player",
            "CFBundleIdentifier": PLAYER_BUNDLE_ID, "CFBundleExecutable": "mpv",
            "CFBundleIconFile": "app.icns", "CFBundlePackageType": "APPL",
            "CFBundleInfoDictionaryVersion": "6.0", "CFBundleVersion": "1",
            "NSHighResolutionCapable": True,
            # mpv's Cocoa startup checks this; runtime subprocess env sets it too.
            "LSEnvironment": {"MPVBUNDLE": "true"}}


def bundle_player(app: Path, executable: Path, icon: Path, app_name: str,
                  notices: Path, *, architecture: str | None = None,
                  vulkan_icd: Path | None = None) -> Path:
    """Embed, relink, verify and ad-hoc sign mpv, then reseal its containing app."""
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Prepared mpv executable is missing: {executable}")
    if not notices.is_dir() or not any(path.is_file() for path in notices.rglob("*")):
        raise RuntimeError("--mpv-notices must contain the redistribution notices for mpv and its dependencies")
    architecture = architecture or platform.machine()
    inventory = dependency_inventory(executable, architecture)
    icd_data, icd_library = None, None
    if any(path.name.lower().startswith("libvulkan") for path in inventory):
        if vulkan_icd is None:
            raise RuntimeError("This mpv links Vulkan: --mpv-vulkan-icd must supply its MoltenVK_icd.json driver manifest")
    if vulkan_icd is not None:
        icd_data = json.loads(vulkan_icd.read_text(encoding="utf-8"))
        library_name = icd_data.get("ICD", {}).get("library_path")
        if not isinstance(library_name, str) or not library_name:
            raise RuntimeError(f"Invalid Vulkan ICD.library_path: {vulkan_icd}")
        icd_library = (vulkan_icd.parent / library_name).resolve()
        if not icd_library.is_file():
            raise RuntimeError(f"Missing Vulkan ICD library: {icd_library}")
        inventory.update(dependency_inventory(icd_library, architecture, runtime_executable=executable))
    player = app / PLAYER_APP_RELATIVE
    if player.exists():
        raise FileExistsError(f"Refusing to overwrite an existing player payload: {player}")
    contents = player / "Contents"
    for name in ("MacOS", "Frameworks", "Resources"):
        (contents / name).mkdir(parents=True, exist_ok=True)
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(player_plist(app_name), stream)
    shutil.copy2(icon, contents / "Resources/app.icns")
    shutil.copytree(notices, contents / "Resources/ThirdPartyNotices")
    destinations = {executable: contents / "MacOS/mpv"}
    # Stable unique names avoid conflating different dylibs with the same basename.
    for index, source in enumerate(sorted(set(inventory) - {executable}), 1):
        destinations[source] = contents / "Frameworks" / f"{index:03d}-{source.name}"
    if icd_data is not None:
        icd_data["ICD"]["library_path"] = f"../../../Frameworks/{destinations[icd_library].name}"
        manifest = contents / "Resources/vulkan/icd.d/MoltenVK_icd.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(icd_data, indent=2) + "\n", encoding="utf-8")
    for source, destination in destinations.items():
        copy_native(source, destination, architecture)
        info, dependencies = inventory[source]
        if info.dylib_id:
            command("/usr/bin/install_name_tool", "-id", f"@rpath/{destination.name}", str(destination))
        for old, dependency in dependencies.items():
            if system_library(dependency):
                replacement = str(dependency)
            else:
                relative = os.path.relpath(destinations[dependency], destination.parent).replace(os.sep, "/")
                replacement = f"@loader_path/{relative}"
            if old != replacement:
                command("/usr/bin/install_name_tool", "-change", old, replacement, str(destination))
        # No runtime search path may refer back to the build machine.
        for rpath in info.rpaths:
            command("/usr/bin/install_name_tool", "-delete_rpath", rpath, str(destination))
    verify_payload(contents / "MacOS/mpv", architecture)
    for path in [*sorted((contents / "Frameworks").iterdir()), contents / "MacOS/mpv", player]:
        command("/usr/bin/codesign", "--force", "--sign", "-", str(path))
    smoke_environment = {key: value for key, value in os.environ.items()
                         if not key.startswith(("DYLD_", "VK_")) and key not in {"MPV_HOME", "MPVBUNDLE"}}
    smoke_environment.update({"PATH": "/usr/bin:/bin", "MPVBUNDLE": "true"})
    if icd_data is not None:
        smoke_environment.update({"VK_DRIVER_FILES": str(manifest), "VK_ICD_FILENAMES": str(manifest)})
    command(str(contents / "MacOS/mpv"), "--no-config", "--load-scripts=no", "--version", env=smoke_environment)
    options = command(str(contents / "MacOS/mpv"), "--no-config", "--load-scripts=no", "--list-options", env=smoke_environment)
    if "macos-app-activation-policy" not in options:
        raise RuntimeError("Prepared mpv lacks the macOS Cocoa application options needed for the player bundle")
    command("/usr/bin/codesign", "--force", "--sign", "-", str(app))
    command("/usr/bin/codesign", "--verify", "--deep", "--strict", str(app))
    return player


def verify_payload(executable: Path, architecture: str) -> None:
    """Reinspect rewritten binaries; any non-system reference must stay inside helper."""
    contents = executable.parents[1].resolve()
    for path in [executable, *sorted((contents / "Frameworks").iterdir())]:
        info = parse_load_commands(command("/usr/bin/otool", "-arch", architecture, "-l", str(path)))
        if info.rpaths:
            raise RuntimeError(f"Unexpected runtime search paths remain in bundled mpv: {path}: {info.rpaths}")
        for name in info.dependencies:
            if system_library(name):
                continue
            resolved = resolve_dependency(name, path, executable, ())
            if not resolved.is_relative_to(contents) or not resolved.is_file():
                raise RuntimeError(f"Bundled mpv dependency escapes its payload: {path}: {name}")
