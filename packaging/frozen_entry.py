"""Single frozen executable entry; import application code only after routing."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


PLAYER_MODULE = "oh_streamer.player.oh_player"
QUEUE_MODULE = "oh_streamer.queue.queue_app"
WORKER_MODULE = "oh_streamer.queue.queue_worker"
PERF_MODULE = "oh_streamer.perf.__main__"
BROWSER_MODULE = "oh_streamer.sources.resolvers.resolve_browser"
# Hidden child routes (see oh_streamer.subtitles.environment.subtitle_command).
SUBTITLE_ROUTES = {
    "--subtitle-worker": "oh_streamer.subtitles.worker",
}


class DispatchError(ValueError):
    pass


def route(args: list[str]) -> tuple[str, list[str]]:
    """Accept public modes and legacy ``sys.executable script.py`` child argv."""
    args = list(args)
    if args and args[0] in ("--queue", "--worker", "--perf"):
        return {"--queue": QUEUE_MODULE, "--worker": WORKER_MODULE, "--perf": PERF_MODULE}[args[0]], args[1:]
    if args and args[0] in SUBTITLE_ROUTES:
        return SUBTITLE_ROUTES[args[0]], args[1:]
    if args and args[0] == "--resolver":
        if len(args) < 2 or args[1] != "browser":
            raise DispatchError("Available bundled resolver route: browser.")
        return BROWSER_MODULE, args[2:]
    if args:
        script = args[0].replace("\\", "/")
        basename = script.rsplit("/", 1)[-1]
        known = {
            "oh_player.py": PLAYER_MODULE,
            "queue_app.py": QUEUE_MODULE,
            "queue_worker.py": WORKER_MODULE,
        }
        if basename in known:
            return known[basename], args[1:]
        if script == "resolvers/resolve_browser.py" or script.endswith("/resolvers/resolve_browser.py"):
            return BROWSER_MODULE, args[1:]
        if basename.lower().endswith(".py"):
            raise DispatchError("Custom Python scripts are unsupported by the frozen executable; "
                                "configure an absolute external Python interpreter in the resolver command.")
    return (QUEUE_MODULE if not args else PLAYER_MODULE), args


def configure_import_path() -> None:
    """Expose the source package or frozen resource root before importing app code."""
    if getattr(sys, "frozen", False):
        roots = [Path(sys._MEIPASS).resolve()]
    else:
        project_root = Path(__file__).resolve().parent.parent
        roots = [project_root.parent, project_root, project_root / "src"]
    for root in reversed(roots):
        value = str(root.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def resource_root() -> Path:
    """Return the shared source or bundle resource directory."""
    from oh_streamer.paths import resource_root as resolve_resource_root

    return resolve_resource_root()


def configure_frozen_environment() -> None:
    if not getattr(sys, "frozen", False):
        return
    from oh_streamer.paths import bundle_root
    from oh_streamer.platform import platform_support

    if sys.platform == "win32":
        root = bundle_root()
        if root is not None:
            tool_dirs = [root / "tools", root / "tools" / "mpv"]
            bundled_paths = [str(path) for path in tool_dirs if path.is_dir()]
            paths = os.environ.get("PATH", "").split(os.pathsep)
            os.environ["PATH"] = os.pathsep.join(dict.fromkeys([*bundled_paths, *paths]))
            os.environ.setdefault("LADA_MODEL_WEIGHTS_DIR", str(root / "model_weights"))
        return

    # Finder's PATH usually omits Homebrew. FFmpeg/ffprobe remain external;
    # player_runtime selects our nested mpv helper explicitly, ahead of PATH.
    paths = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys(["/opt/homebrew/bin", "/usr/local/bin", *paths]))
    os.environ.setdefault("LADA_MODEL_WEIGHTS_DIR", str(platform_support.data_dir() / "model_weights"))


def main(argv: list[str] | None = None) -> int:
    try:
        module, args = route(sys.argv[1:] if argv is None else argv)
    except DispatchError as exc:
        print(f"oh-streamer: {exc}", file=sys.stderr)
        return 2
    configure_import_path()
    configure_frozen_environment()
    from oh_streamer.paths import checkout_root

    root = checkout_root()
    package_path = Path(*module.split(".")).with_suffix(".py")
    if not getattr(sys, "frozen", False):
        package_path = root / "src" / package_path
    previous = sys.argv
    # run_module is required: player startup behavior is guarded by __name__ == '__main__'.
    sys.argv = [str(root / package_path), *args]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
