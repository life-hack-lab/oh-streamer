"""Run the player, queue, worker, or browser resolver from the package."""
from __future__ import annotations

import runpy
import sys


_MODULES = {
    "player": "oh_streamer.player.oh_player",
    "queue": "oh_streamer.queue.queue_app",
    "worker": "oh_streamer.queue.queue_worker",
    "resolver": "oh_streamer.sources.resolvers.resolve_browser",
    "perf": "oh_streamer.perf.__main__",
}


def route(argv: list[str]) -> tuple[str, list[str]]:
    args = list(argv)
    if args and args[0] in ("--queue", "queue"):
        return _MODULES["queue"], args[1:]
    if args and args[0] in ("--worker", "worker"):
        return _MODULES["worker"], args[1:]
    if args and args[0] in ("--perf", "perf"):
        return _MODULES["perf"], args[1:]
    if args and args[0] in ("--resolver", "resolver"):
        if len(args) < 2 or args[1] != "browser":
            raise ValueError("Available bundled resolver: browser.")
        return _MODULES["resolver"], args[2:]
    if args and args[0] == "player":
        args = args[1:]
    return _MODULES["player"], args


def main(argv: list[str] | None = None) -> int:
    try:
        module, args = route(sys.argv[1:] if argv is None else argv)
    except ValueError as exc:
        print(f"oh-streamer: {exc}", file=sys.stderr)
        return 2
    previous = sys.argv
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous
    return 0


def queue_main() -> int:
    """Console entry point that preserves the queue module's script lifecycle."""
    return main(["queue", *sys.argv[1:]])


def worker_main() -> int:
    """Console entry point that preserves the worker module's script lifecycle."""
    return main(["worker", *sys.argv[1:]])


def resolver_main() -> int:
    """Console entry point for the bundled browser resolver."""
    return main(["resolver", "browser", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
