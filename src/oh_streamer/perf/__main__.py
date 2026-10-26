"""CLI: python -m oh_streamer.perf run [--quick|--full] | show [FILE]   (also: python -m oh_streamer perf ...)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _utf8_console() -> None:
    """Windows consoles are cp950 etc.: emoji / zh-TW text must not crash printing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - not a TextIOWrapper (redirected/captured)
            pass


def _table(profile: dict) -> str:
    try:
        from oh_streamer.perf import profile as profile_mod
        return profile_mod.format_table(profile)
    except Exception as e:  # noqa: BLE001
        return f"(table unavailable: {type(e).__name__}: {e})"


def default_profile_path() -> Path:
    try:
        from oh_streamer.perf import profile
        fn = getattr(profile, "default_path", None)
        if callable(fn):
            return Path(fn(profile.current_engine()))
    except Exception:  # noqa: BLE001
        pass
    from oh_streamer.platform import platform_support
    return platform_support.data_dir() / "perf_profile.json"


def cmd_run(ns) -> int:
    from oh_streamer.perf import bench
    options = bench.options_from_args(ns)
    options["out"] = options["out"] or str(default_profile_path())
    try:
        profile = bench.run_benchmark(options)
    except bench.Cancelled:
        print("[perf] cancelled", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("[perf] interrupted", file=sys.stderr)
        return 130
    print(bench.format_summary(profile))
    print()
    print(_table(profile))
    print(f"profile written to {options['out']}")
    return 0


def cmd_show(ns) -> int:
    path = Path(ns.file) if ns.file else default_profile_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        return 1
    print(_table(data))
    return 0


def cmd_child(ns) -> int:
    from oh_streamer.perf import bench
    return bench.child_main(ns.payload)


def main(argv: list[str] | None = None) -> int:
    from oh_streamer.runtime_env.app_running import hold_app_mutex
    hold_app_mutex()
    _utf8_console()
    ap = argparse.ArgumentParser(prog="oh_streamer perf", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    from oh_streamer.perf import bench
    run = sub.add_parser("run", help="measure this computer (about 3.5 minutes, --quick about 1.5)")
    bench.build_run_parser(run)
    run.set_defaults(func=cmd_run)
    show = sub.add_parser("show", help="print a saved profile")
    show.add_argument("file", nargs="?", default=None)
    show.set_defaults(func=cmd_show)
    child = sub.add_parser("child", help=argparse.SUPPRESS)
    child.add_argument("payload")
    child.set_defaults(func=cmd_child)
    ns = ap.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
