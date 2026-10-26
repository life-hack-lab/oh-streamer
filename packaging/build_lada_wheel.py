"""Build the release lada wheel from a pinned commit of official upstream lada.

The wheel is built from ``git archive <commit>`` so neither working-tree edits nor
local-only commits can leak in. The version is baked into ``lada/__init__.py``
because user PCs have no git; ``lada.VERSION`` must equal ``<base>+<short7>``.
Needs ``git`` and ``uv`` on PATH (build machines only).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


STREAMER_ROOT = Path(__file__).resolve().parent.parent

# Official upstream https://github.com/ladaapp/lada, reachable from origin/main.
# Its lada.VERSION is "0.11.1-dev+20cb34a".
LADA_COMMIT = "20cb34a20a83c72c87a991d2c949032c70085b16"
SHORT_LENGTH = 7

# Matches: VERSION = _get_version('0.11.1-dev')
_VERSION_LINE = re.compile(
    r"""^VERSION[ \t]*=[ \t]*_get_version\([ \t]*(?P<q>['"])(?P<base>[^'"\r\n]+)(?P=q)[ \t]*\)[ \t]*(?=\r?$)""",
    re.MULTILINE,
)


def bake_version(text: str, commit: str) -> str:
    """Return ``text`` with the ``VERSION = _get_version('<base>')`` line replaced by a literal.

    Raises ValueError when the line is missing or ambiguous: a silent miss would
    ship a wheel whose version is wrong (or falls back to the bare base version).
    """
    short = commit[:SHORT_LENGTH]
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError(f"invalid commit id: {commit!r}")
    matches = list(_VERSION_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one 'VERSION = _get_version(...)' line in lada/__init__.py, found {len(matches)}")
    base = matches[0].group("base")
    if base.endswith("+" + short) or "+" in base:
        raise ValueError(f"base version already carries a local part: {base!r}")
    literal = f'VERSION = "{base}+{short}"'
    return text[:matches[0].start()] + literal + text[matches[0].end():]


def default_lada_repo() -> Path:
    configured = os.environ.get("OH_STREAMER_PROJECT_ROOT")
    root = Path(configured).expanduser().resolve() if configured else STREAMER_ROOT.parent
    return root / "lada"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def commit_timestamp(repo: Path, commit: str) -> str:
    out = _run(["git", "-C", str(repo), "log", "-1", "--format=%ct", commit],
               capture_output=True, text=True).stdout.strip()
    if not out.isdigit():
        raise RuntimeError(f"cannot read commit timestamp for {commit}: {out!r}")
    return out


def resolve_commit(repo: Path, commit: str) -> str:
    return _run(["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"],
                capture_output=True, text=True).stdout.strip()


def export_source(repo: Path, commit: str, dest: Path) -> None:
    """Extract the tree of ``commit`` into ``dest`` (no working tree, no .git)."""
    data = _run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def bake_source(source: Path, commit: str) -> None:
    init = source / "lada" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    init.write_text(bake_version(text, commit), encoding="utf-8", newline="")


def build_once(repo: Path, commit: str, out_dir: Path, timestamp: str) -> Path:
    """Build one wheel into ``out_dir`` from a fresh export; return its path."""
    with tempfile.TemporaryDirectory(prefix="lada-wheel-src-") as tmp:
        source = Path(tmp) / "src"
        source.mkdir()
        export_source(repo, commit, source)
        bake_source(source, commit)
        env = dict(os.environ, SOURCE_DATE_EPOCH=timestamp)
        built = Path(tmp) / "dist"
        _run(["uv", "build", "--wheel", "--out-dir", str(built), str(source)], env=env)
        wheels = sorted(built.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one wheel, got {[w.name for w in wheels]}")
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / wheels[0].name
        shutil.copyfile(wheels[0], target)
        return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lada-repo", type=Path, default=None,
                        help="official lada clone (default: <OH_STREAMER_PROJECT_ROOT or checkout parent>/lada)")
    parser.add_argument("--commit", default=LADA_COMMIT, help="commit to build (default: pinned release commit)")
    parser.add_argument("--out", type=Path, default=STREAMER_ROOT / "packaging" / "wheels",
                        help="directory receiving the wheel (default: packaging/wheels)")
    parser.add_argument("--verify-reproducible", action="store_true",
                        help="build twice and fail unless both wheels are byte-identical")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = (args.lada_repo or default_lada_repo()).resolve()
    if not (repo / ".git").exists():
        print(f"error: not a git clone: {repo}", file=sys.stderr)
        return 2
    commit = resolve_commit(repo, args.commit)
    timestamp = commit_timestamp(repo, commit)
    print(f"commit {commit} (SOURCE_DATE_EPOCH={timestamp})")
    wheel = build_once(repo, commit, args.out.resolve(), timestamp)
    first = sha256_of(wheel)
    if args.verify_reproducible:
        with tempfile.TemporaryDirectory(prefix="lada-wheel-check-") as tmp:
            second = sha256_of(build_once(repo, commit, Path(tmp), timestamp))
        if second != first:
            print(f"error: wheel is not reproducible ({first} != {second})", file=sys.stderr)
            return 1
        print("reproducible: two builds are byte-identical")
    print(f"wheel  {wheel}")
    print(f"size   {wheel.stat().st_size}")
    print(f"sha256 {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
