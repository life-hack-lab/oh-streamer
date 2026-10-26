"""Check that repo paths referenced from docs and source files exist in the tracked tree.

Scans tracked *.md files (markdown links and bare mentions) and tracked
.py/.js/.lua/.iss/.json files (bare mentions) for references such as
``docs/<name>.md``, ``docs_<name>.md``, ``docs/<dir>/``, ``benchmarks/<name>.py``,
``tools/<name>``, ``scripts/<name>``. Exits 1 when any referenced target is missing.
"""
from __future__ import annotations

import posixpath
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCAN_SUFFIXES = (".md", ".py", ".js", ".lua", ".iss", ".json")

MD_LINK_RE = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
BARE_PATTERNS = [
    re.compile(r"(?<![\w./-])(docs/[\w.-]+(?:/[\w.-]+)*\.md)\b"),
    re.compile(r"(?<![\w./-])(docs_[\w-]+\.md)\b"),
    re.compile(r"(?<![\w./-])(docs/[\w-]+/)(?![\w.-])"),
    re.compile(r"(?<![\w./-])(benchmarks/[\w-]+\.py)\b"),
    re.compile(r"(?<![\w./-])(tools/[\w-]+(?:\.[A-Za-z0-9]+)?)(?![\w/-])"),
    re.compile(r"(?<![\w./-])(scripts/[\w-]+(?:\.[A-Za-z0-9]+)?)(?![\w/-])"),
]

# Paths intentionally outside the repository (prefix match).
ALLOW_PREFIXES = ("../lada", "testdata/", "%LOCALAPPDATA%", "~", "$")


# Exact paths that are not repo files on purpose: docs/macOS_install.md is hashed in the upstream Lada checkout
# (borrow_mac_bundle baseline), tools/nvencc is an optional untracked dev-machine tool.
ALLOW_EXACT = {"docs/macOS_install.md", "tools/nvencc"}
# Test fixtures build fake trees in temp dirs; their paths are not references to this repo.
SKIP_PREFIXES = ("tests/",)


def _allowed(target: str) -> bool:
    return target.startswith(ALLOW_PREFIXES) or "%" in target or target.rstrip("/") in ALLOW_EXACT


def _exists(path: str, existing: set[str], dirs: set[str]) -> bool:
    path = path.rstrip("/")
    return path in existing or path in dirs


def find_missing(files: dict[str, str], existing: set[str]) -> list[tuple[str, int, str]]:
    """Return (file, line_no, reference) for each reference whose target is not in `existing`."""
    dirs: set[str] = set()
    for p in existing:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    missing: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()

    def report(rel: str, lineno: int, ref: str) -> None:
        key = (rel, lineno, ref)
        if key not in seen:
            seen.add(key)
            missing.append(key)

    for rel, text in sorted(files.items()):
        if rel.startswith(SKIP_PREFIXES):
            continue
        is_md = rel.endswith(".md")
        base = posixpath.dirname(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            if is_md:
                for m in MD_LINK_RE.finditer(line):
                    target = m.group(1)
                    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                        continue
                    target = target.split("#", 1)[0].split("?", 1)[0]
                    if not target or _allowed(target) or not re.search(r"\w", target):
                        continue
                    resolved = posixpath.normpath(posixpath.join(base, target))
                    if resolved.startswith(".."):  # points outside the repo (sibling checkout)
                        continue
                    if not _exists(resolved, existing, dirs):
                        report(rel, lineno, target)
            for pat in BARE_PATTERNS:
                for m in pat.finditer(line):
                    ref = m.group(1)
                    if _allowed(ref):
                        continue
                    cands = {ref, posixpath.normpath(posixpath.join(base, ref))}
                    if ref.startswith("docs_"):
                        cands.add("docs/" + ref)
                    if not any(_exists(c, existing, dirs) for c in cands):
                        report(rel, lineno, ref)
    return missing


def _git_ls_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"git ls-files failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    return [p for p in out.decode("utf-8").split("\0") if p]


def main() -> int:
    tracked = _git_ls_files()
    existing = {p for p in tracked if (REPO_ROOT / p).exists()}
    files: dict[str, str] = {}
    for rel in existing:
        if rel.endswith(SCAN_SUFFIXES):
            try:
                files[rel] = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    missing = find_missing(files, existing)
    print(f"scanned {len(files)} files, {len(existing)} tracked paths")
    for rel, lineno, ref in missing:
        print(f"{rel}:{lineno}: missing -> {ref}")
    print(f"{len(missing)} missing reference(s)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
