"""Scan tracked text files for terms that must not appear in the public repository.

Terms come from a small built-in generic list plus an optional private list read from the
file named by the OH_STREAMER_BLOCKLIST_FILE environment variable (one term per line,
blank lines and lines starting with ``#`` are ignored). Matching is case-insensitive
substring search; a term written as ``re:<pattern>`` is a case-insensitive regex instead.
The built-in terms are skipped under ``tests/``, whose fixtures use synthetic paths and
addresses. Exits 1 and prints ``file:line: term`` for every hit.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BUILTIN_TERMS = ("C:\\Users\\","/Users/", "192.168.")

# Paths exempt from scanning (prefix or basename match): license texts and this script.
EXEMPT_PREFIXES = ("third_party/", "packaging/installer/Languages/", "scripts/check_blocklist.py")
BUILTIN_EXEMPT_PREFIXES = ("tests/",)
EXEMPT_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "NOTICE", "THIRD_PARTY_NOTICES.md")


def load_terms(extra_file: str | None) -> list[str]:
    terms = list(BUILTIN_TERMS)
    if extra_file:
        text = Path(extra_file).read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return terms


def is_exempt(rel: str) -> bool:
    return rel.startswith(EXEMPT_PREFIXES) or rel.rsplit("/", 1)[-1] in EXEMPT_NAMES


def _matcher(term: str):
    if term.startswith("re:"):
        return re.compile(term[3:], re.IGNORECASE).search
    low_term = term.lower()
    return lambda line: low_term in line.lower()


def scan_text(text: str, terms: list[str]) -> list[tuple[int, str]]:
    matchers = [(t, _matcher(t)) for t in terms]
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for term, match in matchers:
            if match(line):
                hits.append((lineno, term))
    return hits


def main() -> int:
    extra = os.environ.get("OH_STREAMER_BLOCKLIST_FILE")
    if extra and not Path(extra).is_file():
        print(f"blocklist file not found: {extra}", file=sys.stderr)
        return 2
    terms = load_terms(extra)
    private_terms = terms[len(BUILTIN_TERMS):]
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
    total = 0
    scanned = 0
    for rel in (p for p in out.decode("utf-8").split("\0") if p):
        if is_exempt(rel):
            continue
        path = REPO_ROOT / rel
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:8192]:
            continue  # binary
        scanned += 1
        file_terms = private_terms if rel.startswith(BUILTIN_EXEMPT_PREFIXES) else terms
        for lineno, term in scan_text(raw.decode("utf-8", errors="replace"), file_terms):
            print(f"{rel}:{lineno}: {term}")
            total += 1
    print(f"scanned {scanned} files with {len(terms)} terms ({'private list' if extra else 'built-in only'}); {total} hit(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
