"""Scan staged or repository files to ensure no API keys or secrets are committed or packaged."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess
import sys

# High-confidence secret patterns that must never appear in source or release artifacts
SECRET_PATTERNS = [
    ("Groq API Key", re.compile(r"\bgsk_[A-Za-z0-9_-]{20,}\b")),
    ("Google/Gemini API Key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")),
    ("OpenAI/OpenRouter Key", re.compile(r"\bsk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{24,}\b")),
    ("HuggingFace Token", re.compile(r"\bhf_[A-Za-z0-9]{24,}\b")),
    (
        "Azure/Microsoft Translator Key",
        re.compile(
            r"(?i)(?:Ocp-Apim-Subscription-Key|(?:AZURE|TRANSLATOR|COGNITIVE)[A-Z0-9_]*(?:KEY|SECRET))"
            r"[\"']?\s*[:=,]\s*[\"']?(?:[0-9a-f]{32}|[A-Za-z0-9]{84})(?![A-Za-z0-9])"
        ),
    ),
    ("GitHub Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("GitHub Fine-grained Token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("Private Key Block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

FORBIDDEN_FILENAMES = {
    "subtitle_models.json",
    "subtitle_quota.json",
    "lada_settings.json",  # retired name, still never commit
    "oh_settings.json",
    "airplay_credentials.json",
    ".env",
}

ALLOW_MARKER = "pragma: allowlist secret"


def scan_text(content: str, rel_path: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if ALLOW_MARKER in line or "TEST-KEY-NOT-REAL" in line or "fake_" in line or "env_secret" in line:
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(f"{rel_path}:{lineno}: detected potential {label}")
    return findings


def get_candidate_files(repo_root: Path, staged_only: bool) -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged_only else ["git", "ls-files"]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {proc.returncode}) in {repo_root}: {proc.stderr.strip()}")
    return [repo_root / line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Only scan git staged files (for pre-commit)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    try:
        files = get_candidate_files(repo_root, staged_only=args.staged)
    except (RuntimeError, OSError) as exc:
        print(f"[SECRET SCAN ERROR] cannot obtain file list: {exc}", file=sys.stderr)
        return 2
    if not files and not args.staged:
        print("[SECRET SCAN ERROR] file list is empty; refusing to report a false pass", file=sys.stderr)
        return 2
    errors: list[str] = []

    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        if path.name in FORBIDDEN_FILENAMES:
            errors.append(f"{rel}: forbidden credential/settings file must never be tracked by git")
            continue
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        errors.extend(scan_text(text, rel))

    if errors:
        print("[SECRET SCAN FAILED] Potential credentials or forbidden files detected:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
