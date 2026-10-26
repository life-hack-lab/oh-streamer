"""Which upstream lada versions this engine adapter was verified against (engine-decoupling plan, section 6).

Pure python: no torch / lada import. check() runs when the Lada engine loads its models: an unverified version is a
single log warning (and shows in the perf profile / portability doctor), it never blocks anything.
Upgrade SOP: update the lada checkout (or rebuild the release wheel from the new official commit), run the GPU
engine contract tests, fix engines/lada/ only, then add ("<base version>", "<full 40-char commit sha>") to VERIFIED:
the base is the part of LadaEngine.info().version before "+" and the sha is `git rev-parse <commit>` of the official
commit. The version a running lada reports is "<base>+<short commit>" (a git checkout computes it at import time, the
release wheel has it baked into lada.VERSION), so the check is: same base and the verified full sha starts with the
reported commit (at least 7 hex chars, case-insensitive).
"""

import re
import sys
from typing import NamedTuple

VERIFIED = (
    ("0.11.1-dev", "20cb34a20a83c72c87a991d2c949032c70085b16"),
    ("0.11.1-dev", "e4e08dc29f8f584baf403fa228ad52f003a6a328"),
)
MIN_COMMIT_CHARS = 7
_HEX = re.compile(r"[0-9a-f]+")


def _short(entry: tuple) -> str:
    return f"{entry[0]}+{entry[1][:MIN_COMMIT_CHARS]}"


class Compat(NamedTuple):
    version: str
    verified: tuple      # readable short forms, e.g. ("0.11.1-dev+20cb34a", ...)
    ok: bool

    @property
    def message(self) -> str:
        if self.ok:
            return f"lada {self.version} (verified)"
        return f"lada {self.version or 'unknown'} not verified; verified: {', '.join(self.verified)}"


_warned: list = []
_last: list = []


def _matches(version: str) -> bool:
    base, plus, commit = (version or "").partition("+")
    commit = commit.strip().lower()
    if not plus or len(commit) < MIN_COMMIT_CHARS or not _HEX.fullmatch(commit):
        return False
    return any(base == v_base and v_sha.lower().startswith(commit) for v_base, v_sha in VERIFIED)


def evaluate(version: str) -> Compat:
    """The comparison alone: no log, no state."""
    return Compat(version, tuple(_short(v) for v in VERIFIED), _matches(version))


def check(version: str, log=None) -> Compat:
    """Compare `version` with VERIFIED; an unverified one is logged once per process. Remembered for last_result()."""
    result = evaluate(version)
    _last[:] = [result]
    if not result.ok and not _warned:
        _warned.append(True)
        (log or (lambda m: print(f"[oh_player] {m}", file=sys.stderr, flush=True)))(result.message)
    return result


def last_result():
    """The result of the last check() in this process (None before the engine loaded)."""
    return _last[0] if _last else None
