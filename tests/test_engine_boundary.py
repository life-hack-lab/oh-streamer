"""Engine boundary: only src/oh_streamer/engines/lada/ may import the upstream `lada` package or touch the
internals of its FrameRestorer pipeline (engine-decoupling plan, section 5)."""

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "oh_streamer"
ENGINE_DIR = SRC / "engines" / "lada"

LADA_INTERNALS = {
    "frame_detection_queue", "mosaic_clip_queue", "restored_clip_queue", "frame_feeder_queue",
    "_restore_frame", "_restore_clip", "mosaic_restoration_model", "mosaic_detector",
}

# Names of Lada's pipeline types / modules: outside engines/ they may appear in comments and docstrings only, never as
# identifiers (imports, names, attributes, definitions, parameters, keywords).
LADA_NAMES = {"FrameRestorer", "RestorerFactory", "ModelFiles", "STOP_MARKER", "ErrorMarker", "threading_utils",
              "video_utils"}

# file (relative to src/oh_streamer, posix) -> internal attribute names still used there. Phase 3 (the upper layers
# switch to the engine interface) removes every entry.
WHITELIST: dict[str, set[str]] = {
    # (empty: phase 2 routed every remaining use through engines/lada/probes.py)
}


def _py_files():
    for p in sorted(SRC.rglob("*.py")):
        if ENGINE_DIR not in p.parents:
            yield p


def _rel(p: Path) -> str:
    return p.relative_to(SRC).as_posix()


class EngineBoundaryTests(unittest.TestCase):
    def test_lada_imported_only_by_engine_package(self):
        bad = []
        for p in _py_files():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                if any(n == "lada" or n.startswith("lada.") for n in names):
                    bad.append(f"{_rel(p)}:{node.lineno}")
        self.assertEqual([], bad)

    def test_lada_internals_only_in_engine_package(self):
        found: dict[str, set[str]] = {}
        for p in _py_files():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and node.attr in LADA_INTERNALS:
                    found.setdefault(_rel(p), set()).add(node.attr)
                elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                      and node.value in LADA_INTERNALS):          # getattr(obj, "name") / setattr
                    found.setdefault(_rel(p), set()).add(node.value)
        unexpected = {f: sorted(n - WHITELIST.get(f, set())) for f, n in found.items() if n - WHITELIST.get(f, set())}
        self.assertEqual({}, unexpected)
        stale = {f: sorted(n - found.get(f, set())) for f, n in WHITELIST.items() if n - found.get(f, set())}
        self.assertEqual({}, stale, "whitelist entries no longer needed: remove them")

    def test_lada_names_are_not_identifiers_outside_engine_package(self):
        found: dict[str, set[str]] = {}
        for p in _py_files():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, ast.Attribute):
                    names = [node.attr]
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names = [node.name]
                elif isinstance(node, ast.arg):
                    names = [node.arg]
                elif isinstance(node, ast.keyword) and node.arg:
                    names = [node.arg]
                elif isinstance(node, ast.alias):
                    names = [node.name.split(".")[-1], node.asname or ""]
                elif isinstance(node, ast.ImportFrom):
                    names = (node.module or "").split(".")
                for n in names:
                    if n in LADA_NAMES:
                        found.setdefault(_rel(p), set()).add(n)
        self.assertEqual({}, {f: sorted(n) for f, n in found.items()})


if __name__ == "__main__":
    unittest.main()
