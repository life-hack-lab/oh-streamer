"""Check canonical package imports and their flat compatibility aliases."""
from pathlib import Path
import sys
import unittest
import importlib


STREAMER_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = STREAMER_ROOT / "src"
for import_root in (STREAMER_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


# Canonical package modules; the real implementations must live under src/.
CANONICAL_MODULES = (
    "oh_streamer.player.single_instance",
    "oh_streamer.queue.queue_ipc",
    "oh_streamer.processing.detect_gate",
    "oh_streamer.sources.qualities",
    "oh_streamer.sources.source_resolve",
    "oh_streamer.sources.resolvers.resolve_browser",
    "oh_streamer.platform.platform_support",
)
# Only the root shims that stay after the public cleanup (oh_player / queue_app are launcher entry points).
MODULE_ALIASES = (
    ("oh_streamer.sources.resolvers.resolve_browser", "resolvers.resolve_browser"),
)


class PackageImportTests(unittest.TestCase):
    def test_package_modules_are_the_implementations(self):
        for canonical_name in CANONICAL_MODULES:
            with self.subTest(module=canonical_name):
                module = importlib.import_module(canonical_name)
                module_path = Path(module.__file__).resolve()
                self.assertTrue(module_path.is_relative_to(SOURCE_ROOT.resolve()), module_path)

    def test_flat_imports_are_module_identity_aliases(self):
        for canonical_name, legacy_name in MODULE_ALIASES:
            with self.subTest(module=canonical_name):
                canonical = importlib.import_module(canonical_name)
                legacy = importlib.import_module(legacy_name)
                self.assertIs(legacy, canonical)


if __name__ == "__main__":
    unittest.main()
