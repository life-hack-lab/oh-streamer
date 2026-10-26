"""Contracts for source, wheel, and frozen runtime assets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from oh_streamer import paths


RESOURCE_ROOT = ROOT / "src" / "oh_streamer" / "resources"


class PackagedResourceTests(unittest.TestCase):
    def test_source_roots_keep_assets_separate_from_checkout_and_upstream(self):
        with patch.dict(os.environ, {"OH_STREAMER_PROJECT_ROOT": ""}):
            self.assertEqual(paths.resource_root(), RESOURCE_ROOT.resolve())
            self.assertEqual(paths.checkout_root(), ROOT.resolve())
            self.assertEqual(paths.upstream_root(), ROOT.parent.resolve())

    def test_frozen_resources_live_under_bundle_resources_directory(self):
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "_MEIPASS", "/bundle/root", create=True), \
                patch.object(sys, "platform", "darwin"), \
                patch.dict(os.environ, {}, clear=True):
            bundle_root = Path("/bundle/root").resolve()
            self.assertEqual(paths.resource_root(), bundle_root / "resources")
            self.assertEqual(paths.checkout_root(), bundle_root)
            self.assertEqual(paths.upstream_root(), bundle_root.parent)

    def test_explicit_upstream_override_is_preserved(self):
        with patch.dict(os.environ, {"OH_STREAMER_PROJECT_ROOT": "~/external/lada"}):
            self.assertEqual(paths.upstream_root(), Path("~/external/lada").expanduser().resolve())

    def test_wheel_declares_all_immutable_runtime_asset_types(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = set(project["tool"]["setuptools"]["package-data"]["oh_streamer"])
        self.assertTrue({
            "resources/*.lua",
            "resources/*.json",
            "resources/queue_ui/*.html",
            "resources/queue_ui/*.css",
            "resources/queue_ui/*.js",
            "resources/icons/*.ico",
            "resources/icons/*.icns",
            "resources/icons/*.png",
        }.issubset(patterns))
        for relative in (
            "oh_seek.lua", "oh_compare_wipe.lua", "oh_sub_combos.lua", "resolvers.default.json",
            "resolvers.example.json", "queue_ui/index.html", "queue_ui/app.js", "queue_ui/app.css", "queue_ui/app-modern.css", "queue_ui/branding.js",
            "icons/app.ico", "icons/app.icns", "icons/app.png",
        ):
            self.assertTrue((RESOURCE_ROOT / relative).is_file(), relative)

    def test_default_site_resolvers_use_the_package_browser_module(self):
        defaults = json.loads((RESOURCE_ROOT / "resolvers.default.json").read_text(encoding="utf-8"))
        site_rules = [rule for rule in defaults if rule.get("command")]
        self.assertTrue(site_rules)
        for rule in site_rules:
            self.assertEqual(rule["command"], [
                "{python}", "-m", "oh_streamer.sources.resolvers.resolve_browser",
            ], rule.get("match"))


if __name__ == "__main__":
    unittest.main()
