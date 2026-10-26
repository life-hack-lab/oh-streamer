"""Lada model catalog: the registry is located without importing lada/torch, and weights come from LADA_MODEL_WEIGHTS_DIR."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.engines import base
from oh_streamer.engines.lada import catalog

# Same shape as lada/lada/__init__.py's ModelFiles (the parts the catalog parses with ast).
FAKE_REGISTRY = '''
import os
MODEL_WEIGHTS_DIR = os.environ.get("LADA_MODEL_WEIGHTS_DIR", "model_weights")
raise RuntimeError("the catalog must never execute this module")

class ModelFiles:
    _WELL_KNOWN_RESTORATION_MODELS = [
        ModelFile('basicvsrpp-v1.2', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth')),
        ModelFile('deepmosaics', None, os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'clean_youknow_video.pth')),
    ]
    _WELL_KNOWN_DETECTION_MODELS = [
        ModelFile('v4-fast', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v4_fast.pt')),
        ModelFile('v4-accurate', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v4_accurate.pt')),
    ]
'''


def put(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def fill_weights(weights: Path) -> None:
    put(weights / "lada_mosaic_detection_model_v4_fast.pt")
    put(weights / "lada_mosaic_detection_model_custom.pt")                  # not in the registry: picked up by prefix
    put(weights / "lada_mosaic_restoration_model_generic_v1.2.pth")
    put(weights / "lada_mosaic_restoration_model_mine.pth")                 # custom -> basicvsrpp-mine
    put(weights / "3rd_party" / "clean_youknow_video.pth")
    put(weights / "README.txt")


EXPECTED = base.ModelCatalog(("v4-fast", "custom"), ("basicvsrpp-v1.2", "deepmosaics", "basicvsrpp-mine"))


class CatalogLocationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "install_root"       # has no lada/ folder at all
        self.root.mkdir()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LADA_MODEL_WEIGHTS_DIR", None)
        patch_root = mock.patch.object(catalog, "upstream_root", return_value=self.root)
        patch_root.start()
        self.addCleanup(patch_root.stop)

    def test_found_via_find_spec_with_weights_dir_env_and_no_root_lada(self):
        site = self.tmp / "site-packages"
        put(site / "lada" / "__init__.py", FAKE_REGISTRY)
        weights = self.tmp / "model_weights"
        fill_weights(weights)
        os.environ["LADA_MODEL_WEIGHTS_DIR"] = str(weights)
        self.assertFalse((self.root / "lada").exists())
        modules = {k: v for k, v in sys.modules.items() if k != "lada" and not k.startswith("lada.")}
        with mock.patch.object(sys, "path", [str(site)] + sys.path), mock.patch.dict(sys.modules, modules, clear=True):
            importlib.invalidate_caches()
            self.assertEqual(catalog._registry_file(), site / "lada" / "__init__.py")
            self.assertEqual(catalog.installed_models(), EXPECTED)
            self.assertNotIn("lada", sys.modules)       # located, never imported (the fake raises on execution)
            self.assertNotIn("torch", set(sys.modules) - set(modules))
        importlib.invalidate_caches()

    def test_find_spec_does_not_import_the_real_lada_or_torch(self):
        before = {"lada" in sys.modules, "torch" in sys.modules}
        if True in before:
            self.skipTest("lada/torch already imported by another test in this process")
        importlib.util.find_spec("lada")
        self.assertNotIn("lada", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_fallback_to_root_checkout_when_lada_not_importable(self):
        put(self.root / "lada" / "lada" / "__init__.py", FAKE_REGISTRY)
        fill_weights(self.root / "lada" / "model_weights")                     # default weights dir, env var unset
        with mock.patch.object(catalog.importlib.util, "find_spec", return_value=None):
            self.assertEqual(catalog._registry_file(), self.root / "lada" / "lada" / "__init__.py")
            self.assertEqual(catalog.installed_models(), EXPECTED)

    def test_fallback_when_spec_has_no_origin_or_lookup_fails(self):
        put(self.root / "lada" / "lada" / "__init__.py", FAKE_REGISTRY)
        fallback = self.root / "lada" / "lada" / "__init__.py"
        namespace_like = mock.Mock(origin=None)
        with mock.patch.object(catalog.importlib.util, "find_spec", return_value=namespace_like):
            self.assertEqual(catalog._registry_file(), fallback)
        with mock.patch.object(catalog.importlib.util, "find_spec", side_effect=ValueError("lada.__spec__ is None")):
            self.assertEqual(catalog._registry_file(), fallback)

    def test_importable_lada_wins_over_root_checkout(self):
        put(self.root / "lada" / "lada" / "__init__.py", "this is not python (")
        site = self.tmp / "site"
        installed = put(site / "lada" / "__init__.py", FAKE_REGISTRY)
        with mock.patch.object(catalog.importlib.util, "find_spec", return_value=mock.Mock(origin=str(installed))):
            self.assertEqual(catalog._registry_file(), installed)

    def test_empty_catalog_when_no_registry_anywhere(self):
        os.environ["LADA_MODEL_WEIGHTS_DIR"] = str(self.tmp / "nowhere")
        with mock.patch.object(catalog.importlib.util, "find_spec", return_value=None):
            self.assertEqual(catalog.installed_models(), base.ModelCatalog((), ()))

    def test_missing_weights_dir_lists_nothing(self):
        put(self.root / "lada" / "lada" / "__init__.py", FAKE_REGISTRY)
        os.environ["LADA_MODEL_WEIGHTS_DIR"] = str(self.tmp / "absent")
        with mock.patch.object(catalog.importlib.util, "find_spec", return_value=None):
            self.assertEqual(catalog.installed_models(), base.ModelCatalog((), ()))


if __name__ == "__main__":
    unittest.main()
