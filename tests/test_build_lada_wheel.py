"""CPU-only checks for the lada wheel build helper (no git, no uv)."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_lada_wheel", ROOT / "packaging" / "build_lada_wheel.py")
build_lada_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_lada_wheel)

COMMIT = "20cb34a20a83c72c87a991d2c949032c70085b16"
SAMPLE = (
    "import os\n"
    "def _get_version(version: str):\n"
    "    return version\n"
    "\n"
    "VERSION = _get_version('0.11.1-dev')\n"
    "\n"
    "LOG_LEVEL = os.environ.get('LOG_LEVEL', 'WARNING')\n"
)


class BakeVersionTests(unittest.TestCase):
    def test_replaces_version_line_with_literal(self):
        result = build_lada_wheel.bake_version(SAMPLE, COMMIT)
        self.assertIn('VERSION = "0.11.1-dev+20cb34a"\n', result)
        self.assertNotIn("_get_version('0.11.1-dev')", result)
        # Everything else is untouched, including the helper definition.
        self.assertIn("def _get_version(version: str):", result)
        self.assertEqual(result.replace('VERSION = "0.11.1-dev+20cb34a"', "VERSION = _get_version('0.11.1-dev')"), SAMPLE)

    def test_accepts_double_quotes_and_crlf(self):
        text = 'VERSION = _get_version( "0.12.0-dev" )\r\nX = 1\r\n'
        result = build_lada_wheel.bake_version(text, COMMIT)
        self.assertEqual(result, 'VERSION = "0.12.0-dev+20cb34a"\r\nX = 1\r\n')

    def test_missing_line_fails(self):
        with self.assertRaises(ValueError):
            build_lada_wheel.bake_version("VERSION = '1.0'\n", COMMIT)

    def test_duplicate_line_fails(self):
        with self.assertRaises(ValueError):
            build_lada_wheel.bake_version(SAMPLE + "VERSION = _get_version('0.1')\n", COMMIT)

    def test_already_baked_fails(self):
        baked = build_lada_wheel.bake_version(SAMPLE, COMMIT)
        with self.assertRaises(ValueError):
            build_lada_wheel.bake_version(baked, COMMIT)

    def test_invalid_commit_fails(self):
        for bad in ("", "xyz1234", "20CB34A", "20cb"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_lada_wheel.bake_version(SAMPLE, bad)

    def test_pinned_constant_is_full_sha(self):
        self.assertRegex(build_lada_wheel.LADA_COMMIT, r"^[0-9a-f]{40}$")
        self.assertTrue(build_lada_wheel.LADA_COMMIT.startswith("20cb34a"))


if __name__ == "__main__":
    unittest.main()
