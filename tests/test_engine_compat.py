"""engines/lada/compat.py: verified lada versions (pure python, no lada import)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import unittest

from oh_streamer.engines.lada import compat


class CompatTests(unittest.TestCase):

    def setUp(self):
        warned, last = list(compat._warned), list(compat._last)
        compat._warned.clear()
        compat._last.clear()
        self.addCleanup(lambda: (compat._warned.__setitem__(slice(None), warned), compat._last.__setitem__(slice(None), last)))

    def test_verified_version_passes_silently(self):
        logs = []
        r = compat.check("0.11.1-dev+20cb34a", logs.append)
        self.assertTrue(r.ok)
        self.assertEqual(logs, [])
        self.assertIs(compat.last_result(), r)

    def test_unverified_version_warns_once_and_never_raises(self):
        logs = []
        r = compat.check("0.12.0+abcdef0", logs.append)
        self.assertFalse(r.ok)
        compat.check("0.12.0+abcdef0", logs.append)
        self.assertEqual(len(logs), 1)
        self.assertIn("lada 0.12.0+abcdef0 not verified; verified: 0.11.1-dev+20cb34a, 0.11.1-dev+e4e08dc", logs[0])
        self.assertEqual(r.verified, ("0.11.1-dev+20cb34a", "0.11.1-dev+e4e08dc"))

    def test_a_missing_commit_is_not_verified(self):
        self.assertFalse(compat.evaluate("0.11.1-dev").ok)
        self.assertFalse(compat.evaluate("").ok)
        self.assertIn("unknown", compat.evaluate("").message)

    def test_verified_entries_are_base_and_full_sha(self):
        for base, sha in compat.VERIFIED:
            self.assertEqual(base, "0.11.1-dev")
            self.assertRegex(sha, r"^[0-9a-f]{40}$")

    def test_short_and_longer_prefixes_match(self):
        for v in ("0.11.1-dev+20cb34a", "0.11.1-dev+e4e08dc", "0.11.1-dev+20cb34a20", "0.11.1-dev+e4e08dc29f8f584baf403fa228ad52f003a6a328"):
            self.assertTrue(compat.evaluate(v).ok, v)
        self.assertEqual(compat.evaluate("0.11.1-dev+20cb34a").message, "lada 0.11.1-dev+20cb34a (verified)")

    def test_uppercase_hex_matches(self):
        self.assertTrue(compat.evaluate("0.11.1-dev+20CB34A").ok)
        self.assertTrue(compat.evaluate("0.11.1-dev+E4E08DC29F").ok)

    def test_wrong_base_is_not_verified(self):
        self.assertFalse(compat.evaluate("0.11.2+20cb34a").ok)
        self.assertFalse(compat.evaluate("0.11.1+20cb34a").ok)

    def test_unknown_or_too_short_or_malformed_commit_is_not_verified(self):
        for v in ("0.11.1-dev+abcdef0", "0.11.1-dev+20cb34", "0.11.1-dev+20cb3", "0.11.1-dev+", "0.11.1-dev+20cb34z",
                  "0.11.1-dev+20cb34a20a83c72c87a991d2c949032c70085b16f", "0.11.1-dev+20cb34a+x", "+20cb34a"):
            self.assertFalse(compat.evaluate(v).ok, v)

    def test_no_plus_and_empty_are_not_verified(self):
        self.assertFalse(compat.evaluate("0.11.1-dev").ok)
        self.assertFalse(compat.evaluate("").ok)
        self.assertFalse(compat.evaluate(None).ok)

    def test_evaluate_has_no_side_effects(self):
        compat.evaluate("1.0")
        self.assertIsNone(compat.last_result())


if __name__ == "__main__":
    unittest.main()
