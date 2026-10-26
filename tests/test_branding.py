"""The display name comes from queue_ui/branding.js; bad content falls back safely."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer import branding


class BrandingTests(unittest.TestCase):
    def test_shipped_file_parses_to_non_empty_name(self):
        text = (branding.resource_root() / branding.BRANDING_RELATIVE).read_text(encoding="utf-8")
        parsed = branding.parse_branding(text)
        self.assertEqual(parsed["name"], branding.APP_NAME)
        self.assertEqual(parsed["logo"], branding.LOGO_TEXT)
        self.assertNotEqual(branding.APP_NAME, "")

    def test_parses_object(self):
        parsed = branding.parse_branding('// c\nwindow.BRAND = {"name": "X Player", "logo": "XP"};\n')
        self.assertEqual(parsed, {"name": "X Player", "logo": "XP"})

    def test_bad_content_falls_back(self):
        for text in ("", "nothing here", "window.BRAND = {name: 'x'};", 'window.BRAND = {"name": 3};'):
            self.assertEqual(branding.parse_branding(text)["name"], branding.FALLBACK_NAME, text)


if __name__ == "__main__":
    unittest.main()
