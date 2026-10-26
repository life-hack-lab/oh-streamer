"""Platform selection and launch fallback for the shared browser resolver."""
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
from unittest.mock import MagicMock

from oh_streamer.sources.resolvers import browser


class BrowserOptionsTest(unittest.TestCase):
    def test_macos_prefers_chrome_and_uses_mac_user_agent(self):
        user_agent, channels = browser.browser_options("darwin")
        self.assertIn("Macintosh; Intel Mac OS X", user_agent)
        self.assertEqual(channels, ["chrome", "msedge", None])

    def test_windows_preferences_remain_unchanged(self):
        user_agent, channels = browser.browser_options("win32")
        self.assertIn("Windows NT 10.0", user_agent)
        self.assertEqual(channels, ["msedge", "chrome", None])

    def test_browser_fallback_and_persistent_profile(self):
        playwright = MagicMock()
        playwright.chromium.launch.side_effect = [RuntimeError("Edge missing"), "Chrome"]
        self.assertEqual(browser.launch_browser(playwright, channels=["msedge", "chrome", None]), "Chrome")
        self.assertEqual(playwright.chromium.launch.call_args.kwargs["channel"], "chrome")
        browser.launch_browser(playwright, channels=["chrome"], profile_dir="profile", user_agent="UA")
        self.assertEqual(playwright.chromium.launch_persistent_context.call_args.kwargs["user_data_dir"], "profile")
        self.assertEqual(playwright.chromium.launch_persistent_context.call_args.kwargs["user_agent"], "UA")


if __name__ == "__main__":
    unittest.main()
