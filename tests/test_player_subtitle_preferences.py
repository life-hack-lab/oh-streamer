"""mpv subtitle selection preferences without launching the player."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from oh_streamer.player import oh_player as L


class PlayerSubtitlePreferenceTests(unittest.TestCase):
    def test_start_mpv_prefers_traditional_chinese_then_chinese(self):
        with patch.object(L.subprocess, "Popen") as popen, \
                patch.object(L, "primary_display_height", return_value=None):
            L.start_mpv("mpv", "ipc", "title", 1.0)

        command = popen.call_args.args[0]
        self.assertIn("--slang=chi,zho,zh", command)
        self.assertTrue(any(arg.endswith("oh_subtitle_preferences.lua") for arg in command))


if __name__ == "__main__":
    unittest.main()
