"""Compatibility checks for legacy callers and injected player runtime operations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / 'src'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from fractions import Fraction
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oh_streamer.player import oh_player as L


class PlayerSplitCompatibilityTest(unittest.TestCase):
    def test_controller_dependencies_resolve_callbacks_after_construction(self):
        deps = L._controller_dependencies()
        player = object()
        with patch.object(L, 'start_mpv', return_value=player) as start:
            self.assertIs(deps.start_mpv('mpv', 'ipc', 'title', 1.0), player)
        start.assert_called_once_with('mpv', 'ipc', 'title', 1.0)
        self.assertIs(deps.info_panel, L.info_panel)

    def test_legacy_import_is_the_same_facade_module(self):
        self.assertIs(importlib.import_module('oh_player'), L)

    def test_session_uses_runtime_callbacks_patched_after_construction(self):
        source = SimpleNamespace(src='clip.mp4', audio_src='audio.mp4', has_audio=True,
                                 fmt_start=1.0, http=None,
                                 meta=SimpleNamespace(video_fps_exact=Fraction(25)))
        session = L.Session(object(), source, 'libx264', None)
        muxer = object()
        messages = []
        with patch.object(L, 'free_port', return_value=1234), \
                patch.object(L, 'start_muxer', return_value=muxer) as start, \
                patch.object(L, 'log', messages.append):
            self.assertTrue(session._open_muxer(2.0, 0))
        self.assertIs(session.muxer, muxer)
        self.assertEqual(start.call_args.args[3], 1.0)
        self.assertEqual(start.call_args.kwargs['ts_offset'], 2.0)
        self.assertTrue(messages)

    def test_open_source_uses_facade_resolution_and_probe_callbacks(self):
        resolved, opened, args = object(), object(), object()
        with patch.object(L, 'resolve_input', return_value=resolved) as resolve, \
                patch.object(L, 'probe_input', return_value=opened) as probe:
            self.assertIs(L.open_source('clip.mp4', {'X-Test': 'header'}, args, 720, False), opened)
        resolve.assert_called_once_with('clip.mp4', {'X-Test': 'header'}, args, 720, False)
        probe.assert_called_once_with(resolved)


if __name__ == '__main__':
    unittest.main()
