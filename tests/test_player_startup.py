"""Offline unit test for the persistent player's startup (run: python -P streamer/tests/test_player_startup.py).

Imports oh_player (torch + lada, no models / GPU / mpv needed): run_player is driven with fake mpv / IPC / cast /
GPU monitor / opener / session objects."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import os
import sys
import unittest
from unittest import mock
import oh_streamer.player.oh_player as L  # noqa: E402


class FakePlayer:
    """The mpv process: runs until the test says it has seen enough."""

    def __init__(self):
        self.pid, self.exit = 4242, False

    def poll(self):
        return 0 if self.exit else None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.exit = True


class FakeIpc:
    """Property store standing in for mpv's IPC; acknowledges oh_seek.lua's reset like the script does."""

    def __init__(self, *_a):
        self.props, self.sets, self.commands = {}, [], []

    def get(self, name, default=None):
        return self.props.get(name, default)

    def set(self, name, value):
        self.sets.append((name, value))
        self.props[name] = value

    def command(self, *a):
        self.commands.append(a)
        if a[:2] == ("script-message", "oh-reset"):
            self.props["user-data/oh/reset-ack"] = a[2]

    def close(self):
        pass


class FakeCast:
    active = None

    def __init__(self, *_a):
        pass

    def poll(self):
        pass

    def close(self):
        pass


class FakeGpu:
    def start(self):
        pass

    def set_active(self, _on):
        pass

    def snapshot(self):
        return None

    def close(self):
        pass


class FakeMeta:
    duration, video_width, video_height, video_fps_exact = 60.0, 1920, 1080, 30


class FakeSource:
    meta = FakeMeta()

    def describe(self):
        return "fake source"


class FakeSession:
    instances = []

    def __init__(self, restorer, source, encoder, max_frames):
        self.source, self.meta, self.encoder = source, source.meta, encoder
        self.target, self.url, self.stats, self.doubler = 0.0, "tcp://127.0.0.1:1", None, None
        self.frames_total, self.fps = 0, 30.0
        self.started = self.stopped = False
        FakeSession.instances.append(self)

    def start(self, start, gate_release_s=0.0):
        self.started, self.target = True, start

    def ready(self):
        return False

    def stop(self, fast=False):
        self.stopped = True
        return 0.0

    def release_restorer(self):
        pass


class FakeQualityControl:
    def __init__(self, quals, chosen, *_a):
        self.qualities, self.current = quals, chosen


class FakeLada:
    def open(self, source):
        return object()


class FakeSettings:
    cfg = {"interpolation": "off", "vsr": "off", "smooth_playback": False, "encoder": "libx264"}
    override_base = {}      # per-source overrides in effect (SettingsControl): none

    def menu(self):
        return [{"key": "fake"}]

    def resolve_encoder(self, name):
        return name


class ModelsFinishWhileResolving:
    """Opener whose first result arrives and whose on-demand model load (Engine.ensure_loaded, called by the player
    for a source that needs the engine) finishes in the same loop iteration: after the loop synced its settings
    controller and before it checks engine.ready."""

    def __init__(self, engine, opened, player):
        self.engine, self.opened, self.player, self.polls = engine, opened, player, 0
        self.busy = False

        def ensure_loaded():
            self.engine.lada, self.engine.sc = FakeLada(), FakeSettings()   # what Engine.load() does,
            self.engine.ready.set()                                               # in that order
            self.loads += 1
        self.loads = 0
        engine.ensure_loaded = ensure_loaded

    def submit(self, *_a, **_k):
        pass

    def poll(self):
        self.polls += 1
        if self.polls == 1:
            return self.opened, None, 0.1, "命令列"
        self.player.exit = True   # the next iteration is the last
        return None


class TestFirstSourceWhileModelsLoad(unittest.TestCase):
    def test_source_resolved_as_models_finish_starts(self):
        args = argparse.Namespace(mpv=None, initial_prebuffer=1.0, prebuffer=3.0, seek_prebuffer=1.0, duration=0.0,
                                  thumbnails=False, settings_file=None, output="play")
        cfg = {"interpolation": "off", "vsr": "off", "smooth_playback": False}
        engine = L.Engine(args, {}, cfg, ["det"], ["res"])
        player, ipcs, logs = FakePlayer(), [], []
        opened = L.Opened(quals=[{"id": "source", "label": "1080p"}], chosen={"id": "source", "label": "1080p"},
                          source=FakeSource(), title="restore_bench.mp4", source_text="restore_bench.mp4")
        FakeSession.instances.clear()

        def make_ipc(*a):
            ipcs.append(FakeIpc(*a))
            return ipcs[-1]

        with mock.patch.object(L, "find_mpv", lambda _m: "mpv"), \
                mock.patch.object(L, "start_mpv", lambda *a, **k: player), \
                mock.patch.object(L, "MpvIpc", make_ipc), \
                mock.patch.object(L, "CastManager", FakeCast), \
                mock.patch.object(L.info_panel, "GpuMonitor", FakeGpu), \
                mock.patch.object(L, "Opener", lambda *_a: ModelsFinishWhileResolving(engine, opened, player)), \
                mock.patch.object(L, "Session", FakeSession), \
                mock.patch.object(L, "QualityControl", FakeQualityControl), \
                mock.patch.object(L, "log", logs.append):
            self.assertEqual(L.run_player(args, engine), 0)

        self.assertFalse([m for m in logs if m.startswith("could not start")], logs)
        self.assertEqual(len(FakeSession.instances), 1)
        sess = FakeSession.instances[0]
        self.assertTrue(sess.started)
        ipc = ipcs[0]
        self.assertIn(("user-data/oh/settings", '[{"key": "fake"}]'), ipc.sets)
        self.assertEqual([v for k, v in ipc.sets if k == "user-data/oh/active"][-1], "1")
        self.assertTrue(any(m.startswith("opened 'restore_bench.mp4'") for m in logs), logs)


if __name__ == "__main__":
    unittest.main()
