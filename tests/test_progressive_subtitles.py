"""CPU-only progressive external subtitle parsing, clock mapping and mpv lifecycle."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.player import progressive_stream as P
from oh_streamer.player import queue_bridge as QB
from oh_streamer.player.controller import PlayerController


class ProgressiveSubtitleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "zh.srt"
        self.path.write_text("1\n00:00:01,000 --> 00:00:02,500\n<i>你好</i> 00:00:01,000\n", encoding="utf-8")
        self.track = dict(path=str(self.path), lang="chi", title="繁體中文（AI）", default=True)

    def open(self, **extra):
        return P.progressive_from_message(dict(input=str(self.path), work_dir=self.tmp.name,
                                               progressive=True, **extra),
            probe=lambda _: dict(fps=25, video_start=0, frames=100, duration=4, has_audio=True))

    def test_optional_payload_validation_does_not_reject_video(self):
        self.assertEqual(self.open(subtitles=[self.track, self.track]).subtitles, [self.track])
        for invalid in (None, {}, "x", [None], [dict(self.track, path="missing")],
                        [dict(self.track, default="yes")], [dict(self.track, lang=2)]):
            self.assertEqual(self.open(subtitles=invalid).subtitles, [])
        for offset in (True, "1.4", float("nan"), float("inf")):
            self.assertEqual(self.open(subtitles=[self.track], subtitle_offset=offset).subtitles, [])
        po = self.open(subtitles=[self.track], subtitle_offset=1.4, subtitle_error=True)
        self.assertEqual(po.subtitle_offset, 1.4)
        self.assertTrue(po.subtitle_error)
        partial = self.open(subtitle_covered_s=2.5, subtitle_complete=False)
        self.assertEqual(partial.subtitle_covered_s, 2.5)
        self.assertFalse(partial.subtitle_complete)

    def test_initial_open_keeps_island_coverage_and_ignores_malformed_optional_ranges(self):
        po = self.open(subtitle_covered_s=0, subtitle_complete=False,
                       subtitle_covered_ranges=[[2390, 2420]])
        self.assertEqual(po.subtitle_covered_ranges, [[2390.0, 2420.0]])
        for invalid in ({}, [[0, float("nan")]], [[10, 5]]):
            self.assertIsNone(self.open(subtitle_covered_ranges=invalid).subtitle_covered_ranges)

    def test_shift_preserves_dialogue_and_source(self):
        original = self.path.read_text(encoding="utf-8")
        shifted = P.shifted_srt(str(self.path), 1.4, self.tmp.name, 0)
        text = Path(shifted).read_text(encoding="utf-8")
        self.assertIn("00:00:02,400 --> 00:00:03,900", text)
        self.assertIn("<i>你好</i> 00:00:01,000", text)
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)
        self.assertEqual(P.shifted_srt(str(self.path), 0, self.tmp.name, 0), str(self.path))

    def controller(self):
        c = PlayerController.__new__(PlayerController)
        c._deps = SimpleNamespace(progressive_stream=P, tempfile=tempfile, shutil=shutil, os=os, log=Mock(),
                                  _float_or=lambda value, default=0.0: default if value is None else float(value))
        c.progressive = SimpleNamespace(po=self.open(queue_id="q1", subtitles=[self.track], subtitle_offset=1.4),
                                        feeder=SimpleNamespace(url="tcp://first"))
        c.processed = None
        c.progressive_sub_dir = None
        c.progressive_sub_loaded = None
        c.progressive_sub_selection = None
        c.progressive_sub_visibility = None
        c.pending_subtitle_updates = {}
        c.subtitle_gap_item_id = None
        c.subtitle_gap_warned = False
        c.osd = Mock()
        c.link = Mock()
        c.processed_clock_next = 0.0
        c.ipc = Mock()
        c.ipc.get.return_value = []
        self.addCleanup(lambda: shutil.rmtree(c.progressive_sub_dir, ignore_errors=True)
                        if c.progressive_sub_dir else None)
        return c

    def test_one_attach_per_load_and_seek_reloads_without_duplicates(self):
        c = self.controller()
        c.load_progressive_subtitles("tcp://first")
        c.load_progressive_subtitles("tcp://first")
        self.assertEqual(c.ipc.command.call_count, 1)
        self.assertEqual(c.ipc.command.call_args.args[2:], ("select", self.track["title"], "chi"))
        c.load_progressive_subtitles("tcp://after-seek")
        self.assertEqual(c.ipc.command.call_count, 2)
        self.assertFalse(any(call.args[0] == "sub-delay" for call in c.ipc.set.call_args_list))

    def test_failure_is_logged_and_disabled_selection_survives(self):
        c = self.controller()
        c.progressive_sub_selection = False
        c.progressive_sub_visibility = False  # preserve a visibility choice previously made by the user
        c.ipc.command.side_effect = OSError("missing subtitle")
        c.load_progressive_subtitles("tcp://first")
        failures = [call for call in c._deps.log.call_args_list if not call.args[0].startswith("subtitle timing:")]
        self.assertEqual(len(failures), 1)
        c.ipc.set.assert_any_call("sid", "no")
        c.ipc.set.assert_any_call("sub-visibility", False)

    def test_concise_preferred_and_explicit_full_selection_survives_seek(self):
        c = self.controller()
        full = dict(self.track, title="繁體中文（AI・完整）", default=True)
        concise = dict(self.track, title="繁體中文（AI・精簡）", default=False)
        c.progressive.po.subtitles = [full, concise]
        c.load_progressive_subtitles("tcp://first")
        self.assertEqual([call.args[2] for call in c.ipc.command.call_args_list], ["auto", "select"])
        c.ipc.command.reset_mock()
        c.progressive_sub_selection = (full["title"], full["lang"])
        c.load_progressive_subtitles("tcp://seek")
        self.assertEqual([call.args[2] for call in c.ipc.command.call_args_list], ["select", "auto"])

    def test_existing_external_track_is_not_duplicated(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.ipc.get.return_value = [{"type": "sub", "external-filename": str(self.path)}]
        c.load_progressive_subtitles("tcp://first")
        c.ipc.command.assert_not_called()

    def test_dynamic_update_reloads_existing_track_and_preserves_user_selection(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.progressive_sub_loaded = "tcp://first"
        current = {"type": "sub", "id": 7, "external": True, "external-filename": str(self.path),
                   # mpv reports the filename as title for an external track despite sub-add metadata.
                   "title": self.path.name, "lang": "chi"}
        props = {"path": "tcp://first", "track-list": [current], "sid": 7}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(update)

        c.ipc.command.assert_called_once_with("sub-reload", 7)
        self.assertEqual(c.progressive_sub_selection, (self.track["title"], "chi"))
        self.assertEqual(c.progressive.po.subtitle_covered_s, 3.0)
        self.assertFalse(c.progressive.po.subtitle_complete)

    def test_dynamic_sub_reload_restores_selection_after_mpv_reassigns_track_ids(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.progressive.po.subtitles = [self.track]
        c.progressive_sub_loaded = "tcp://first"
        ja_path = Path(self.tmp.name) / "ja.srt"
        ja_path.write_text("1\n00:00:01,000 --> 00:00:02,000\n日本語\n", encoding="utf-8")
        ja = {"path": str(ja_path), "lang": "jpn", "title": "日本語（AI）", "default": False}
        c.progressive.po.subtitles.append(ja)
        zh_loaded = {"type": "sub", "id": 7, "external": True, "external-filename": str(self.path),
                     "title": self.path.name, "lang": "chi"}
        ja_loaded = {"type": "sub", "id": 8, "external": True, "external-filename": str(ja_path),
                     "title": ja_path.name, "lang": "jpn"}
        props = {"path": "tcp://first", "track-list": [zh_loaded, ja_loaded], "sid": 7,
                 "sub-visibility": True}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        reload_count = [0]

        def command(*args):
            if args[0] == "sub-reload":
                reload_count[0] += 1
                zh_id, ja_id = 10 + reload_count[0] * 2 - 1, 10 + reload_count[0] * 2
                props["track-list"] = [dict(zh_loaded, id=zh_id), dict(ja_loaded, id=ja_id)]
                # Simulate mpv switching to the other subtitle after reloading the old selected track.
                props["sid"] = ja_id
        c.ipc.command.side_effect = command
        c.ipc.set.side_effect = lambda name, value: props.__setitem__(name, value)
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": [str(self.path), str(ja_path)],
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track, ja],
        })

        c.apply_subtitle_update(update)

        self.assertEqual(reload_count[0], 2)
        self.assertEqual(props["sid"], 13)
        c.ipc.set.assert_any_call("sid", 13)

    def test_dynamic_update_keeps_subtitles_disabled(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.progressive_sub_loaded = "tcp://first"
        current = {"type": "sub", "id": 7, "external": True, "external-filename": str(self.path),
                   "title": self.track["title"], "lang": "chi"}
        props = {"path": "tcp://first", "track-list": [current], "sid": "no"}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        def remove_track(*args):
            if args[0] == "sub-remove":
                props["track-list"] = []
        c.ipc.command.side_effect = remove_track
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(update)

        commands = [call.args for call in c.ipc.command.call_args_list]
        self.assertIn(("sub-remove", 7), commands)
        self.assertIn(("sub-add", str(self.path), "auto", self.track["title"], "chi"), commands)
        c.ipc.set.assert_any_call("sid", "no")
        self.assertNotIn(("sub-visibility", False), [call.args for call in c.ipc.set.call_args_list])
        self.assertIs(c.progressive_sub_selection, False)

    def test_selecting_track_after_disabled_update_restores_visible_track_on_next_update(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.progressive_sub_loaded = "tcp://first"
        original = {"type": "sub", "id": 7, "external": True, "external-filename": str(self.path),
                    "title": self.track["title"], "lang": "chi"}
        props = {"path": "tcp://first", "track-list": [original], "sid": "no", "sub-visibility": True}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        def command(*args):
            if args[0] == "sub-remove":
                props["track-list"] = []
            elif args[0] == "sub-add":
                props["track-list"] = [dict(original, id=8, **{"external-filename": args[1]})]
        c.ipc.command.side_effect = command
        first = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(first)

        self.assertIs(c.progressive_sub_selection, False)
        self.assertTrue(props["sub-visibility"])
        # Simulate selecting the track through mpv's track menu; visibility is untouched.
        props["sid"] = 8
        c.ipc.command.reset_mock()
        second = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 4.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(second)

        self.assertEqual(c.progressive_sub_selection, (self.track["title"], "chi"))
        self.assertTrue(props["sub-visibility"])
        self.assertNotIn(("sub-visibility", False), [call.args for call in c.ipc.set.call_args_list])
        self.assertIn(("sub-reload", 8), [call.args for call in c.ipc.command.call_args_list])

    def test_dynamic_update_preserves_hidden_visibility_with_selected_track(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.progressive_sub_loaded = "tcp://first"
        current = {"type": "sub", "id": 7, "external": True, "external-filename": str(self.path),
                   "title": self.track["title"], "lang": "chi"}
        props = {"path": "tcp://first", "track-list": [current], "sid": 7, "sub-visibility": False}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(update)

        self.assertEqual(c.progressive_sub_selection, (self.track["title"], "chi"))
        self.assertFalse(c.progressive_sub_visibility)
        c.ipc.set.assert_any_call("sub-visibility", False)

    def test_dynamic_update_also_reloads_processed_queue_output(self):
        c = self.controller()
        c.progressive = None
        c.processed = SimpleNamespace(queue_id="q1", path="output.mkv", subtitles=[self.track],
                                      subtitle_offset=0.0, subtitle_covered_s=2.0, subtitle_complete=False)
        c.progressive_sub_loaded = "output.mkv"
        current = {"type": "sub", "id": 9, "external": True, "external-filename": str(self.path),
                   "title": self.track["title"], "lang": "chi"}
        props = {"path": "output.mkv", "track-list": [current], "sid": 9}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 4.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(update)

        c.ipc.command.assert_called_once_with("sub-reload", 9)
        self.assertEqual(c.processed.subtitle_covered_s, 4.0)

    def test_dynamic_update_replaces_changed_track_path_and_restores_selection(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 0
        c.progressive_sub_loaded = "tcp://first"
        old_track = {"type": "sub", "id": 7, "external": True, "external-filename": str(self.path),
                     "title": self.track["title"], "lang": "chi"}
        new_path = Path(self.tmp.name) / "new.srt"
        new_path.write_text("1\n00:00:01,000 --> 00:00:02,000\n新字幕\n", encoding="utf-8")
        new_track = dict(self.track, path=str(new_path))
        props = {"path": "tcp://first", "track-list": [old_track], "sid": 7}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(new_path),
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [new_track],
        })

        c.apply_subtitle_update(update)

        commands = [call.args for call in c.ipc.command.call_args_list]
        self.assertIn(("sub-remove", 7), commands)
        self.assertIn(("sub-add", str(new_path), "select", self.track["title"], "chi"), commands)

    def test_subtitle_gap_warning_once_until_coverage_catches_up(self):
        c = self.controller()
        c.progressive.po.subtitle_covered_s = 10.0
        c.progressive.po.subtitle_offset = 2.0
        c.progressive.po.subtitle_complete = False
        position = [13.0]
        c.ipc.get.side_effect = lambda name: position[0] if name == "time-pos" else []

        c.subtitle_gap_tick()
        c.subtitle_gap_tick()
        c.osd.assert_called_once_with("此處字幕尚未產生", 4000)
        position[0] = 12.0             # covered_s + offset catches up to the viewer
        c.subtitle_gap_tick()
        position[0] = 13.0
        c.subtitle_gap_tick()
        self.assertEqual(c.osd.call_count, 2)

    def test_island_gap_uses_audio_offset_and_ranges_instead_of_prefix(self):
        c = self.controller()
        po = c.progressive.po
        po.subtitle_covered_s, po.subtitle_offset, po.subtitle_complete = 30.0, 2.0, False
        po.subtitle_covered_ranges = [[0.0, 30.0], [2390.0, 2420.0]]
        position = [2402.0]
        c.ipc.get.side_effect = lambda name: position[0] if name == "time-pos" else []
        c.subtitle_gap_tick()
        c.osd.assert_not_called()
        for value, count in ((2391.0, 1), (2391.0, 1), (2392.0, 1), (2422.0, 1), (2422.1, 2)):
            position[0] = value
            c.subtitle_gap_tick()
            self.assertEqual(c.osd.call_count, count)
        po.subtitle_covered_ranges = []
        position[0] = 12.0
        c.subtitle_gap_warned = False
        c.subtitle_gap_tick()
        self.assertEqual(c.osd.call_count, 3)

    def test_processed_queue_clock_keeps_absolute_and_audio_relative_positions(self):
        c = self.controller()
        c.progressive = None
        c.processed = SimpleNamespace(queue_id="q1", path="output.mkv", subtitle_offset=1.25)
        props = {"path": "output.mkv", "time-pos": 12.75, "pause": False, "eof-reached": False}
        c.ipc.get.side_effect = lambda name, default=None: props.get(name, default)

        c.send_processed_clock(1.0)

        c.link.post.assert_called_once_with({"cmd": "clock", "id": "q1", "pos": 12.75,
                                             "playing": True, "subtitle_pos": 11.5})

    def test_subtitle_update_failures_are_logged_without_raising(self):
        c = self.controller()
        c.progressive.po.subtitle_offset = 1.4
        c.progressive_sub_loaded = "tcp://first"
        missing = str(Path(self.tmp.name) / "gone.srt")
        c.progressive.po.subtitles = [dict(self.track, path=missing)]
        c.load_progressive_subtitles("tcp://first", c.progressive.po, force_reload=True,
                                     previous_tracks=c.progressive.po.subtitles)
        self.assertTrue(c._deps.log.called)

        c._deps.log.reset_mock()
        def get_with_failure(name, default=None):
            if name == "path":
                return "tcp://first"
            if name == "track-list":
                raise OSError("track list unavailable")
            return []
        c.ipc.get.side_effect = get_with_failure
        update = QB.subtitle_update_from_message({
            "cmd": "subtitle_update", "item_id": "q1", "srt": str(self.path),
            "covered_s": 3.0, "complete": False, "subtitle_offset": 0,
            "subtitles": [self.track],
        })

        c.apply_subtitle_update(update)
        self.assertTrue(c._deps.log.called)

    def test_hls_audio_beginning_does_not_seek_before_video_start(self):
        # HLS rejects an audio seek at 8.388 when its first video starts at 8.4.
        # Starting by demuxing preserves the first samples and avoids an empty track.
        audio = SimpleNamespace(start_time=754920, time_base=Fraction(1, 90000),
                                disposition=SimpleNamespace(value=1))
        video = SimpleNamespace(start_time=756000, time_base=Fraction(1, 90000))
        class Streams(list):
            pass
        streams = Streams([video, audio])
        streams.audio = [audio]
        src = Mock(streams=streams)
        src.demux.return_value = iter(())
        av = Mock()
        av.open.return_value = src
        out = Mock()
        out.add_stream_from_template.return_value.codec_context.extradata = b"aac"
        feeder = P.Feeder.__new__(P.Feeder)
        feeder.source, feeder.first_s, feeder.cache_source = "local.m3u8", 8.4, False
        feeder._sides, feeder.log = [], Mock()
        self.assertIs(feeder._add_audio(out, av), src)
        src.seek.assert_not_called()
        self.assertEqual(len(feeder._sides), 1)
        feeder.first_s = 24.4
        self.assertIs(feeder._add_audio(out, av), src)
        src.seek.assert_called_once()


if __name__ == "__main__":
    unittest.main()
