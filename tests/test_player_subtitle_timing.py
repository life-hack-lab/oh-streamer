"""Subtitle timing diagnostics for local playback (sample video A): pure line builders and the controller hooks."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.player import subtitle_timing as T
from oh_streamer.player.progressive_stream import PlayInfo


def fake_controller(props, po, play=None):
    logs = []
    ipc = SimpleNamespace(get=lambda key, default=None: props.get(key, default))
    watcher = SimpleNamespace(state=SimpleNamespace(play=play))
    progressive = SimpleNamespace(po=po, watcher=watcher)
    return SimpleNamespace(ipc=ipc, _deps=SimpleNamespace(log=logs.append), progressive=progressive,
                           progressive_sub_dir="C:/tmp/subs"), logs


class SubtitleTimingTests(unittest.TestCase):
    def setUp(self):
        self.po = SimpleNamespace(queue_id="abc12345", subtitle_offset=1.4, subtitle_covered_s=300.0,
                                  info={"video_start": 1.467},
                                  subtitles=[{"path": "C:/q/zh.srt", "title": "中文", "lang": "chi", "default": True}])
        self.play = PlayInfo(30, 120, video_start=1.467)

    def test_load_line_lists_every_term(self):
        props = {"sub-delay": 0.25, "demuxer-start-time": 1.467, "time-pos": 12.5, "sid": 2,
                 "track-list": [{"type": "sub", "id": 2, "title": "中文"}]}
        ctl, logs = fake_controller(props, self.po, self.play)
        T.log_load(ctl, "tcp://x", self.po, True)
        self.assertEqual(len(logs), 1)
        line = logs[0]
        for part in ("reload (progressive)", "item=abc12345", "track-0.srt<-zh.srt", "selected=#2:中文",
                     "subtitle_offset(audio_offset)=+1.400s", "sub-delay=+0.250s", "stream_start=1.467s",
                     "source_video_start=1.467s", "demuxer-start-time=1.467s", "time-pos=12.500s", "covered_s=300.000"):
            self.assertIn(part, line)

    def test_missing_values_never_raise(self):
        ctl, logs = fake_controller({}, SimpleNamespace(subtitles=[], subtitle_offset=None))
        T.log_load(ctl, "u", ctl.progressive.po, False)
        self.assertIn("load (progressive)", logs[0])
        self.assertIn("sub-delay=?s", logs[0])
        broken, logs = fake_controller({}, SimpleNamespace(subtitles=[{}]))
        T.log_load(broken, "u", broken.progressive.po, False)
        self.assertIn("load line skipped", logs[0])

    def test_seek_line_maps_time_pos_to_srt_time(self):
        ctl, logs = fake_controller({"sub-delay": 0.5}, self.po, self.play)
        T.log_seek(ctl, 3757.19)
        line = logs[0]
        self.assertIn("seek time-pos 3757.190s", line)
        self.assertIn("segment 938 starting", line)
        self.assertIn(f"srt_time={3757.19 - 1.4 - 0.5:.3f}s", line)


if __name__ == "__main__":
    unittest.main()
