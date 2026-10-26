"""Real subtitle conversion and token-protected HLS delivery, without a TV/GPU."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
from oh_streamer.platform.cast_subtitles import (
    HlsSubtitles,
    HlsSubtitleTracks,
    owner_subtitles,
    seek_line,
    selected_subtitles,
)
from oh_streamer.subtitles.tracks import CONCISE_TITLE, FULL_TITLE
from oh_streamer.platform.hls_cast import HlsServer, SegmentPlan


class CastSubtitleTests(unittest.TestCase):
    def setUp(self):
        self.plan = SegmentPlan(30, 360, 64, 48)

    def test_boundaries_empty_segments_delay_and_timestamp_map(self):
        subs = HlsSubtitles(self.plan, "WEBVTT\n\n00:00:03.000 --> 00:00:05.000\n繁體字幕\n\n"
                            "00:00:07.000 --> 00:00:08.000\nsecond\n", 1)
        self.assertNotIn("繁體字幕", subs.segment(0).decode())
        self.assertIn("00:00:04.000 --> 00:00:06.000\n繁體字幕", subs.segment(1).decode())
        self.assertIn("second", subs.segment(2).decode())
        self.assertIn("MPEGTS:1026000", subs.segment(0).decode())
        self.assertNotIn("seg_", subs.playlist())
        crossing = HlsSubtitles(self.plan, "WEBVTT\n\n00:00:03.000 --> 00:00:05.000\n跨片段\n")
        self.assertIn("跨片段", crossing.segment(0).decode())
        self.assertIn("跨片段", crossing.segment(1).decode())
        self.assertNotIn("跨片段", crossing.segment(2).decode())

    def test_negative_offset_clips_and_drops_expired_cues(self):
        subs = HlsSubtitles(self.plan, "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nexpired\n\n"
                            "00:00:02.000 --> 00:00:04.000\nvisible\n", -3)
        self.assertNotIn("expired", subs.segment(0).decode())
        self.assertIn("00:00:00.000 --> 00:00:01.000\nvisible", subs.segment(0).decode())

    def test_timing_log_lines_carry_every_term_of_the_map(self):
        plan = SegmentPlan(30, 360, 64, 48, video_start=1.467)
        subs = HlsSubtitles(plan, "WEBVTT\n\n00:00:05.000 --> 00:00:06.000\nb\n\n00:00:03.000 --> 00:00:04.000\na\n",
                            -1.467 + 0.5, timing={"delay": 0.5, "origin": 0.0, "external": True, "ff_index": 0,
                                                  "path": "C:/x/track-0.srt"})
        self.assertEqual(subs.cue_count, 2)
        self.assertEqual(subs.first_cue[0], 3.0 - 1.467 + 0.5)
        self.assertEqual(subs.timestamp_clock(), 1026000)
        line = subs.describe(start_k=2)
        for part in ("cue offset -0.967s", "sub-delay +0.500", "video_start 1.467", "external ff-index 0 track-0.srt",
                     "2 cues", "first 00:00:02.033 --> 00:00:03.033", "(source 3.500s)", "MPEGTS=1026000 (11.400s",
                     "first video segment pts 11.400s = source 1.467s", "start segment 2 pts 19.400s = source 9.467s"):
            self.assertIn(part, line)
        long_plan = SegmentPlan(30, 3600, 64, 48, video_start=1.467)
        seek = seek_line(long_plan, 1.467, 31.467, 30.0, subs)
        self.assertIn("playlist 30.000s -> segment 7", seek)
        self.assertIn("video pts 39.400s", seek)
        self.assertIn("cue offset -0.967s", seek)
        self.assertNotIn("cue offset", seek_line(plan, 1.467, 31.467, 30.0, None))

    def test_http_rendition_and_allowlist(self):
        subs = HlsSubtitles(self.plan, "WEBVTT\n\n00:00:03.000 --> 00:00:05.000\n中文\n")
        vod = Mock(plan=self.plan)
        server = HlsServer(vod, "127.0.0.1", subtitles=subs).start()
        self.addCleanup(server.stop)
        def get(name):
            with urllib.request.urlopen(server.url(name)) as response:
                return response.headers, response.read().decode()
        _, master = get("master.m3u8")
        self.assertIn('SUBTITLES="subs"', master)
        self.assertIn('DEFAULT=YES', master)
        self.assertIn('URI="subtitles.m3u8"', master)
        _, playlist = get("subtitles.m3u8")
        self.assertIn("sub_2.vtt", playlist)
        headers, cue = get("sub_1.vtt")
        self.assertIn("text/vtt", headers["Content-Type"])
        self.assertIn("中文", cue)
        self.assertIn("WEBVTT", get("sub_2.vtt")[1])
        for url in (server.url("sub_3.vtt"), server.url("../sub_0.vtt"),
                    server.url("sub_0.vtt").replace(server.token, "wrong")):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url)
            self.assertEqual(error.exception.code, 404)
        vod.get_segment.assert_not_called()

    def test_disabled_or_unselected_subtitles_do_not_convert(self):
        for props in ({"sub-visibility": False}, {"track-list": []},
                      {"track-list": [{"type": "sub", "selected": False}]}):
            self.assertIsNone(selected_subtitles(lambda method, key: props.get(key), self.plan))

    def test_primary_track_wins_over_secondary_track_order(self):
        props = {"track-list": [
            {"type": "sub", "selected": True, "main-selection": 1, "external-filename": "secondary.srt"},
            {"type": "sub", "selected": True, "main-selection": 0, "external-filename": "primary.srt"}]}
        with patch("oh_streamer.platform.cast_subtitles.subprocess.run",
                   return_value=Mock(returncode=0, stdout=b"WEBVTT\n")) as run:
            selected_subtitles(lambda method, key: props.get(key), self.plan)
        self.assertIn("primary.srt", run.call_args.args[0])

    def test_chinese_text_tracks_keep_selected_default_and_exclude_japanese_images(self):
        props = {"path": "video.mkv", "track-list": [
            {"type": "sub", "lang": "chi", "title": "繁體中文（AI・精簡）", "ff-index": 1},
            {"type": "sub", "lang": "zh-Hant", "title": "繁體中文（AI・完整）", "ff-index": 2,
             "selected": True, "main-selection": 0},
            {"type": "sub", "lang": "jpn", "ff-index": 3},
            {"type": "sub", "lang": "chi", "codec": "hdmv_pgs_subtitle", "ff-index": 4}]}
        with patch("oh_streamer.platform.cast_subtitles.subprocess.run",
                   return_value=Mock(returncode=0, stdout=b"WEBVTT\n")) as run:
            subs = selected_subtitles(lambda method, key: props.get(key), self.plan)
        self.assertIsInstance(subs, HlsSubtitleTracks)
        self.assertEqual([t.default for t in subs.renditions], [False, True])
        self.assertEqual([c.args[0][c.args[0].index("-map") + 1] for c in run.call_args_list], ["0:1", "0:2"])
        master = self.plan.master_playlist(subtitles=subs.renditions)
        renditions = [line for line in master.splitlines() if line.startswith("#EXT-X-MEDIA:")]
        self.assertEqual(len(renditions), 2)
        self.assertIn('NAME="繁體中文（AI・完整）",DEFAULT=YES', renditions[1])
        self.assertEqual(master.count("DEFAULT=YES"), 1)

    def test_selected_image_subtitle_retains_conversion_error(self):
        props = {"path": "video.mkv", "track-list": [
            {"type": "sub", "lang": "chi", "ff-index": 1, "codec": "hdmv_pgs_subtitle", "selected": True},
            {"type": "sub", "lang": "chi", "ff-index": 2}]}
        with patch("oh_streamer.platform.cast_subtitles.subprocess.run", return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(ValueError, "字幕無法轉為投放格式"):
                selected_subtitles(lambda method, key: props.get(key), self.plan)

    def test_multitrack_http_routes_and_token_allowlist(self):
        subs = HlsSubtitleTracks([
            HlsSubtitles(self.plan, f"WEBVTT\n\n00:00:03.000 --> 00:00:05.000\n軌{i}\n",
                         title=f"軌{i}", default=i == 0, track_index=i) for i in range(2)])
        vod = Mock(plan=self.plan)
        server = HlsServer(vod, "127.0.0.1", subtitles=subs).start()
        self.addCleanup(server.stop)
        def get(name):
            with urllib.request.urlopen(server.url(name)) as response:
                return response.read().decode()
        master = get("master.m3u8")
        self.assertEqual(master.count("#EXT-X-MEDIA:TYPE=SUBTITLES"), 2)
        self.assertEqual(master.count("DEFAULT=YES"), 1)
        for i in range(2):
            self.assertIn(f'URI="subtitles_{i}.m3u8"', master)
            self.assertIn(f"sub_{i}_1.vtt", get(f"subtitles_{i}.m3u8"))
            body = get(f"sub_{i}_1.vtt")
            self.assertIn(f"軌{i}", body)
            self.assertNotIn(f"軌{1 - i}", body)
            self.assertIn("MPEGTS:1026000", body)
        for name in ("subtitles.m3u8", "sub_0.vtt", "subtitles_2.m3u8", "sub_2_0.vtt",
                     "sub_0_3.vtt", "sub_0_00000000.vtt", "sub_-1_0.vtt", "sub_0_0.vtt/extra",
                     "../sub_0_0.vtt", "%2e%2e/sub_0_0.vtt"):
            with self.assertRaises(urllib.error.HTTPError) as error:
                get(name)
            self.assertEqual(error.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(server.url("sub_0_0.vtt").replace(server.token, "wrong"))
        self.assertEqual(error.exception.code, 404)
        vod.get_segment.assert_not_called()

    def test_metadata_title_cannot_inject_master_playlist_lines(self):
        track = HlsSubtitles(self.plan, "WEBVTT\n", title='中文"\r\n#EXT-X-ENDLIST', track_index=0)
        master = self.plan.master_playlist(subtitles=(track,))
        self.assertNotIn("\n#EXT-X-ENDLIST", master)
        self.assertEqual(master.count("#EXT-X-MEDIA:"), 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_external_tracks_with_same_basename_share_snapshotted_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracks = []
            for i in range(2):
                directory = Path(tmp) / str(i)
                directory.mkdir()
                path = directory / "sub.srt"
                path.write_text(f"1\n00:00:13,000 --> 00:00:15,000\n軌{i}\n", encoding="utf-8")
                tracks.append({"type": "sub", "lang": "chi", "selected": i == 0,
                               "external-filename": str(path)})
            props = {"track-list": tracks, "sub-delay": 1.5, "demuxer-start-time": 10}
            ipc = Mock(side_effect=lambda method, key: props.get(key))
            plan = SegmentPlan(30, 360, 64, 48, video_start=10)
            subs = selected_subtitles(ipc, plan)
            for i, track in enumerate(subs.renditions):
                self.assertIn(f"00:00:04.500 --> 00:00:06.500\n軌{i}", track.segment(1).decode())
                self.assertNotIn(f"軌{1 - i}", track.segment(1).decode())
            self.assertEqual(ipc.call_args_list.count((("get", "sub-delay"), {})), 1)

    def test_manager_uses_master_for_selected_subtitles_even_in_media_mode(self):
        from types import SimpleNamespace
        from test_progressive_cast_controller import manager_type
        from oh_streamer.platform import hls_cast as hls
        vod, server = Mock(), Mock()
        server.start.return_value = server
        fake_hls = SimpleNamespace(FileCastSource=hls.FileCastSource, SegmentPlan=SegmentPlan,
                                   HlsVod=Mock(return_value=vod), HlsServer=Mock(return_value=server))
        manager = object.__new__(manager_type(fake_hls))
        manager.active = None
        manager.args = SimpleNamespace(cast_bind="127.0.0.1", cast_port=0,
                                       cast_cache_gb=1, cast_playlist="media")
        manager._ipc, manager._osd, manager.publish_menu = Mock(), Mock(), Mock()
        session = SimpleNamespace(meta=SimpleNamespace(video_fps_exact=30, frames_count=360,
                                  duration=12, video_width=64, video_height=48,
                                  start_pts=0, time_base=1), fmt_start=0,
                                  has_audio=True, encoder="libx264", target=0, stop=Mock(return_value=0))
        subtitles = HlsSubtitles(self.plan, "WEBVTT\n")
        with patch("oh_streamer.platform.cast_subtitles.selected_subtitles", return_value=subtitles):
            self.assertTrue(manager.start("hls", session, 0))
        self.assertIs(fake_hls.HlsServer.call_args.kwargs["subtitles"], subtitles)
        server.url.assert_called_once_with("master.m3u8")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_real_external_srt_and_ass_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt = Path(tmp) / "字幕 測試.srt"
            srt.write_text("1\n00:00:03,000 --> 00:00:05,000\n你好世界\n", encoding="utf-8")
            ass = srt.with_suffix(".ass")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(srt), str(ass)], check=True)
            for path in (srt, ass):
                props = {"sub-visibility": True, "track-list": [
                    {"type": "sub", "selected": True, "external-filename": str(path)}], "sub-delay": 1}
                subs = selected_subtitles(lambda method, key: props.get(key), self.plan)
                self.assertIn("00:00:04.000 --> 00:00:06.000\n你好世界", subs.segment(1).decode())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_real_embedded_subtitle_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt = Path(tmp) / "sub.srt"
            srt.write_text("1\n00:00:03,000 --> 00:00:05,000\nEmbedded\n", encoding="utf-8")
            for origin in (0, 10):
                video = Path(tmp) / f"video-{origin}.mkv"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:d=6",
                                "-i", str(srt), "-c:v", "libx264", "-c:s", "srt",
                                "-output_ts_offset", str(origin), str(video)], check=True)
                props = {"track-list": [{"type": "sub", "selected": True, "ff-index": 1}],
                         "path": str(video), "demuxer-start-time": origin}
                plan = SegmentPlan(30, 360, 64, 48, video_start=origin)
                subs = selected_subtitles(lambda method, key: props.get(key), plan)
                self.assertIn("00:00:03.000 --> 00:00:05.000\nEmbedded", subs.segment(1).decode())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_real_two_embedded_tracks_nonzero_origin_delay_and_hls_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            concise, full = Path(tmp) / "concise.srt", Path(tmp) / "full.srt"
            concise.write_text("1\n00:00:03,000 --> 00:00:05,000\n共同譯文\n", encoding="utf-8")
            full.write_text(concise.read_text(encoding="utf-8") +
                            "\n2\n00:00:07,000 --> 00:00:08,000\n[低信心] 完整軌\n", encoding="utf-8")
            for origin in (0, 10):
                video = Path(tmp) / f"two-{origin}.mkv"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:d=12",
                                "-i", str(concise), "-i", str(full), "-map", "0:v", "-map", "1:s",
                                "-map", "2:s", "-c:v", "libx264", "-c:s", "srt",
                                "-output_ts_offset", str(origin), str(video)], check=True)
                plan = SegmentPlan(30, 360, 64, 48, video_start=origin)
                props = {"track-list": [
                    {"type": "sub", "lang": "chi", "title": "繁體中文（AI・精簡）", "selected": True, "ff-index": 1},
                    {"type": "sub", "lang": "chi", "title": "繁體中文（AI・完整）", "ff-index": 2}],
                    "path": str(video), "demuxer-start-time": origin, "sub-delay": 1.25}
                subs = selected_subtitles(lambda method, key: props.get(key), plan)
                server = HlsServer(Mock(plan=plan), "127.0.0.1", subtitles=subs).start()
                try:
                    for i in range(2):
                        with urllib.request.urlopen(server.url(f"sub_{i}_1.vtt")) as response:
                            body = response.read().decode()
                        self.assertIn("00:00:04.250 --> 00:00:06.250\n共同譯文", body)
                        self.assertIn("MPEGTS:1026000", body)
                    self.assertNotIn("完整軌", subs.renditions[0].segment(2).decode())
                    self.assertIn("00:00:08.250 --> 00:00:09.250\n[低信心] 完整軌",
                                  subs.renditions[1].segment(2).decode())
                finally:
                    server.stop()


    def test_empty_external_srt_converts_to_empty_webvtt_without_ffmpeg_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_srt = Path(tmp) / "empty.srt"
            empty_srt.write_text("", encoding="utf-8")
            self.assertEqual(empty_srt.stat().st_size, 0)
            props = {"sub-visibility": True, "track-list": [
                {"type": "sub", "selected": True, "external-filename": str(empty_srt)}]}
            subs = selected_subtitles(lambda method, key: props.get(key), self.plan)
            self.assertIsInstance(subs, HlsSubtitles)
            self.assertEqual(subs.cue_count, 0)
            self.assertIn("WEBVTT", subs.segment(0).decode())

            po = SimpleNamespace(subtitles=[{"path": str(empty_srt), "title": "外部字幕", "lang": "chi"}])
            subs_owner = owner_subtitles(po, lambda method, key: props.get(key), self.plan)
            self.assertIsInstance(subs_owner, HlsSubtitles)
            self.assertEqual(subs_owner.cue_count, 0)
            self.assertIn("WEBVTT", subs_owner.segment(0).decode())

    def test_japanese_ai_track_is_included_in_switchable_multitrack_renditions(self):
        track_templates = [
            {"type": "sub", "lang": "chi", "title": "繁體中文（AI・精簡）", "ff-index": 1},
            {"type": "sub", "lang": "chi", "title": "繁體中文（AI・完整）", "ff-index": 2},
            {"type": "sub", "lang": "jpn", "title": "日文（AI）", "ff-index": 3},
        ]
        for sel_idx in range(3):
            tracks = [dict(t) for t in track_templates]
            tracks[sel_idx]["selected"] = True
            tracks[sel_idx]["main-selection"] = 0
            props = {"path": "video.mkv", "track-list": tracks}
            with patch("oh_streamer.platform.cast_subtitles.subprocess.run",
                       return_value=Mock(returncode=0, stdout=b"WEBVTT\n\n00:00:03.000 --> 00:00:05.000\ntext\n")):
                subs = selected_subtitles(lambda method, key: props.get(key), self.plan)
            self.assertIsInstance(subs, HlsSubtitleTracks)
            self.assertEqual(len(subs.renditions), 3)
            self.assertEqual([t.default for t in subs.renditions], [i == sel_idx for i in range(3)])
            self.assertEqual([t.title for t in subs.renditions],
                             ["繁體中文（AI・精簡）", "繁體中文（AI・完整）", "日文（AI）"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_owner_subtitles_stream_placeholder_and_server_live_update_with_no_cache(self):
        po = SimpleNamespace(is_stream=True, subtitles=[], subtitle_offset=1.5)
        subs = owner_subtitles(po, None, self.plan)
        self.assertIsInstance(subs, HlsSubtitleTracks)
        self.assertEqual(len(subs.renditions), 2)
        self.assertEqual(subs.renditions[0].title, CONCISE_TITLE)
        self.assertTrue(subs.renditions[0].default)
        self.assertEqual(subs.renditions[0].cue_count, 0)
        self.assertEqual(subs.renditions[1].title, FULL_TITLE)
        self.assertFalse(subs.renditions[1].default)
        self.assertEqual(subs.renditions[1].cue_count, 0)

        server = HlsServer(Mock(plan=self.plan), "127.0.0.1", subtitles=subs).start()
        self.addCleanup(server.stop)

        def get(name):
            with urllib.request.urlopen(server.url(name)) as response:
                return response.headers, response.read().decode("utf-8")

        headers_m3u8, _ = get("subtitles_0.m3u8")
        self.assertEqual(headers_m3u8.get("Cache-Control"), "no-cache")

        headers_vtt, body_vtt = get("sub_0_1.vtt")
        self.assertEqual(headers_vtt.get("Cache-Control"), "no-cache")
        self.assertIn("WEBVTT", body_vtt)
        self.assertNotIn("-->", body_vtt)

        with tempfile.TemporaryDirectory() as tmp:
            concise_path = Path(tmp) / "concise.srt"
            full_path = Path(tmp) / "full.srt"
            concise_path.write_text("1\n00:00:03,000 --> 00:00:05,000\n精簡即時字幕\n", encoding="utf-8")
            full_path.write_text("1\n00:00:03,000 --> 00:00:05,000\n完整即時字幕\n", encoding="utf-8")

            po.subtitles = [
                {"path": str(concise_path), "title": CONCISE_TITLE, "lang": "chi"},
                {"path": str(full_path), "title": FULL_TITLE, "lang": "chi"},
            ]
            server.update_subtitles(owner_subtitles(po, None, self.plan))

            _, updated_sub_0_1 = get("sub_0_1.vtt")
            _, updated_sub_1_1 = get("sub_1_1.vtt")

            self.assertIn("00:00:04.500 --> 00:00:06.500", updated_sub_0_1)
            self.assertIn("精簡即時字幕", updated_sub_0_1)
            self.assertNotIn("完整即時字幕", updated_sub_0_1)

            self.assertIn("00:00:04.500 --> 00:00:06.500", updated_sub_1_1)
            self.assertIn("完整即時字幕", updated_sub_1_1)
            self.assertNotIn("精簡即時字幕", updated_sub_1_1)


if __name__ == "__main__":
    unittest.main()
