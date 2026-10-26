"""Queue backend of the intake panel (解析面板): parse_inputs / resolve / defaults / estimate /
stream_resolved / add_resolved / save_defaults and the pinned per-item overrides. Fake resolver + thumbnail grabber,
temp profile / player settings, no GPU, no network. Run from streamer/: python -m unittest tests.test_queue_intake."""
import json
import sys
import threading
import types
from pathlib import Path
from unittest import mock

try:                                    # run_tests.py imports tests.* ; "-s tests" imports flat names
    from tests.test_queue_service import ServiceTestBase, wait_for
    from tests.test_perf_profile import fixture, two_detectors
except ImportError:
    from test_queue_service import ServiceTestBase, wait_for
    from test_perf_profile import fixture, two_detectors

from oh_streamer.perf import profile as P
from oh_streamer.player import player_settings as PS
from oh_streamer.queue import queue_app, queue_store
from oh_streamer.queue import queue_service as QS
from oh_streamer.queue import stream_audio
from oh_streamer.sources import media_input, source_info

MODELS = {"detection_model": [{"value": "v4-fast", "label": "v4-fast（建議）"},
                              {"value": "v4-accurate", "label": "v4-accurate"}],
          "restoration_model": [{"value": "basicvsrpp-v1.2", "label": "basicvsrpp-v1.2（建議）"},
                                {"value": "deepmosaics", "label": "deepmosaics"}]}


def intake_profile():
    """720p is fast (live at 30 and 60 fps), 1080p is heavy, 480p is the lightest."""
    p = fixture()
    p["restore"] = {"480": {"d0_fps": 300.0, "d1_fps": 120.0}, "720": {"d0_fps": 200.0, "d1_fps": 80.0},
                    "1080": {"d0_fps": 60.0, "d1_fps": 25.0}}
    p["rife"] = {"480": {"pair_ms": 5.0}, "720": {"pair_ms": 9.5}, "1080": {"pair_ms": 20.0}}
    p["encode"] = {"hevc_nvenc": {"480": {"fps": 2000.0}, "720": {"fps": 1000.0}, "1080": {"fps": 450.0}}}
    p["combined"] = {"height": 720, "density": 1.0, "encoder": "hevc_nvenc", "rife": "full", "fps": 30.0}
    return p


def resolved(target="https://x.test/v", heights=((1080, 30.0), (720, 30.0)), duration=600.0, kind="url", quals=True,
             headers=None, title="A video", default_height=None, fps=None):
    hs = [{"height": h, "fps": f, "label": f"{h}p", "size_bytes": None} for h, f in heights]
    ql = [{"id": str(h), "height": h, "fps": f, "url": f"{target}/{h}", "http_headers": {}} for h, f in heights] \
        if quals else None
    dh = default_height or (heights[0][0] if heights else None)
    return source_info.Resolved(target, kind, dict(headers or {}), title, "x.test", duration=duration,
                                fps=fps or (heights[0][1] if heights else None), default_height=dh, vcodec="h264",
                                acodec="aac", heights=hs, quals=ql,
                                fps_by_height={h: f for h, f in heights if f})


class IntakeBase(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.resolver_calls, self.thumb_calls = [], []
        self.resolver_result = resolved()
        self.thumb_result = "data:image/jpeg;base64,AAAA"
        self.svc._source_resolver = self._resolve
        self.svc._thumb_grabber = self._thumb
        self.profile_path = self.tmp / "perf" / "perf_profile.json"
        self.player_file = self.tmp / "player" / "oh_settings.json"
        perf = self.svc.perf
        perf._profile_path = self.profile_path
        perf._player_settings_path = self.player_file
        perf._fingerprint = lambda: {"gpu": "RTX 5060"}
        real = QS.platform_support.capabilities()
        patcher = mock.patch.object(QS.platform_support, "capabilities", lambda: dict(real, rife=True))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(QS.QueueService, "_model_options", staticmethod(lambda: json.loads(json.dumps(MODELS))))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.svc.pause_all()                      # items added by the tests are never started

    def _resolve(self, oi):
        self.resolver_calls.append(oi)
        if isinstance(self.resolver_result, Exception):
            raise self.resolver_result
        return self.resolver_result

    def _thumb(self, r):
        self.thumb_calls.append(r)
        if isinstance(self.thumb_result, Exception):
            raise self.thumb_result
        return self.thumb_result

    def save_profile(self, prof=None):
        P.save(prof or intake_profile(), self.profile_path)

    def write_player(self, **settings):
        self.player_file.parent.mkdir(parents=True, exist_ok=True)
        self.player_file.write_text(json.dumps(settings), encoding="utf-8")

    def token(self, r=None):
        return self.svc._tokens.put(r or self.resolver_result)

    def player_settings(self):
        return json.loads(self.player_file.read_text(encoding="utf-8"))


class ParseResolveTests(IntakeBase):
    def test_parse_inputs_list_curl_and_header_forms(self):
        r = self.svc.parse_inputs("https://a.test/1\n\nC:\\videos\\b.mp4\nhttps://a.test/1\n")
        self.assertTrue(r["ok"])
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["entries"], [
            {"text": "https://a.test/1", "kind": "url", "target": "https://a.test/1", "headers_count": 0},
            {"text": "C:\\videos\\b.mp4", "kind": "file", "target": "C:\\videos\\b.mp4", "headers_count": 0}])
        curl = "curl 'https://a.test/2' -H 'Referer: https://a.test/' -H 'Cookie: s=1'"
        r = self.svc.parse_inputs(curl)
        self.assertEqual(r["entries"], [{"text": curl, "kind": "url", "target": "https://a.test/2", "headers_count": 2}])
        text = "https://a.test/3\nReferer: https://a.test/"
        r = self.svc.parse_inputs(text)
        self.assertEqual([(e["text"], e["headers_count"]) for e in r["entries"]], [(text, 1)])
        # an entry's text round-trips through resolve()
        self.svc.resolve(r["entries"][0]["text"])
        self.assertEqual(self.resolver_calls[0].headers, {"Referer": "https://a.test/"})

    def test_parse_inputs_errors(self):
        self.assertFalse(self.svc.parse_inputs("")["ok"])
        self.assertFalse(self.svc.parse_inputs(None)["ok"])
        self.assertIn("error", self.svc.parse_inputs("Referer: x\nCookie: y"))
        r = self.svc.parse_inputs("https://a.test/1\nhttps://a.test/with space")
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["entries"]), 1)
        self.assertEqual(len(r["errors"]), 1)
        self.assertFalse(self.svc.parse_inputs("https://a.test/with space\nhttps://a.test/also bad")["ok"])

    def test_resolve_info_and_token(self):
        self.resolver_result = resolved("https://x.test/v", headers={"Cookie": "secret"})
        r = self.svc.resolve("curl 'https://x.test/v' -H 'Cookie: secret'")
        self.assertTrue(r["ok"])
        info = r["info"]
        self.assertEqual(self.resolver_calls[0].target, "https://x.test/v")
        self.assertEqual({k for k in info}, {"token", "kind", "target", "title", "site", "duration", "fps", "heights",
                                              "default_height", "vcodec", "acodec", "hls", "headers_count", "thumb",
                                              "dupe", "subtitle_audio"})
        self.assertEqual(info["subtitle_audio"], "lowest")
        self.assertEqual((info["kind"], info["target"], info["title"], info["site"], info["duration"], info["fps"]),
                         ("url", "https://x.test/v", "A video", "x.test", 600.0, 30.0))
        self.assertEqual(info["heights"][0], {"height": 1080, "fps": 30.0, "label": "1080p", "size_bytes": None})
        self.assertEqual((info["default_height"], info["vcodec"], info["acodec"], info["hls"], info["headers_count"],
                          info["thumb"], info["dupe"]), (1080, "h264", "aac", False, 1, self.thumb_result, None))
        self.assertNotIn("secret", json.dumps(r))                      # headers stay server side
        self.assertIs(self.svc._tokens.get(info["token"]), self.resolver_result)
        self.assertEqual(len({self.svc.resolve("https://x.test/v")["info"]["token"] for _ in range(3)}), 3)

    def test_resolve_errors(self):
        self.resolver_result = source_info.ResolveError("不支援這個網站或網址", "ERROR: Unsupported URL")
        self.assertEqual(self.svc.resolve("https://x.test/v"),
                         {"ok": False, "error": "不支援這個網站或網址", "detail": "ERROR: Unsupported URL"})
        self.resolver_result = RuntimeError("first line\nlast line")
        r = self.svc.resolve("https://x.test/v")
        self.assertEqual((r["ok"], r["detail"]), (False, "last line"))
        r = self.svc.resolve("")
        self.assertFalse(r["ok"])
        self.assertTrue(r["error"] and r["detail"])
        json.dumps(r)

    def test_thumbnail_failure_is_not_an_error(self):
        self.thumb_result = RuntimeError("ffmpeg died")
        r = self.svc.resolve("https://x.test/v")
        self.assertTrue(r["ok"])
        self.assertIsNone(r["info"]["thumb"])

    def test_default_thumb_grabber_uses_the_thumb_source(self):
        svc = QS.QueueService(self.tmp / "root2", player_name=self.name, launch_player=lambda *a: None,
                              send_to_player=lambda p: None)
        r = resolved()
        r.thumb_src = ("https://x.test/low.mp4", {"Referer": "r"})
        with mock.patch.object(source_info, "grab_thumbnail", return_value="data:x") as grab:
            self.assertEqual(svc._thumb_grabber(r), "data:x")
        grab.assert_called_once_with("https://x.test/low.mp4", {"Referer": "r"}, 600.0)
        r.thumb_src = None
        self.assertIsNone(svc._thumb_grabber(r))

    def test_dupe_of_existing_item(self):
        a = self.svc.store.add("https://x.test/v", "url", {}, 720, {}, title="t")
        self.svc.store.update(a.id, state="processing", progress={"percent": 42.0})
        d = self.svc.resolve("https://x.test/v")["info"]["dupe"]
        self.assertEqual(d, {"id": a.id, "state": "processing", "progress": 0.42})   # a 0..1 fraction
        b = self.svc.store.add("https://x.test/v", "url", {}, 480, {}, title="t")      # same target, other quality
        self.svc.store.update(b.id, state="done", output_path=str(self.tmp / "missing.mkv"))
        self.assertEqual(self.svc.resolve("https://x.test/v")["info"]["dupe"]["id"], a.id)   # done without a file: skipped
        out = self.tmp / "out.mkv"
        out.write_bytes(b"x")
        self.svc.store.update(b.id, output_path=str(out))
        self.assertEqual(self.svc.resolve("https://x.test/v")["info"]["dupe"],
                         {"id": b.id, "state": "done", "progress": 1.0})
        self.resolver_result = resolved("https://x.test/other")
        self.assertIsNone(self.svc.resolve("https://x.test/other")["info"]["dupe"])


class DefaultsTests(IntakeBase):
    def test_effective_values_and_choices(self):
        self.write_player(detection_model="v4-accurate", restoration_model="deepmosaics", interpolation="auto",
                          restoration_fps="30", vsr="off")
        d = self.svc.defaults()
        self.assertEqual(d["stream"], {"detection_model": "v4-accurate", "restoration_model": "deepmosaics",
                                       "interpolation": "auto", "restoration_fps": "30"})
        self.assertEqual(d["queue"], {"detection_model": "v4-accurate", "restoration_model": "deepmosaics",
                                  "interpolation": "auto", "encoder": "hevc", "save_compare": False,
                                  "subtitles": False})   # inherits
        self.assertEqual(d["choices"], MODELS)
        self.assertFalse(d["has_profile"])
        json.dumps(d)
        self.svc.store.set_settings(detection_model="v4-fast", interpolation="off", encoder="h264", save_compare=True)
        q = self.svc.defaults()["queue"]
        self.assertEqual((q["detection_model"], q["restoration_model"], q["interpolation"], q["encoder"], q["save_compare"]),
                         ("v4-fast", "deepmosaics", "off", "h264", True))
        self.assertEqual(self.svc.defaults()["stream"]["detection_model"], "v4-fast")   # one setting for stream and queue
        self.save_profile()
        self.assertTrue(self.svc.defaults()["has_profile"])

    def test_settings_dialog_models_and_clip_length_are_mirrored_to_the_player(self):
        self.assertTrue(self.svc.set_settings({"detection_model": "v4-accurate", "max_clip_length": 120})["ok"])
        p = self.svc._player_cfg()
        self.assertEqual((p["detection_model"], p["max_clip_length"]), ("v4-accurate", 120))
        self.assertEqual(self.svc._effective_queue_settings()["max_clip_length"], 120)
        self.assertFalse(self.svc.set_settings({"max_clip_length": 0})["ok"])

    def test_model_labels_name_the_version(self):
        from oh_streamer.player import player_settings as PS
        self.assertEqual((PS.model_label("v4-fast"), PS.model_label("v4-accurate"), PS.model_label("v2")),
                         ("V4 快速", "V4 精準", "V2"))
        self.assertEqual(PS.model_label("basicvsrpp-v1.2"), "basicvsrpp-v1.2")

    def test_player_defaults_without_a_settings_file(self):
        d = self.svc.defaults()
        self.assertEqual((d["stream"]["detection_model"], d["stream"]["interpolation"], d["stream"]["restoration_fps"]),
                         ("v4-fast", "off", "original"))
        self.assertEqual(d["queue"]["interpolation"], "auto")         # queue default is "full"


class EstimateTests(IntakeBase):
    def setUp(self):
        super().setUp()
        self.prof = intake_profile()

    def view(self, detector=None):
        return P.for_detector(self.prof, detector)

    def test_without_a_profile(self):
        t = self.token()
        e = self.svc.estimate([t], {})
        self.assertEqual(e, {"has_profile": False, "queue_position": 1, "queue_ahead_s": None,
                             "items": [{"token": t, "stream": None, "stream_by_height": {}, "queue": None}]})
        self.svc.store.add("https://x.test/other", "url", {}, None, {})
        self.assertEqual(self.svc.estimate([t])["queue_position"], 2)

    def test_30fps_stream_and_queue(self):
        self.save_profile()
        t = self.token(resolved(heights=((1080, 30.0), (720, 30.0), (480, 30.0))))
        e = self.svc.estimate([t], {})
        self.assertTrue(e["has_profile"])
        item = e["items"][0]
        self.assertEqual(item["token"], t)
        self.assertEqual(item["stream_by_height"], {"1080": "prebuffer", "720": "live", "480": "live"})
        s = item["stream"]                                        # default rendition: 1080p, saved player settings
        v = P.verdict(self.view("v4-fast"), height=1080, src_fps=30.0, mode="off")
        self.assertEqual((s["verdict"], s["detector"], s["need_fps"]), ("prebuffer", "v4-fast", 30.0))
        self.assertAlmostEqual(s["ratio"], v["ratio_typical"], places=3)
        self.assertAlmostEqual(s["restore_fps"], v["ratio_typical"] * 30.0, places=1)
        self.assertAlmostEqual(s["prebuffer_s"], v["prebuffer_s"])
        self.assertIsNone(s["interp_share"])                      # interpolation off
        self.assertEqual(s["suggest"], {"height": 720})            # highest lower height that is live
        q = item["queue"]                                         # queue default: interpolation full, best height
        self.assertAlmostEqual(q["eta_s"], P.job_seconds(self.view("v4-fast"), duration_s=600.0, height=1080,
                                                         src_fps=30.0, mode="full"))
        self.assertAlmostEqual(q["fps"], P.estimate(self.view("v4-fast"), height=1080, src_fps=30.0, mode="full")["fps"],
                               places=1)
        json.dumps(e)

    def test_overrides_change_height_and_interpolation(self):
        self.save_profile()
        t = self.token(resolved(heights=((1080, 30.0), (720, 30.0))))
        e = self.svc.estimate([t], {"height": 720, "interpolation": "auto"})["items"][0]
        self.assertEqual(e["stream"]["verdict"], "live")
        self.assertIsNone(e["stream"]["suggest"])
        self.assertTrue(0 < e["stream"]["interp_share"] <= 1)     # auto interpolation at 30 fps: some share
        self.assertAlmostEqual(e["queue"]["eta_s"], P.job_seconds(self.view("v4-fast"), duration_s=600.0, height=720,
                                                                  src_fps=30.0, mode="full"))
        off = self.svc.estimate([t], {"height": 720, "interpolation": "off"})["items"][0]["queue"]
        self.assertAlmostEqual(off["eta_s"], P.job_seconds(self.view("v4-fast"), duration_s=600.0, height=720,
                                                           src_fps=30.0, mode="off"))
        self.assertLess(off["eta_s"], e["queue"]["eta_s"])
        no_dur = self.token(resolved(duration=None))
        self.assertIsNone(self.svc.estimate([no_dur])["items"][0]["queue"])

    def test_60fps_restoration_fps_and_suggestion(self):
        self.save_profile()
        t = self.token(resolved(heights=((1080, 60.0), (720, 60.0))))
        orig = self.svc.estimate([t], {})["items"][0]["stream"]
        self.assertEqual((orig["verdict"], orig["need_fps"]), ("queue", 60.0))
        # 720p keeps up with 60 fps, 1080p does not (even at 30 fps it is only a prebuffer): suggest the lower height
        self.assertEqual(orig["suggest"], {"height": 720})
        self.assertEqual(self.svc.estimate([t], {})["items"][0]["stream_by_height"], {"1080": "queue", "720": "live"})
        # restoration_fps 30 on the 1080p rendition: restored at 30 fps, interpolation forced to auto
        e = self.svc.estimate([t], {"restoration_fps": "30", "interpolation": "off"})["items"][0]["stream"]
        self.assertEqual(e["need_fps"], 30.0)
        v = P.verdict(self.view("v4-fast"), height=1080, src_fps=30.0, mode="auto")
        self.assertEqual(e["verdict"], v["level"])
        self.assertAlmostEqual(e["interp_share"], v["interp_fraction"], places=3)
        self.assertEqual(e["suggest"], {"height": 720})
        # queue ignores restoration_fps: original 60 fps
        q = self.svc.estimate([t], {"restoration_fps": "30"})["items"][0]["queue"]
        self.assertAlmostEqual(q["eta_s"], P.job_seconds(self.view("v4-fast"), duration_s=600.0, height=1080,
                                                         src_fps=60.0, mode="full"))

    def test_suggest_restoration_fps_when_it_alone_makes_it_live(self):
        prof = intake_profile()
        prof["restore"]["1080"] = {"d0_fps": 100.0, "d1_fps": 40.0}      # live at 30 fps, queue at 60 fps
        self.save_profile(prof)
        t = self.token(resolved(heights=((1080, 60.0),)))
        s = self.svc.estimate([t], {})["items"][0]["stream"]
        self.assertEqual((s["verdict"], s["suggest"]), ("queue", {"restoration_fps": "30"}))
        done = self.svc.estimate([t], {"restoration_fps": "30"})["items"][0]["stream"]
        self.assertEqual((done["verdict"], done["suggest"]), ("live", None))
        self.write_player(restoration_fps="30")                          # saved setting counts as well
        self.assertEqual(self.svc.estimate([t], {})["items"][0]["stream"]["verdict"], "live")

    def test_stream_and_queue_use_their_own_detectors(self):
        self.save_profile(two_detectors())
        self.write_player(detection_model="v4-fast")
        self.svc.store.set_settings(detection_model="v4-accurate")
        t = self.token(resolved(heights=((480, 30.0),)))
        e = self.svc.estimate([t], {})["items"][0]
        prof = two_detectors()
        self.assertEqual(e["stream"]["detector"], "v4-fast")
        self.assertAlmostEqual(e["stream"]["ratio"], P.verdict(P.for_detector(prof, "v4-fast"), height=480, src_fps=30.0,
                                                               mode="off")["ratio_typical"], places=3)
        self.assertAlmostEqual(e["queue"]["eta_s"], P.job_seconds(P.for_detector(prof, "v4-accurate"), duration_s=600.0,
                                                                  height=480, src_fps=30.0, mode="full"))
        picked = self.svc.estimate([t], {"detection_model": "v4-accurate"})["items"][0]       # picking one unifies
        self.assertEqual(picked["stream"]["detector"], "v4-accurate")
        self.assertLess(picked["stream"]["ratio"], e["stream"]["ratio"])

    def test_batch_has_no_stream_numbers(self):
        self.save_profile()
        a, b = self.token(resolved("https://x.test/a")), self.token(resolved("https://x.test/b"))
        e = self.svc.estimate([a, b], {})
        self.assertEqual([i["stream"] for i in e["items"]], [None, None])
        self.assertTrue(all(i["queue"] and i["stream_by_height"] for i in e["items"]))

    def test_expired_token(self):
        self.save_profile()
        e = self.svc.estimate(["gone"], {})
        self.assertEqual(e["items"][0]["stream"], None)
        self.assertIn("error", e["items"][0])

    def test_queue_position_and_wait_ahead(self):
        self.save_profile()
        facts = {"height": 720, "fps": 30.0, "duration": 600.0}
        ids = []
        for state in ("processing", "queued", "queued", "paused", "failed", "done"):
            it = self.svc.store.add(f"https://x.test/{len(ids)}", "url", {}, None, {})
            src = self.tmp / f"s{len(ids)}.mp4"
            self.svc.store.update(it.id, state=state, source_path=str(src), source_status="complete",
                                  progress={"frame": 100, "frames": 18000, "eta": 100.0} if state == "processing" else {})
            self.svc._eta_facts[it.id] = ((str(src), "complete"), dict(facts), QS.time.monotonic())
            ids.append(it.id)
        t = self.token()
        e = self.svc.estimate([t], {})
        self.assertEqual(e["queue_position"], 4)                       # processing + 2 queued ahead
        prior = P.job_seconds(self.view("v4-fast"), duration_s=600.0, height=720, src_fps=30.0, mode="full")
        self.assertAlmostEqual(e["queue_ahead_s"], 100.0 + 2 * prior, places=0)


class StreamSubtitleGpuEstimateTests(IntakeBase):
    """T4.3: stream.subtitle_gpu = the stream verdict with local subtitles counted in (queue/stream_subtitle_gpu.py)."""

    def setUp(self):
        super().setUp()
        from oh_streamer.queue import subtitle_local as sl
        from oh_streamer.queue import stream_subtitle_gpu as ssg
        self.sl, self.ssg = sl, ssg
        self.gpu = sl.GpuInfo("Test GPU", 8192.0)
        self.svc._subtitle_gpu_info = lambda: self.gpu
        patcher = mock.patch.dict("os.environ", {"LOCALAPPDATA": str(self.tmp / "localappdata")})   # never the real memory
        patcher.start()
        self.addCleanup(patcher.stop)
        self.save_profile()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")      # a fresh store's cloud profile is unconfigured

    def estimate(self, ov, heights=((1080, 30.0), (720, 30.0))):
        t = self.token(resolved(heights=heights))
        return self.svc.estimate([t], ov)["items"][0]["stream"]

    def cloud_snapshot(self, ready=True):
        cloud = {"id": "cloud-asr", "provider": "openai", "base_url": "https://api.example.test/v1"}
        return dict(self.svc.store.get_settings(), subtitle_runtime_contract="cloud-first-1",
                    subtitle_configs=[{"id": "primary"}],
                    subtitle_resolved={"models": {"asr": cloud, "translation": dict(cloud, id="cloud-mt")}, "ready": True,
                                       "fallback": {"asr": [{"id": "whisper-ja-1.5B", "provider": "local"}],
                                                    "translation": []}})

    def test_local_subtitles_slow_the_verdict_and_suggest_queue(self):
        plain = self.estimate({"subtitles": False})
        self.assertNotIn("subtitle_gpu", plain)
        self.assertEqual(plain["verdict"], "prebuffer")
        g = self.estimate({"subtitles": True})["subtitle_gpu"]
        self.assertEqual(g["verdict_without"], "prebuffer")
        self.assertEqual(g["verdict"], "queue")
        self.assertLess(g["ratio"], plain["ratio"])
        self.assertFalse(g["learned"])
        self.assertTrue(g["vram_ok"])
        self.assertEqual(g["s"], self.ssg.subtitle_seconds_per_media_second("", None))
        self.assertEqual(g["suggest"][0], {"kind": "queue"})
        self.assertNotIn("prebuffer", [x["kind"] for x in g["suggest"]])
        json.dumps(g)

    def test_fast_source_stays_live_without_suggestions(self):
        self.ssg.SubtitleSpeedMemory().record(
            self.sl.gt.gpu_key("Test GPU", 8192.0), 30.0, 3000.0)          # a very fast card: s = 0.01
        g = self.estimate({"subtitles": True, "height": 720}, heights=((720, 30.0),))["subtitle_gpu"]
        self.assertEqual((g["verdict"], g["verdict_without"], g["learned"]), ("live", "live", True))
        self.assertEqual(g["suggest"], [])

    def test_learned_speed_is_used(self):
        key = self.sl.gt.gpu_key("Test GPU", 8192.0)
        self.ssg.SubtitleSpeedMemory().record(key, 600.0, 1000.0)          # s = 0.6
        g = self.estimate({"subtitles": True})["subtitle_gpu"]
        self.assertTrue(g["learned"])
        self.assertAlmostEqual(g["s"], 0.6, places=3)

    def test_learned_switch_costs_are_used_by_the_verdict(self):
        g = self.estimate({"subtitles": True, "height": 720}, heights=((720, 30.0),))["subtitle_gpu"]
        self.assertEqual((g["load_s"], g["restart_s"], g["costs_learned"]), (20.0, 10.0, False))   # stream defaults
        key = self.sl.gt.gpu_key("Test GPU", 8192.0)
        mem = self.ssg.SwitchCostMemory()
        mem.record(key, self.ssg.COST_ASR_LOAD, 12.0)
        mem.record(key, self.ssg.COST_MT_LOAD, 5.0)
        mem.record(key, self.ssg.COST_RESTART, 2.0)
        g2 = self.estimate({"subtitles": True, "height": 720}, heights=((720, 30.0),))["subtitle_gpu"]
        self.assertEqual((g2["load_s"], g2["restart_s"], g2["costs_learned"]), (17.0, 2.0, True))
        self.assertGreater(g2["ratio"], g["ratio"])
        json.dumps(g2)

    def test_small_card_cannot_hold_restoration_and_asr(self):
        self.gpu = self.sl.GpuInfo("Small GPU", 4096.0)
        g = self.estimate({"subtitles": True, "height": 720}, heights=((720, 30.0),))["subtitle_gpu"]
        self.assertFalse(g["vram_ok"])
        self.assertGreater(g["vram_need_mib"], 4096)
        self.assertEqual(g["suggest"][0], {"kind": "queue"})
        self.assertNotIn("prebuffer", [x["kind"] for x in g["suggest"]])

    def test_learned_paused_restore_vram_replaces_the_fixed_figure(self):
        g = self.estimate({"subtitles": True, "height": 720}, heights=((720, 30.0),))["subtitle_gpu"]
        self.assertEqual((g["restore_mib"], g["restore_mib_learned"], g["vram_ok"]), (1232, False, True))
        self.assertEqual(g["vram_need_mib"], 1232 + 4801 + 1536)
        key = self.sl.gt.gpu_key("Test GPU", 8192.0)
        self.ssg.RestoreVramMemory().record(key, 2100.0)                   # the player reported a bigger paused footprint
        g = self.estimate({"subtitles": True, "height": 720}, heights=((720, 30.0),))["subtitle_gpu"]
        self.assertEqual((g["restore_mib"], g["restore_mib_learned"]), (2100, True))
        self.assertEqual(g["vram_need_mib"], 2100 + 4801 + 1536)
        self.assertFalse(g["vram_ok"])                                       # 8438 > 8192
        self.assertEqual(g["suggest"][0], {"kind": "queue"})

    def test_prebuffer_level_suggests_prebuffer_seconds(self):
        base = {"level": "live", "ratio_typical": 1.6, "ratio_heavy": 1.0, "prebuffer_s": None}
        with mock.patch.object(self.ssg, "verdict_with_subtitles",
                               return_value={"level": "prebuffer", "ratio_typical": 1.2, "ratio_heavy": 0.9,
                                             "prebuffer_s": 120.0}):
            g = self.svc._stream_subtitle_gpu(base, {"subtitles": True})
        self.assertEqual([x["kind"] for x in g["suggest"]], ["queue", "prebuffer"])
        self.assertEqual(g["suggest"][1]["seconds"], 120.0)

    def test_cloud_subtitles_add_no_key(self):
        with mock.patch.object(self.svc, "_subtitle_snapshot", return_value=self.cloud_snapshot()):
            self.assertNotIn("subtitle_gpu", self.estimate({"subtitles": True}))

    def test_cloud_suggestion_only_when_cloud_is_usable(self):
        with mock.patch.object(QS.QueueService, "_cloud_subtitles_usable", return_value=True):
            kinds = [x["kind"] for x in self.estimate({"subtitles": True})["subtitle_gpu"]["suggest"]]
        self.assertEqual(kinds, ["queue", "cloud"])
        with mock.patch.object(QS.QueueService, "_cloud_subtitles_usable", return_value=False):
            kinds = [x["kind"] for x in self.estimate({"subtitles": True})["subtitle_gpu"]["suggest"]]
        self.assertEqual(kinds, ["queue"])

    def test_cloud_usable_needs_consent_and_a_configured_cloud_profile(self):
        snap = self.svc._subtitle_snapshot({})
        self.assertFalse(QS.QueueService._cloud_subtitles_usable(snap))      # default settings: nothing configured
        self.assertFalse(QS.QueueService._cloud_subtitles_usable({}))

    def test_never_breaks_the_estimate(self):
        with mock.patch.object(self.svc, "_subtitle_snapshot", side_effect=ValueError("not ready")):
            s = self.estimate({"subtitles": True})
        self.assertNotIn("subtitle_gpu", s)
        self.assertEqual(s["verdict"], "prebuffer")
        self.svc._subtitle_gpu_info = mock.Mock(side_effect=RuntimeError("no nvidia-smi"))
        s = self.estimate({"subtitles": True})
        self.assertNotIn("subtitle_gpu", s)
        self.assertIn("restore_fps", s)


class StreamResolvedTests(IntakeBase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(stream_audio, "probe_audio_start", return_value=0.0)    # never reach the network
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_payload_quality_and_overrides(self):
        self.player_reply = {"ok": True}
        self.resolver_result = resolved("https://x.test/v", heights=((1080, 60.0), (720, 60.0)),
                                        headers={"Referer": "https://x.test/"})
        t = self.token()
        r = self.svc.stream_resolved(t, {"height": 720, "detection_model": "v4-accurate", "interpolation": "off",
                                         "restoration_fps": "30", "encoder": "h264", "save_compare": True})
        self.assertEqual((r["ok"], r["launched"]), (True, False))
        self.assertEqual(self.players, [{
            "cmd": "open", "input": "https://x.test/v", "headers": {"Referer": "https://x.test/"}, "quality": 720,
            "overrides": {"detection_model": "v4-accurate", "interpolation": "auto", "restoration_fps": "30"}}])
        self.assertTrue(self.svc.is_player_busy())                    # the opening guard, like stream()

    def test_no_height_no_overrides(self):
        self.player_reply = {"ok": True}
        t = self.token(resolved(heights=((1080, 30.0), (720, 30.0))))
        self.svc.stream_resolved(t, {"height": 360})                  # not offered
        self.svc.stream_resolved(t, None)
        self.svc.stream_resolved(t, {"interpolation": "auto", "restoration_fps": "30", "bogus": 1, "detection_model": ""})
        self.assertEqual([(p["quality"], p["overrides"]) for p in self.players],
                         [(None, {}), (None, {}), (None, {"interpolation": "auto", "restoration_fps": "30"})])

    def test_restoration_fps_30_only_forces_auto_on_fast_sources(self):
        self.player_reply = {"ok": True}
        t = self.token(resolved(heights=((1080, 30.0),)))
        self.svc.stream_resolved(t, {"restoration_fps": "30", "interpolation": "off"})
        self.assertEqual(self.players[0]["overrides"], {"restoration_fps": "30", "interpolation": "off"})

    def test_launch_when_the_player_is_not_running(self):
        t = self.token()
        r = self.svc.stream_resolved(t, {"height": 720})
        self.assertTrue(r["launched"])
        path, payload = self.launched[0]
        self.assertEqual((path, payload), ("https://x.test/v", self.players[0]))
        self.assertEqual(payload["quality"], 720)

    def test_guards(self):
        self.assertFalse(self.svc.stream_resolved("gone", {})["ok"])
        self.assertEqual(self.players, [])
        t = self.token(resolved(str(self.tmp / "missing.mp4"), kind="file", quals=False))
        self.assertEqual(self.svc.stream_resolved(t, {})["error"], "找不到影片檔案")
        self.player_reply = {"ok": False, "error": "bad source"}
        self.assertEqual(self.svc.stream_resolved(self.token(), {})["error"], "bad source")
        self.assertFalse(self.svc.is_player_busy())                  # rejected: the guard is released again

    def test_shutdown_is_cancelled_like_stream(self):
        self.player_reply = {"ok": True}
        with mock.patch.object(QS.platform_support, "supports_shutdown", return_value=True):
            self.svc._shutdown_started = True
            self.assertTrue(self.svc.stream_resolved(self.token(), {})["ok"])
            self.assertEqual(self.shutdowns, [["shutdown", "/a"]])

    def subtitles_ready(self):
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")      # a fresh store's cloud profile is unconfigured

    def test_subtitles_off_creates_no_stream_item(self):
        self.player_reply = {"ok": True}
        r = self.svc.stream_resolved(self.token(), {"subtitles": False})
        self.assertNotIn("stream_item", r)
        self.assertNotIn("stream_item", self.players[0])
        self.assertEqual(self.svc.store.stream_items(), [])

    def test_subtitles_on_url_creates_stream_item_and_tells_the_player(self):
        self.subtitles_ready()
        self.player_reply = {"ok": True}
        self.resolver_result = resolved("https://x.test/v", headers={"Referer": "https://x.test/"}, title="Clip")
        r = self.svc.stream_resolved(self.token(), {"height": 720, "subtitles": True})
        item, = self.svc.store.stream_items()
        self.assertEqual((r["ok"], r["stream_item"]), (True, item.id))
        self.assertEqual((item.kind, item.target, item.headers, item.title, item.stream["source_kind"]),
                         ("stream", "https://x.test/v", {"Referer": "https://x.test/"}, "Clip", "url"))
        self.assertTrue(item.settings["subtitles"])
        self.assertEqual(item.subtitle["status"], "queued")
        payload = self.players[0]                                    # the player's way of reading the source is unchanged
        self.assertEqual((payload["input"], payload["quality"], payload["stream_item"]), ("https://x.test/v", 720, item.id))
        self.assertEqual(self.svc.store.items(), [])                 # not a queue job
        self.assertTrue(self.svc.is_player_busy())
        self.assertEqual(self.svc._stream_created[item.id], self.now[0])   # T2.7: the "never got a watching" timer starts

    def test_subtitles_on_local_file(self):
        self.subtitles_ready()
        self.player_reply = {"ok": True}
        clip = self.tmp / "clip.mp4"
        clip.write_bytes(b"test")
        r = self.svc.stream_resolved(self.token(resolved(str(clip), kind="file", quals=False, title="")), {"subtitles": True})
        item, = self.svc.store.stream_items()
        self.assertEqual((item.target, item.stream["source_kind"], self.players[0]["stream_item"]),
                         (str(clip), "file", item.id))
        self.assertEqual(r["stream_item"], item.id)

    def test_rejected_open_removes_the_new_item_and_keeps_the_old_one(self):
        self.subtitles_ready()
        self.player_reply = {"ok": True}
        first = self.svc.stream_resolved(self.token(), {"subtitles": True})["stream_item"]
        self.player_reply = {"ok": False, "error": "bad source"}
        self.assertEqual(self.svc.stream_resolved(self.token(), {"subtitles": True})["error"], "bad source")
        self.assertEqual([it.id for it in self.svc.store.stream_items()], [first])

    def test_a_new_stream_replaces_the_previous_stream_item(self):
        self.subtitles_ready()
        self.player_reply = {"ok": True}
        first = self.svc.stream_resolved(self.token(), {"subtitles": True})["stream_item"]
        folder = self.svc.store.item_dir(first)
        second = self.svc.stream_resolved(self.token(), {"subtitles": True})["stream_item"]
        self.assertEqual([it.id for it in self.svc.store.stream_items()], [second])
        self.assertFalse(folder.exists())
        self.svc.stream_resolved(self.token(), {})                    # so does a stream without subtitles: the video changed
        self.assertEqual(self.svc.store.stream_items(), [])

    def test_unready_subtitle_configuration_refuses_before_opening_the_player(self):
        r = self.svc.stream_resolved(self.token(), {"subtitles": True})
        self.assertFalse(r["ok"])
        self.assertTrue(r["error"])
        self.assertEqual((self.players, self.launched, self.svc.store.stream_items()), ([], [], []))

    def test_plain_stream_payload_is_unchanged(self):
        self.player_reply = {"ok": True}
        self.svc.stream("https://x.test/plain")
        self.assertEqual(self.players, [{"cmd": "open", "input": "https://x.test/plain", "headers": {}}])


class StreamAudioWiringTests(IntakeBase):
    """T2.3 task D: stream_resolved attaches the subtitle audio, resolve() reports its kind, every removal path
    unregisters the url refresher. The start-time probe is mocked: no ffprobe, no network."""

    def setUp(self):
        super().setUp()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")
        self.player_reply = {"ok": True}
        patcher = mock.patch.object(stream_audio, "probe_audio_start", return_value=0.25)
        self.probe = patcher.start()
        self.addCleanup(patcher.stop)
        # attach creates daemon probes. Join this fixture's threads before its mocked probe disappears,
        # otherwise late removal cleanup can run inside the next test's global unregister mock.
        self.audio_probes = []
        real_attach = stream_audio.attach
        def attach(*args, **kwargs):
            thread = real_attach(*args, **kwargs)
            if args[0] is self.svc:
                self.audio_probes.append(thread)
            return thread
        attach_patch = mock.patch.object(stream_audio, "attach", side_effect=attach)
        attach_patch.start()
        self.addCleanup(attach_patch.stop)
        self.addCleanup(self.join_audio_probes)

    def join_audio_probes(self):
        for thread in self.audio_probes:
            thread.join(10)
            self.assertFalse(thread.is_alive(), "this fixture's audio probe must finish before the next test")

    def start_stream(self, r=None, ov=None):
        out = self.svc.stream_resolved(self.token(r or resolved()), ov if ov is not None else {"subtitles": True})
        self.assertTrue(out["ok"])
        item = self.svc.store.get(out["stream_item"])
        return item, item.stream.get("audio_marker") or ""

    @staticmethod
    def registered(marker):
        return media_input._key(marker) in media_input._refreshers

    def test_url_with_subtitles_gets_a_marker_a_refresher_and_a_probe(self):
        item, marker = self.start_stream()
        self.assertTrue(marker.endswith(stream_audio.MARKER_NAME))
        self.assertEqual(Path(marker).parent, self.svc.store.item_dir(item.id))
        self.assertTrue(Path(marker).is_file())
        self.assertTrue(item.source_path.endswith(".rstream"))
        self.assertEqual((item.source_path, item.stream["audio_kind"]), (marker, "lowest"))
        self.assertTrue(self.registered(marker))
        self.assertTrue(wait_for(lambda: self.svc.store.get(item.id).stream["audio_status"] == "ready"))
        done = self.svc.store.get(item.id)
        self.assertEqual((done.stream["audio_offset"], done.subtitle["audio_offset"]), (0.25, 0.25))
        self.assertEqual(self.players[0]["stream_item"], item.id)               # the player is told as before

    def test_audio_kinds_by_source(self):
        single = self.start_stream(resolved(heights=((720, 30.0),)))[0]
        self.assertEqual(single.stream["audio_kind"], "same")
        r = resolved()
        r.audio_format = {"id": "140", "url": "https://x.test/a.m4a", "headers": {}, "protocol": "https"}
        self.assertEqual(self.start_stream(r)[0].stream["audio_kind"], "audio_only")

    def test_local_file_has_no_marker_and_keeps_its_path(self):
        clip = self.tmp / "clip.mp4"
        clip.write_bytes(b"test")
        with mock.patch.object(media_input, "register_refresher") as register, \
                mock.patch.object(media_input, "unregister_refresher") as unregister:
            item, marker = self.start_stream(resolved(str(clip), kind="file", quals=False, title=""))
            self.assertEqual((marker, item.stream["audio_kind"], item.source_path), ("", "file", str(clip)))
            self.assertFalse((self.svc.store.item_dir(item.id) / stream_audio.MARKER_NAME).exists())
            register.assert_not_called()
            self.svc._drop_stream_item(item.id)                                  # removing the stream...
            # only this item's calls: an earlier test's probe thread may still unregister its own marker while
            # the mock is active
            own = [c for c in unregister.call_args_list if str(self.svc.store.item_dir(item.id)) in str(c.args[0])]
            self.assertEqual(own, [])
        self.assertTrue(clip.is_file())                                          # ...never touches the user's file
        self.assertEqual(clip.read_bytes(), b"test")

    def test_no_subtitles_means_no_attach(self):
        with mock.patch.object(stream_audio, "attach") as attach:
            out = self.svc.stream_resolved(self.token(), {"subtitles": False})
            self.assertNotIn("stream_item", out)
            attach.assert_not_called()
        self.assertEqual(self.svc.store.stream_items(), [])
        self.probe.assert_not_called()

    def test_attach_gets_the_played_height(self):
        with mock.patch.object(stream_audio, "attach") as attach:
            self.svc.stream_resolved(self.token(), {"subtitles": True, "height": 720})
            self.svc.stream_resolved(self.token(), {"subtitles": True})
        self.assertEqual([c.kwargs["played_height"] for c in attach.call_args_list], [720, 1080])

    def test_the_probe_does_not_hold_back_the_player(self):
        gate = threading.Event()
        self.probe.side_effect = lambda *a, **k: (gate.wait(5), 0.0)[1]
        self.addCleanup(gate.set)
        item, marker = self.start_stream()                                       # returned although the probe is stuck
        self.assertEqual(self.svc.store.get(item.id).stream["audio_status"], "probing")
        self.assertEqual(len(self.players), 1)
        gate.set()
        self.assertTrue(wait_for(lambda: self.svc.store.get(item.id).stream["audio_status"] == "ready"))

    def test_an_unreadable_source_fails_the_subtitle_audio_but_still_plays(self):
        r = resolved(quals=False)                                                # no renditions, no direct media url
        item, marker = self.start_stream(r)
        self.assertEqual(marker, "")
        self.assertEqual((item.stream["audio_status"], item.stream["audio_error"]),
                         ("failed", stream_audio.ERR_UNSUPPORTED))
        self.assertEqual(len(self.players), 1)

    def test_resolve_info_reports_the_subtitle_audio_kind(self):
        def kind(r):
            self.resolver_result = r
            return self.svc.resolve("https://x.test/v")["info"]["subtitle_audio"]
        self.assertEqual(kind(resolved()), "lowest")
        self.assertEqual(kind(resolved(heights=((720, 30.0),))), "same")
        audio = resolved()
        audio.audio_format = {"id": "140", "url": "https://x.test/a.m4a", "headers": {}, "protocol": "https"}
        self.assertEqual(kind(audio), "audio_only")
        self.assertEqual(kind(resolved(str(self.tmp / "c.mp4"), kind="file", quals=False)), "file")
        self.assertIsNone(kind(resolved(quals=False)))                           # nothing readable: the panel falls back

    # --- every removal path unregisters the refresher ----------------------------------------------------------
    def test_stream_end_unregisters(self):
        item, marker = self.start_stream()
        self.assertTrue(self.registered(marker))
        self.assertTrue(wait_for(lambda: self.svc.store.get(item.id).stream["audio_status"] == "ready"))
        self.assertEqual(self.svc._on_message({"cmd": "stream_end", "id": item.id}), {"ok": True})
        self.assertFalse(self.registered(marker))
        self.assertFalse(Path(marker).exists())

    def test_replacing_the_stream_unregisters_the_old_one(self):
        old, old_marker = self.start_stream()
        new, new_marker = self.start_stream()
        self.assertFalse(self.registered(old_marker))
        self.assertTrue(self.registered(new_marker))
        self.assertEqual([it.id for it in self.svc.store.stream_items()], [new.id])

    def test_a_refused_open_unregisters_the_new_one_and_keeps_the_old(self):
        old, old_marker = self.start_stream()
        self.player_reply = {"ok": False, "error": "bad source"}
        self.assertFalse(self.svc.stream_resolved(self.token(), {"subtitles": True})["ok"])
        self.assertTrue(self.registered(old_marker))
        self.assertEqual([it.id for it in self.svc.store.stream_items()], [old.id])

    def test_a_refused_open_unregisters(self):
        self.player_reply = {"ok": False, "error": "bad source"}
        entered, release = threading.Event(), threading.Event()
        def held_probe(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(10))
            return 0.25
        self.probe.side_effect = held_probe
        self.addCleanup(release.set)
        with mock.patch.object(media_input, "unregister_refresher", wraps=media_input.unregister_refresher) as un:
            try:
                out = self.svc.stream_resolved(self.token(), {"subtitles": True})
                self.assertFalse(out["ok"])
                self.assertTrue(entered.wait(5))
                # Only calls for this queue: another fixture's probe can finish in this global mock.
                own = lambda: [c.args[0] for c in un.call_args_list
                               if str(c.args[0]).startswith(str(self.svc.store.root))]
                self.assertEqual(len(own()), 1)             # synchronous refused-open cleanup
                marker = own()[0]
                self.assertTrue(marker.endswith(stream_audio.MARKER_NAME))
                self.assertFalse(self.registered(marker))
                # Windows may retain the marker while run_remote holds it; next startup clears leftovers.
                self.assertEqual(self.svc.store.stream_items(), [])
            finally:
                release.set()
                self.join_audio_probes()
            # The probe sees its item was removed and unregisters idempotently as well. Both orderings
            # are legitimate; controlling this one verifies the late cleanup cannot resurrect anything.
            self.assertEqual(own(), [marker, marker])
            self.assertFalse(self.registered(marker))
            self.assertEqual(self.svc.store.stream_items(), [])

    def test_heartbeat_expiry_unregisters(self):
        item, marker = self.start_stream()
        self.now[0] += self.svc.stream_first_watch_grace + 1
        self.svc._expire_streams()
        self.assertIsNone(self.svc.store.get(item.id))
        self.assertFalse(self.registered(marker))

    def test_stop_unregisters_every_stream(self):
        first, m1 = self.start_stream()
        second, m2 = self.start_stream()
        self.svc.store.add_stream("https://x.test/w", "url", {}, {"subtitles": True}, title="No audio")   # no marker
        self.assertTrue(self.registered(m2))
        self.svc.stop()
        self.assertFalse(self.registered(m1))
        self.assertFalse(self.registered(m2))
        self.assertEqual(self.svc.store.stream_items(), [])


class AddResolvedTests(IntakeBase):
    def test_cover_survives_queue_intake_without_changing_video_frames(self):
        source = resolved()
        source.cover = "data:image/jpeg;base64,AAAA"
        item_id = self.svc.add_resolved([self.token(source)])["ids"][0]
        item = self.svc.store.get(item_id)
        self.assertEqual((self.svc.store.item_dir(item_id) / "cover.txt").read_text(), source.cover)
        fields = {}
        self.svc._thumb_fields(item, fields)
        self.assertEqual(fields["cover"], source.cover)
        self.assertEqual(item.target, source.target)

    def test_pinned_overrides_models_and_quality(self):
        t = self.token(resolved(heights=((1080, 30.0), (720, 30.0)), headers={"Cookie": "c"}, title="Clip"))
        r = self.svc.add_resolved([t], {"height": 720, "interpolation": "auto", "encoder": "h264", "save_compare": True,
                                        "detection_model": "v4-accurate", "restoration_model": "deepmosaics",
                                        "restoration_fps": "30"})
        self.assertTrue(r["ok"])
        self.assertEqual((len(r["ids"]), r["errors"]), (1, []))
        self.assertTrue(r["message"])
        it = self.svc.store.get(r["ids"][0])
        self.assertEqual((it.target, it.kind, it.quality, it.title, it.headers),
                         ("https://x.test/v", "url", 720, "Clip", {"Cookie": "c"}))
        self.assertEqual(it.settings["pinned"], {"interpolation": "full", "encoder": "h264", "save_compare": True})
        self.assertEqual((it.settings["detection_model"], it.settings["restoration_model"]),
                         ("v4-accurate", "deepmosaics"))
        self.assertNotIn("restoration_fps", it.settings)              # stream only
        self.assertEqual(it.cache_key, queue_store.cache_key(it.target, it.quality, it.settings))
        plain = self.svc.add_resolved([t], {"height": 720, "detection_model": "v4-accurate",
                                            "restoration_model": "deepmosaics"})["ids"][0]
        self.assertNotEqual(self.svc.store.get(plain).cache_key, it.cache_key)   # pins are part of the key
        self.assertNotIn("pinned", self.svc.store.get(plain).settings)
        again = self.svc.add_resolved([t], {"height": 720, "interpolation": "auto", "encoder": "h264",
                                            "save_compare": True, "detection_model": "v4-accurate",
                                            "restoration_model": "deepmosaics"})
        self.assertEqual(again["ids"], [it.id])                       # same key: no duplicate item

    def test_quality_only_when_offered(self):
        urls = self.token(resolved(heights=((1080, 30.0), (720, 30.0))))
        ext = self.token(resolved("https://x.test/ext", heights=((1080, 30.0),), quals=False))
        local = self.token(resolved(str(self.tmp / "a.mp4"), kind="file", heights=((720, 30.0),), quals=False))
        r = self.svc.add_resolved([urls, ext, local], {"height": 1080})
        q = [self.svc.store.get(i).quality for i in r["ids"]]
        self.assertEqual(q, [1080, None, None])
        r = self.svc.add_resolved([urls], {"height": 360})
        self.assertIsNone(self.svc.store.get(r["ids"][0]).quality)
        r = self.svc.add_resolved([self.token(resolved("https://x.test/other"))], None)
        self.assertIsNone(self.svc.store.get(r["ids"][0]).quality)

    def test_batch_errors_and_bad_models(self):
        good = self.token(resolved("https://x.test/a"))
        r = self.svc.add_resolved([good, "gone"], {})
        self.assertEqual((len(r["ids"]), len(r["errors"]), r["ok"]), (1, 1, False))
        r = self.svc.add_resolved([good], {"detection_model": "nope"})
        self.assertEqual((r["ok"], r["ids"]), (False, []))
        self.assertEqual(len(self.svc.store.items()), 1)
        self.assertFalse(self.svc.add_resolved([], {})["ok"])

    def test_build_job_and_settings_changes_keep_pins(self):
        pinned = self.svc.add_resolved([self.token(resolved("https://x.test/a"))],
                                       {"interpolation": "off", "encoder": "h264", "save_compare": True})["ids"][0]
        free = self.svc.add_resolved([self.token(resolved("https://x.test/b"))], {})["ids"][0]
        half = self.svc.add_resolved([self.token(resolved("https://x.test/c"))], {"encoder": "h264"})["ids"][0]
        self.svc.store.set_settings(interpolation="full", encoder="hevc", save_compare=False)
        job = self.svc._build_job(self.svc.store.get(pinned))["settings"]
        self.assertEqual((job["interpolation"], job["encoder"], job["compare"]), ("off", "h264", "on"))
        self.assertEqual(job["compare_original"], "copy")
        job = self.svc._build_job(self.svc.store.get(free))["settings"]
        self.assertEqual((job["interpolation"], job["encoder"], job["compare"]), ("full", "hevc", "off"))
        job = self.svc._build_job(self.svc.store.get(half))["settings"]
        self.assertEqual((job["interpolation"], job["encoder"], job["compare"]), ("full", "h264", "off"))
        self.svc.store.set_settings(interpolation="off", encoder="h264", save_compare=True)     # a later change
        job = self.svc._build_job(self.svc.store.get(pinned))["settings"]
        self.assertEqual((job["interpolation"], job["encoder"], job["compare"]), ("off", "h264", "on"))
        job = self.svc._build_job(self.svc.store.get(half))["settings"]
        self.assertEqual((job["interpolation"], job["compare"]), ("off", "on"))
        self.assertEqual(self.svc.store.get(pinned).settings["pinned"],
                         {"interpolation": "off", "encoder": "h264", "save_compare": True})   # still there

    def test_pinned_interpolation_is_still_off_without_rife(self):
        i = self.svc.add_resolved([self.token(resolved("https://x.test/a"))], {"interpolation": "auto"})["ids"][0]
        with mock.patch.object(QS.platform_support, "capabilities", lambda: {"rife": False}):
            self.assertEqual(self.svc._build_job(self.svc.store.get(i))["settings"]["interpolation"], "off")

    def test_apply_model_settings_keeps_pins_and_rekeys(self):
        self.write_player(detection_model="v4-fast", restoration_model="basicvsrpp-v1.2")
        i = self.svc.add_resolved([self.token(resolved("https://x.test/a"))], {"interpolation": "auto",
                                                                              "detection_model": "v4-accurate"})["ids"][0]
        before = self.svc.store.get(i)
        self.assertEqual(before.settings["detection_model"], "v4-accurate")
        self.svc.store.set_settings(detection_model="v4-fast")
        self.assertTrue(self.svc.apply_model_settings(i)["ok"])
        after = self.svc.store.get(i)
        self.assertEqual(after.settings["detection_model"], "v4-fast")
        self.assertEqual(after.settings["pinned"], {"interpolation": "full"})
        self.assertNotEqual(after.cache_key, before.cache_key)
        self.assertEqual(after.cache_key, queue_store.cache_key(after.target, after.quality, after.settings))

    def test_eta_honours_the_pins(self):
        self.save_profile()
        i = self.svc.add_resolved([self.token(resolved("https://x.test/a"))], {"interpolation": "off"})["ids"][0]
        j = self.svc.add_resolved([self.token(resolved("https://x.test/b"))], {})["ids"][0]
        facts = {"height": 720, "fps": 30.0, "duration": 600.0}
        for item_id in (i, j):
            self.svc.store.update(item_id, state="queued", source_path="x.mp4", source_status="complete")
            self.svc._eta_facts[item_id] = (("x.mp4", "complete"), dict(facts), QS.time.monotonic())
        est = self.svc._eta_estimates(self.svc.store.items(), self.svc._effective_queue_settings())
        view = P.for_detector(intake_profile(), "v4-fast")
        self.assertAlmostEqual(est[i], P.job_seconds(view, duration_s=600.0, height=720, src_fps=30.0, mode="off"))
        self.assertAlmostEqual(est[j], P.job_seconds(view, duration_s=600.0, height=720, src_fps=30.0, mode="full"))


class SaveDefaultsTests(IntakeBase):
    def reloads(self):
        return [p for p in self.players if p.get("cmd") == "settings_reload"]

    def test_writes_player_and_queue_settings(self):
        self.write_player(vsr="off", seek_prebuffer=2.0)
        r = self.svc.save_defaults({"detection_model": "v4-accurate", "restoration_model": "deepmosaics",
                                    "interpolation": "auto", "restoration_fps": "30", "encoder": "h264",
                                    "save_compare": True, "height": 720})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["changed"], sorted(["detection_model", "restoration_model", "interpolation",
                                               "restoration_fps", "encoder", "save_compare"]))
        saved = self.player_settings()
        self.assertEqual(saved, {"vsr": "off", "seek_prebuffer": 2.0, "detection_model": "v4-accurate",
                                 "restoration_model": "deepmosaics", "interpolation": "auto", "restoration_fps": "30"})
        s = self.svc.store.get_settings()
        self.assertEqual((s["detection_model"], s["restoration_model"], s["interpolation"], s["encoder"],
                          s["save_compare"]), ("v4-accurate", "deepmosaics", "full", "h264", True))
        self.assertFalse(list(self.player_file.parent.glob("*.tmp")))
        d = self.svc.defaults()
        self.assertEqual((d["stream"]["interpolation"], d["queue"]["interpolation"]), ("auto", "auto"))

    def test_only_the_given_keys_and_off(self):
        self.write_player(interpolation="auto", detection_model="v4-accurate")
        self.assertTrue(self.svc.save_defaults({"interpolation": "off"})["ok"])
        self.assertEqual(self.player_settings(), {"interpolation": "off", "detection_model": "v4-accurate"})
        self.assertEqual(self.svc.store.get_settings()["interpolation"], "off")
        self.assertIsNone(self.svc.store.get_settings()["detection_model"])
        r = self.svc.save_defaults({"height": 720})
        self.assertEqual((r["ok"], r["changed"]), (True, []))
        self.assertTrue(self.svc.save_defaults(None)["ok"])

    def test_creates_the_player_file_and_works_while_the_player_runs(self):
        self.player_reply = {"ok": True}
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.assertTrue(self.svc.is_player_busy())
        self.assertTrue(self.svc.save_defaults({"restoration_fps": "30"})["ok"])
        self.assertEqual(self.player_settings(), {"restoration_fps": "30"})

    def test_a_running_player_is_told_to_reload_the_file_after_it_was_written(self):
        self.player_reply = {"ok": True}
        seen = []
        real = self.svc._send_to_player
        self.svc._send_to_player = lambda p: seen.append((p, self.player_file.exists() and self.player_settings())) or real(p)
        self.assertTrue(self.svc.save_defaults({"restoration_fps": "30", "interpolation": "auto"})["ok"])
        self.assertEqual(seen, [({"cmd": "settings_reload"}, {"restoration_fps": "30", "interpolation": "auto"})])

    def test_no_reload_message_without_player_keys_or_after_a_refusal(self):
        self.assertTrue(self.svc.save_defaults({"encoder": "h264", "save_compare": True})["ok"])   # queue-only keys
        self.assertTrue(self.svc.save_defaults({"height": 720})["ok"])
        self.assertFalse(self.svc.save_defaults({"detection_model": "not-installed", "interpolation": "auto"})["ok"])
        self.assertEqual(self.reloads(), [])

    def test_no_player_running_or_a_failing_send_is_not_an_error(self):
        self.player_reply = None                                 # no player running: send_to_player returns None
        self.assertTrue(self.svc.save_defaults({"restoration_fps": "30"})["ok"])
        self.assertEqual(len(self.reloads()), 1)

        def boom(_payload):
            raise OSError("pipe broke")
        self.svc._send_to_player = boom
        r = self.svc.save_defaults({"restoration_fps": "original"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.player_settings(), {"restoration_fps": "original"})

    def test_a_failed_write_does_not_notify_the_player(self):
        self.player_file.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(PS, "save_file", side_effect=OSError("disk full")):
            r = self.svc.save_defaults({"restoration_fps": "30"})
        self.assertFalse(r["ok"])
        self.assertEqual(self.reloads(), [])

    def test_refusal_leaves_the_player_file_alone(self):
        self.write_player(detection_model="v4-fast")
        r = self.svc.save_defaults({"detection_model": "not-installed", "interpolation": "auto"})
        self.assertFalse(r["ok"])
        self.assertEqual(self.player_settings(), {"detection_model": "v4-fast"})
        self.assertEqual(self.svc.store.get_settings()["interpolation"], "full")


class ApiTests(IntakeBase):
    def make_api(self, paths=()):
        window = types.SimpleNamespace(create_file_dialog=lambda *a, **k: paths, evaluate_js=lambda js: None)
        app = queue_app.QueueApp(self.svc, hidden=False)
        app.window = window
        return queue_app.Api(app), app

    def test_add_files_returns_the_paths_and_adds_nothing(self):
        fake = types.SimpleNamespace(FileDialog=types.SimpleNamespace(OPEN=1))
        with mock.patch.dict(sys.modules, {"webview": fake}):
            api, _app = self.make_api(("C:\\v\\a.mp4", "C:\\v\\b.mp4"))
            self.assertEqual(api.add_files(), {"ok": True, "paths": ["C:\\v\\a.mp4", "C:\\v\\b.mp4"]})
            api, _app = self.make_api(None)
            self.assertEqual(api.add_files(), {"ok": True, "paths": []})
        self.assertEqual(self.svc.store.items(), [])

    def test_api_passthrough(self):
        api, _app = self.make_api()
        self.assertTrue(api.parse_inputs("https://a.test/1")["ok"])
        info = api.resolve("https://a.test/1")["info"]
        self.assertTrue(api.defaults()["choices"])
        self.assertFalse(api.estimate([info["token"]])["has_profile"])
        self.assertEqual(len(api.add_resolved([info["token"]], {})["ids"]), 1)
        self.player_reply = {"ok": True}
        self.assertTrue(api.stream_resolved(info["token"])["ok"])
        self.assertTrue(api.save_defaults({"restoration_fps": "30"})["ok"])
        self.assertFalse(api.resolve("")["ok"])

    def test_drop_opens_the_panel_and_adds_nothing(self):
        api, app = self.make_api()
        calls = []
        app.window.evaluate_js = lambda js: calls.append(js) or True
        app.on_drop({"dataTransfer": {"files": [{"pywebviewFullPath": "C:\\v\\a.mp4"}, {"name": "no path"}]}})
        self.assertEqual(len(calls), 1)
        self.assertIn("window.ohStreamer.openPaths(", calls[0])
        self.assertIn(json.dumps(["C:\\v\\a.mp4"]), calls[0])
        self.assertEqual(self.svc.store.items(), [])
        app.window.evaluate_js = lambda js: 1 / 0                      # a failing page call is only logged
        app.on_drop({"dataTransfer": {"files": [{"pywebviewFullPath": "C:\\v\\a.mp4"}]}})
        self.assertEqual(self.svc.store.items(), [])

    def test_get_state_includes_duration_and_source_fps(self):
        it = self.svc.store.add("https://a.test/v", "url", {}, "best", {})
        self.svc._source_facts = lambda item: {"height": 720, "fps": 29.97, "duration": 1800.5}
        state = self.svc.get_state()
        item_dict = next(x for x in state["items"] if x["id"] == it.id)
        self.assertEqual(item_dict.get("duration"), 1800.5)
        self.assertEqual(item_dict.get("source_fps"), 29.97)


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
