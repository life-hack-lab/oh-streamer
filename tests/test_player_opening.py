"""Focused tests for source resolution and the asynchronous player opener."""

from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace


STREAMER_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = STREAMER_ROOT / "src"
for import_root in (STREAMER_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from oh_streamer.player import open_input, opening  # noqa: E402
from oh_streamer.sources import qualities  # noqa: E402


class FakeMeta:
    video_height = 720
    video_width = 1280
    video_fps_exact = 25


class FakeSource:
    meta = FakeMeta()

    def __init__(self, src, headers, audio_src):
        self.src, self.headers, self.audio_src = src, dict(headers), audio_src

    def describe(self):
        return f"fake source: {self.src}"


def dependencies(**overrides):
    values = {
        "source_factory": FakeSource,
        "find_resolver": lambda _url: None,
        "resolve_external": lambda *_a: (_ for _ in ()).throw(AssertionError("unexpected external resolver")),
        "resolve_with_ytdlp": lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("unexpected yt-dlp")),
        "log": lambda _message: None,
        "is_url": lambda value: str(value).lower().startswith(("http://", "https://")),
        "looks_like_media_url": lambda _value: False,
        "pick_quality": qualities.pick_quality,
        "quality_label": qualities.quality_label,
        "source_label": lambda value: f"label:{value}",
    }
    values.update(overrides)
    return opening.OpeningDependencies(**values)


def args(**overrides):
    values = {"fake_formats": None, "ytdl": False, "max_height": 1080}
    values.update(overrides)
    return SimpleNamespace(**values)


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.001)
    return bool(predicate())


def wait_for_result(opener, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = opener.poll()
        if result is not None:
            return result
        threading.Event().wait(0.001)
    return opener.poll()


class ResolveAndProbeTests(unittest.TestCase):
    def test_fake_ytdlp_resolution_keeps_user_header_precedence_and_refresh(self):
        page = "https://example.test/watch/1"
        user_headers = {"User-Agent": "user-agent", "Cookie": "session=1"}
        picked = {"id": "720", "height": 720, "label": "720p", "url": "https://cdn.test/video",
                  "audio_url": "https://cdn.test/audio",
                  "http_headers": {"User-Agent": "format-agent", "X-Format": "format"}}
        all_qualities = [picked, {**picked, "id": "480", "height": 480, "label": "480p"}]
        fresh = [{**picked, "url": "https://cdn.test/refreshed"}]
        calls = []
        sources, logs = [], []

        def source_factory(*source_args):
            source = FakeSource(*source_args)
            sources.append(source)
            return source

        def resolve(page_url, headers, fake, **kwargs):
            calls.append((page_url, dict(headers), fake, kwargs))
            if kwargs.get("refresh"):
                return fresh, "fresh title", None
            return all_qualities, "resolved title", (
                "https://cdn.test/thumbnail", {"User-Agent": "thumb-agent", "X-Thumb": "thumb"})

        deps = dependencies(source_factory=source_factory, resolve_with_ytdlp=resolve, log=logs.append)
        opened = opening.open_source(page, user_headers, args(fake_formats="formats.json"), 720, deps=deps)

        self.assertEqual(calls[0], (page, user_headers, "formats.json", {}))
        self.assertEqual(sources[0].src, "https://cdn.test/video")
        self.assertEqual(sources[0].audio_src, "https://cdn.test/audio")
        self.assertEqual(sources[0].headers, {"User-Agent": "user-agent", "X-Format": "format",
                                             "Cookie": "session=1"})
        self.assertEqual(opened.title, "resolved title")
        self.assertEqual(opened.source_text, f"label:{page}")
        self.assertEqual(opened.thumb_src,
                         ("https://cdn.test/thumbnail", {"User-Agent": "user-agent", "X-Thumb": "thumb",
                                                         "Cookie": "session=1"}))
        self.assertEqual(opened.refresh(), fresh)
        self.assertEqual(calls[1], (page, user_headers, "formats.json", {"refresh": True}))
        self.assertTrue(any("thumbnail source" in message for message in logs))

    def test_external_resolver_headers_are_overridden_by_user_headers(self):
        page = "https://example.test/protected"
        resolved_calls = []
        deps = dependencies(
            find_resolver=lambda value: {"name": "test"} if value == page else None,
            resolve_external=lambda url, headers, resolver: (
                resolved_calls.append((url, dict(headers), resolver)) or
                ("https://cdn.test/resolved", {"Authorization": "resolver", "X-Resolver": "yes"}, "resolved")),
        )
        opened = opening.open_source(page, {"Authorization": "user", "Cookie": "c=1"}, args(), None,
                                     fake_any=False, deps=deps)

        self.assertEqual(resolved_calls, [(page, {"Authorization": "user", "Cookie": "c=1"}, {"name": "test"})])
        self.assertEqual(opened.source.src, "https://cdn.test/resolved")
        self.assertEqual(opened.source.headers, {"Authorization": "user", "X-Resolver": "yes", "Cookie": "c=1"})
        self.assertEqual(opened.title, "resolved")

    def test_browser_resolver_expands_qualities_and_refreshes_media_url(self):
        page = "https://video.example/watch/1"
        external_calls, ytdlp_calls = [], []

        def external(url, headers, resolver):
            external_calls.append((url, dict(headers)))
            version = len(external_calls)
            return (f"https://media.example/master{version}.m3u8",
                    {"Referer": f"ref{version}", "Authorization": "resolver"}, "Page title")

        def ytdlp(url, headers, fake):
            ytdlp_calls.append((url, dict(headers), fake))
            return ([{"id": "720", "height": 720, "label": "720p", "url": url.replace("master", "variant"),
                     "http_headers": {"Authorization": "format"}}], "Media title", (url + "?thumb", {}))

        deps = dependencies(find_resolver=lambda _url: {"builtin": "browser"},
                            resolve_external=external, resolve_with_ytdlp=ytdlp)
        opened = opening.open_source(page, {"Authorization": "user", "Cookie": "c=1"}, args(), 720,
                                     fake_any=False, deps=deps)
        self.assertEqual(opened.title, "Page title")
        self.assertEqual(opened.source.src, "https://media.example/variant1.m3u8")
        self.assertEqual(opened.source.headers, {"Referer": "ref1", "Authorization": "format"})
        self.assertEqual(opened.thumb_src[1]["Referer"], "ref1")
        self.assertEqual(opened.refresh()[0]["url"], "https://media.example/variant2.m3u8")
        self.assertEqual(opened.refresh()[0]["http_headers"]["Referer"], "ref3")
        self.assertEqual(external_calls, [(page, {"Authorization": "user", "Cookie": "c=1"})] * 3)
        self.assertEqual([call[0] for call in ytdlp_calls],
                         [f"https://media.example/master{i}.m3u8" for i in (1, 2, 3)])

    def test_direct_source_builds_quality_from_probe_metadata(self):
        deps = dependencies()
        resolved = opening.resolve_input("sample.mp4", {}, args(), None, deps=deps)
        opened = opening.probe_input(resolved, deps=deps)

        self.assertTrue(resolved["src"].endswith("sample.mp4"))
        self.assertEqual(opened.chosen["id"], "source")
        self.assertEqual(opened.chosen["height"], 720)
        self.assertEqual(opened.chosen["label"], "720p")
        self.assertEqual(opened.thumb_src, (resolved["src"], {}))

    def test_probe_fills_only_unknown_selected_quality_metadata(self):
        unknown = {"id": "opaque", "height": 0, "width": 0, "fps": None, "label": "source",
                   "url": "https://media.example/opaque", "http_headers": {}}
        known = {"id": "known", "height": 480, "width": 854, "fps": 25.0, "label": "480p",
                 "url": "https://media.example/known", "http_headers": {}}
        opened = opening.probe_input({"input": "https://page.example/watch", "src": unknown["url"],
            "headers": {}, "title": "Title", "quals": [unknown, known], "chosen": unknown,
            "refresh": None, "thumb_src": None, "user_headers": {}}, deps=dependencies())
        self.assertEqual((opened.chosen["height"], opened.chosen["fps"], opened.chosen["label"]),
                         (720, 25.0, "720p"))
        self.assertEqual(known["label"], "480p")

    def test_probe_keeps_known_selected_quality_metadata(self):
        known = {"id": "known", "height": 1080, "width": 1920, "fps": 60.0, "label": "1080p60",
                 "url": "https://media.example/known", "http_headers": {}}
        opened = opening.probe_input({"input": "https://page.example/watch", "src": known["url"],
            "headers": {}, "title": "Title", "quals": [known], "chosen": known,
            "refresh": None, "thumb_src": None, "user_headers": {}}, deps=dependencies())
        self.assertEqual((opened.chosen["height"], opened.chosen["fps"], opened.chosen["label"]),
                         (1080, 60.0, "1080p60"))


class OpenerTests(unittest.TestCase):
    @staticmethod
    def make_opener(open_source, log=None):
        return opening.Opener(args(), None, open_source=open_source, log=log or (lambda _message: None))

    def test_latest_submit_wins_when_older_resolution_finishes_later(self):
        old_started, release_old, old_dropped = threading.Event(), threading.Event(), threading.Event()

        def fake_open(target, *_args, **_kwargs):
            if target == "old":
                old_started.set()
                if not release_old.wait(2):
                    raise TimeoutError("test did not release old resolution")
            return opening.Opened(title=target)

        opener = self.make_opener(fake_open, log=lambda msg: old_dropped.set() if "dropped" in msg else None)
        opener.submit(open_input.OpenInput("old"), "old request", 3.0)
        self.assertTrue(old_started.wait(2))
        opener.submit(open_input.OpenInput("new"), "new request", 9.0)
        try:
            result = wait_for_result(opener)
            self.assertIsNotNone(result)
            self.assertEqual((result[0].title, result[0].start, result[3]), ("new", 9.0, "new request"))
        finally:
            release_old.set()
        self.assertTrue(old_dropped.wait(2))
        self.assertIsNone(opener.poll())
        self.assertFalse(opener.busy)

    def test_cancel_drops_inflight_resolution(self):
        started, release, dropped = threading.Event(), threading.Event(), threading.Event()

        def fake_open(target, *_args, **_kwargs):
            started.set()
            if not release.wait(2):
                raise TimeoutError("test did not release resolution")
            return opening.Opened(title=target)

        opener = self.make_opener(fake_open, log=lambda msg: dropped.set() if "dropped" in msg else None)
        opener.submit(open_input.OpenInput("slow"), "slow", 0.0)
        self.assertTrue(started.wait(2))
        opener.cancel()
        self.assertFalse(opener.busy)
        self.assertIsNone(opener.poll())
        release.set()
        self.assertTrue(dropped.wait(2))
        self.assertIsNone(opener.poll())

    def test_cancel_drops_finished_result_not_yet_polled(self):
        opener = self.make_opener(lambda target, *_a, **_kw: opening.Opened(title=target))
        opener.submit(open_input.OpenInput("done"), "finished", 1.0)
        self.assertTrue(wait_until(lambda: not opener.busy))
        opener.cancel()
        self.assertIsNone(opener.poll())

    def test_failure_is_returned_as_a_clear_result_and_missing_file_is_reported(self):
        opener = self.make_opener(lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("bad source")))
        opener.submit(open_input.OpenInput("https://example.test/video"), "broken")
        result = wait_for_result(opener)
        self.assertIsNotNone(result)
        self.assertIsNone(result[0])
        self.assertIsInstance(result[1], ValueError)
        self.assertGreaterEqual(result[2], 0)
        self.assertEqual(result[3], "broken")
        self.assertEqual(opening.short_error(result[1]), "無法開啟：bad source")

        missing = str(STREAMER_ROOT / "definitely-missing-video.mp4")
        opener.submit(open_input.OpenInput(missing, kind="file"), "missing")
        missing_result = wait_for_result(opener)
        self.assertIsNone(missing_result[0])
        self.assertIsInstance(missing_result[1], open_input.OpenInputError)
        self.assertEqual(opening.short_error(missing_result[1]), "找不到檔案：definitely-missing-video.mp4")


class OpenerPerRequestTests(unittest.TestCase):
    """A request's own quality / overrides; the default quality_spec stays what plain submits use."""

    def test_quality_is_per_request_and_the_default_is_unchanged(self):
        specs = []

        def fake_open(target, _headers, _args, spec, **_kwargs):
            specs.append(spec)
            return opening.Opened(title=target)

        opener = opening.Opener(args(), 1080, open_source=fake_open, log=lambda _message: None)
        for quality in (None, 480, None):
            opener.submit(open_input.OpenInput("https://example.test/v"), "r", quality=quality)
            self.assertIsNotNone(wait_for_result(opener))
        self.assertEqual(specs, [1080, 480, 1080])
        self.assertEqual(opener.quality_spec, 1080)

    def test_requested_height_picks_the_nearest_available_like_the_quality_spec(self):
        quals = [{"id": str(h), "height": h} for h in (1080, 720, 480)]
        self.assertEqual(qualities.pick_quality(quals, 700)["height"], 480)
        self.assertEqual(qualities.pick_quality(quals, 720)["height"], 720)
        self.assertEqual(qualities.pick_quality(quals, 240)["height"], 480)

    def test_overrides_travel_with_the_opened_source(self):
        opener = opening.Opener(args(), None, open_source=lambda t, *_a, **_k: opening.Opened(title=t),
                                log=lambda _message: None)
        opener.submit(open_input.OpenInput("a"), "a", overrides={"interpolation": "auto"})
        self.assertEqual(wait_for_result(opener)[0].overrides, {"interpolation": "auto"})
        opener.submit(open_input.OpenInput("b"), "b")
        self.assertEqual(wait_for_result(opener)[0].overrides, {})

    def test_queue_lookup_gets_the_requested_quality_only_when_there_is_one(self):
        calls = []
        opener = opening.Opener(args(), 1080, open_source=lambda t, *_a, **_k: opening.Opened(title=t),
                                log=lambda _message: None)
        opener.lookup = lambda *a: calls.append(a) or None
        opener.submit(open_input.OpenInput("https://example.test/v"), "plain")
        wait_for_result(opener)
        opener.submit(open_input.OpenInput("https://example.test/v"), "q", quality=480)
        wait_for_result(opener)
        self.assertEqual([c[1:] for c in calls], [(), (480,)])


if __name__ == "__main__":
    unittest.main()
