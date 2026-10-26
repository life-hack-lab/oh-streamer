# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import tempfile
import unittest
from pathlib import Path

from oh_streamer.queue.queue_store import QueueStore, cache_key

SETTINGS = {"detection_model": "v4-fast", "restoration_model": "basicvsrpp-v1.2", "fp16": True,
            "max_clip_length": 180, "encoder": "hevc", "interpolation": "full"}


def write_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class QueueStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "queue"
        self.store = QueueStore(self.root)

    def add(self, n, **kw):
        return self.store.add(f"https://x/{n}", "url", {"Cookie": "a=b"}, "best", dict(SETTINGS, **kw), title=f"t{n}")

    def test_add_and_defaults(self):
        a = self.add(1)
        self.assertEqual((a.state, a.order, a.kind, a.title), ("waiting", 0, "url", "t1"))
        self.assertEqual(len(a.id), 8)
        s = self.store.get_settings()
        self.assertEqual((s["cap_gb"], s["encoder"], s["interpolation"], s["save_compare"], s["prevent_sleep"],
                          s["shutdown_when_done"]), (50, "hevc", "full", False, True, False))
        self.assertNotIn("subtitle_translation_igpu", s)
        self.assertTrue((self.root / "queue.json").exists())

    def test_dedupe_by_cache_key(self):
        a = self.add(1)
        b = self.add(1)
        self.assertEqual(a.id, b.id)
        self.assertEqual(len(self.store.items()), 1)
        self.assertNotEqual(self.add(1, fp16=False).id, a.id)

    def test_cache_key_ignores_non_pixel_settings(self):
        k = cache_key("u", "best", SETTINGS)
        self.assertEqual(k, cache_key("u", "best", dict(SETTINGS, prevent_sleep=False, cap_gb=1)))
        self.assertNotEqual(k, cache_key("u", "720p", SETTINGS))
        self.assertNotEqual(k, cache_key("v", "best", SETTINGS))
        self.assertNotEqual(k, cache_key("u", "best", dict(SETTINGS, encoder="h264")))
        self.assertNotEqual(k, cache_key("u", "best", dict(SETTINGS, save_compare=True)))
        self.assertEqual(cache_key("u", "best", dict(SETTINGS, save_compare=True)),
                         cache_key("u", "best", dict(SETTINGS, compare="on")))

    def test_persistence_round_trip(self):
        a = self.add(1)
        b = self.add(2)
        self.store.update(a.id, state="done", watched=True, keep=True, progress={"frame": 5})
        self.store.set_settings(cap_gb=3, encoder="h264")
        self.store.move(b.id, 0)
        again = QueueStore(self.root)
        self.assertEqual([i.id for i in again.items()], [b.id, a.id])
        got = again.get(a.id)
        self.assertEqual((got.state, got.watched, got.keep, got.progress, got.headers),
                         ("done", True, True, {"frame": 5}, {"Cookie": "a=b"}))
        self.assertEqual((again.get_settings()["cap_gb"], again.get_settings()["encoder"]), (3, "h264"))
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_source_status_and_download_progress_round_trip(self):
        item = self.add(1)
        self.store.update(item.id, source_path="items/source/playlist.m3u8", source_status="downloading",
                          download_progress={"stage": "downloading", "downloaded_bytes": 512})
        again = QueueStore(self.root).get(item.id)
        self.assertEqual(again.source_status, "downloading")
        self.assertEqual(again.download_progress, {"stage": "downloading", "downloaded_bytes": 512})
        self.assertEqual(again.source_path, "items/source/playlist.m3u8")

    def test_reorder(self):
        ids = [self.add(i).id for i in range(4)]
        self.store.move(ids[3], 0)
        self.assertEqual([i.id for i in self.store.items()], [ids[3], ids[0], ids[1], ids[2]])
        self.store.move(ids[3], 99)
        self.assertEqual([i.id for i in self.store.items()], ids[:3] + [ids[3]])
        self.assertEqual([i.order for i in self.store.items()], [0, 1, 2, 3])

    def test_next_runnable_skips_other_states(self):
        a, b, c = self.add(1), self.add(2), self.add(3)
        self.store.update(a.id, state="paused")
        self.store.update(b.id, state="failed")
        self.assertEqual(self.store.next_runnable("waiting").id, c.id)
        self.assertIsNone(self.store.next_runnable("queued"))
        self.store.update(b.id, state="queued")
        self.assertEqual(self.store.next_runnable("queued").id, b.id)

    def test_items_are_copies(self):
        a = self.add(1)
        self.store.items()[0].headers["X"] = "y"
        a.state = "done"
        self.assertEqual(self.store.get(a.id).headers, {"Cookie": "a=b"})
        self.assertEqual(self.store.get(a.id).state, "waiting")

    def test_clear_headers(self):
        a = self.add(1)
        self.store.clear_headers(a.id)
        self.assertEqual(QueueStore(self.root).get(a.id).headers, {})

    def test_remove_deletes_files(self):
        a, b = self.add(1), self.add(2)
        write_file(self.store.item_dir(a.id) / "source.mp4", 100)
        write_file(self.store.item_dir(a.id) / "work" / "segments" / "0.mkv", 10)
        self.assertTrue(self.store.remove(a.id))
        self.assertFalse(self.store.item_dir(a.id).exists())
        self.assertFalse(self.store.remove(a.id))
        self.assertEqual([i.id for i in self.store.items()], [b.id])
        self.assertEqual(self.store.items()[0].order, 0)

    def test_usage_cached_until_refresh(self):
        a = self.add(1)
        write_file(self.store.item_dir(a.id) / "source.mp4", 100)
        self.assertEqual(self.store.usage_bytes(), 100)
        write_file(self.store.item_dir(a.id) / "more.bin", 50)
        self.assertEqual(self.store.usage_bytes(), 100)
        self.assertEqual(self.store.usage_bytes(refresh=True), 150)

    def _done(self, n, size, last_used, watched=True, keep=False, state="done"):
        it = self.add(n)
        write_file(self.store.item_dir(it.id) / "output.mkv", size)
        self.store.update(it.id, state=state, watched=watched, keep=keep, last_used=last_used)
        return it.id

    def test_eviction_lru_and_protection(self):
        self.store.set_settings(cap_gb=1000 / 1024 ** 3)  # 1000 bytes
        old = self._done(1, 300, 10)
        mid = self._done(2, 300, 20)
        keep = self._done(3, 300, 1, keep=True)
        unwatched = self._done(4, 300, 2, watched=False)
        proc = self._done(5, 300, 3, state="processing")
        # usage = 1500 > cap 1000: evict oldest eligible (old) first, then mid, never the others
        self.assertTrue(self.store.ensure_space(0))
        ids = {i.id for i in self.store.items()}
        self.assertTrue({keep, unwatched, proc} <= ids)
        self.assertNotIn(old, ids)
        self.assertNotIn(mid, ids)
        self.assertFalse(self.store.item_dir(old).exists())

    def test_eviction_stops_when_enough(self):
        self.store.set_settings(cap_gb=1000 / 1024 ** 3)
        old = self._done(1, 400, 10)
        mid = self._done(2, 400, 20)
        new = self._done(3, 400, 30)
        self.assertTrue(self.store.ensure_space(100))  # 1200 + 100 -> must drop 300+: only 'old'
        self.assertEqual([i.id for i in self.store.items()], [mid, new])
        self.assertEqual(self.store.usage_bytes(), 800)
        self.assertEqual(old in {i.id for i in self.store.items()}, False)

    def test_eviction_insufficient_deletes_nothing(self):
        self.store.set_settings(cap_gb=1000 / 1024 ** 3)
        a = self._done(1, 300, 10)
        b = self._done(2, 900, 5, watched=False)
        self.assertFalse(self.store.ensure_space(500))
        self.assertEqual({i.id for i in self.store.items()}, {a, b})
        self.assertTrue(self.store.item_dir(a).exists())

    def test_ensure_space_when_room(self):
        self.assertTrue(self.store.ensure_space(1024))

    def test_mark_used(self):
        a = self.add(1)
        self.store.update(a.id, last_used=1.0)
        self.store.mark_used(a.id)
        self.assertGreater(self.store.get(a.id).last_used, 1.0)

    def test_crash_state_recovery(self):
        a, b, c, d = self.add(1), self.add(2), self.add(3), self.add(4)
        self.store.update(a.id, state="downloading")
        self.store.update(b.id, state="processing")
        self.store.update(c.id, state="paused")
        self.store.update(d.id, state="done")
        again = QueueStore(self.root)
        states = {i.id: i.state for i in again.items()}
        self.assertEqual(states, {a.id: "waiting", b.id: "queued", c.id: "paused", d.id: "done"})
        # and it was persisted
        self.assertEqual(QueueStore(self.root).get(b.id).state, "queued")

    def test_unknown_and_missing_keys_tolerated(self):
        data = {"future": 1, "settings": {"cap_gb": 7, "new_option": "z"},
                "items": [{"id": "abc", "target": "t", "state": "done", "brand_new_field": [1, 2]},
                          {"id": "def", "title": "only title"}, "junk", {"no_id": 1}]}
        self.root.joinpath("queue.json").write_text(json.dumps(data), encoding="utf-8")
        s = QueueStore(self.root)
        self.assertEqual([i.id for i in s.items()], ["abc", "def"])
        self.assertEqual(s.get("abc").state, "done")
        self.assertEqual(s.get("def").state, "waiting")
        self.assertEqual(s.get("def").headers, {})
        self.assertEqual((s.get("def").source_status, s.get("def").download_progress), ("", {}))
        self.assertEqual(s.get_settings()["cap_gb"], 7)
        self.assertEqual(s.get_settings()["encoder"], "hevc")

    def test_corrupt_file_starts_fresh(self):
        self.root.joinpath("queue.json").write_text("{not json", encoding="utf-8")
        s = QueueStore(self.root)
        self.assertEqual(s.items(), [])
        self.assertTrue((self.root / "queue.json.bad").exists())

    def test_on_change_called(self):
        calls = []
        self.store.on_change.append(lambda: calls.append(1))
        self.store.on_change.append(lambda: 1 / 0)  # a broken listener must not break the store
        a = self.add(1)
        self.store.update(a.id, state="paused")
        self.store.remove(a.id)
        self.assertEqual(len(calls), 3)

    def test_update_rejects_unknown_field(self):
        a = self.add(1)
        with self.assertRaises(ValueError):
            self.store.update(a.id, bogus=1)
        with self.assertRaises(KeyError):
            self.store.update("nope", state="done")

    def test_compare_option_defaults_and_migration(self):
        s = self.store.get_settings()
        self.assertEqual((s["save_compare"], s["compare_original"], s["compare_quality"]), (False, "copy", "small"))
        old = {"version": 1, "settings": {"cap_gb": 12, "save_compare": True}, "items": []}
        self.root.joinpath("queue.json").write_text(json.dumps(old), encoding="utf-8")
        s = QueueStore(self.root).get_settings()
        self.assertEqual((s["cap_gb"], s["save_compare"], s["compare_original"], s["compare_quality"]),
                         (12, True, "copy", "small"))
        bad = {"version": 1, "settings": {"compare_original": "zip", "compare_quality": "huge"}, "items": []}
        bad["settings"]["subtitle_translation_igpu"] = True      # retired key in an old queue.json
        self.root.joinpath("queue.json").write_text(json.dumps(bad), encoding="utf-8")
        s = QueueStore(self.root).get_settings()
        self.assertEqual((s["compare_original"], s["compare_quality"]), ("copy", "small"))
        self.assertNotIn("subtitle_translation_igpu", s)

    def test_cache_key_compare_options_only_count_with_compare_on(self):
        off = cache_key("u", "best", SETTINGS)
        self.assertEqual(off, cache_key("u", "best", dict(SETTINGS, compare_original="encode", compare_quality="tiny")))
        self.assertEqual(off, cache_key("u", "best", dict(SETTINGS, compare="off", compare_original="encode")))
        on = dict(SETTINGS, save_compare=True)
        k = cache_key("u", "best", on)
        self.assertNotEqual(k, off)
        self.assertNotEqual(k, cache_key("u", "best", dict(on, compare_original="encode")))
        self.assertNotEqual(k, cache_key("u", "best", dict(on, compare_quality="tiny")))
        self.assertEqual(k, cache_key("u", "best", dict(on, compare_original="copy", compare_quality="small")))
        self.assertEqual(cache_key("u", "best", dict(on, compare_original="encode")),
                         cache_key("u", "best", dict(SETTINGS, compare="on", compare_original="encode")))

    def test_items_with_different_compare_options_are_different_outputs(self):
        a = self.add(1, save_compare=True)
        b = self.add(1, save_compare=True, compare_original="encode")
        c = self.add(1, save_compare=True)
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(a.id, c.id)
        self.assertEqual(self.add(2).id, self.add(2, compare_original="encode").id)

    def test_seen_defaults_false_and_legacy_load_is_true(self):
        a = self.add(1)
        self.assertFalse(a.seen)
        self.assertIn("seen", a.to_dict())
        legacy = json.loads(self.root.joinpath("queue.json").read_text(encoding="utf-8"))
        for raw in legacy["items"]:
            raw.pop("seen")
        self.root.joinpath("queue.json").write_text(json.dumps(legacy), encoding="utf-8")
        self.assertTrue(QueueStore(self.root).get(a.id).seen)
        self.store.mark_seen(a.id)                                   # with the field present it round-trips
        self.assertTrue(QueueStore(self.root).get(a.id).seen)

    def test_legacy_transient_subtitle_status_loads_as_valid_state(self):
        from oh_streamer.queue.queue_store import SUBTITLE_STATES
        ids = {}
        for n, status in enumerate(("combos", "asr", "translate")):
            item = self.add(n)
            ids[status] = item.id
            self.store.update(item.id, subtitle={"status": status, "percent": 40, "phase": "x", "detail": "y"})
        reloaded = QueueStore(self.root)
        for status, item_id in ids.items():
            self.assertIn(reloaded.get(item_id).subtitle["status"], SUBTITLE_STATES, status)
            self.assertEqual(reloaded.get(item_id).subtitle["status"], "queued", status)
        self.assertNotIn("combos", SUBTITLE_STATES)

    def test_mark_seen_notifies_only_on_change(self):
        a, b = self.add(1), self.add(2)
        calls = []
        self.store.on_change.append(lambda: calls.append(1))
        self.assertTrue(self.store.mark_seen(a.id))
        self.assertFalse(self.store.mark_seen(a.id))
        self.assertFalse(self.store.mark_seen("nope"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.store.mark_all_seen(), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.store.mark_all_seen(), 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(self.store.get(b.id).seen)

    def add_stream(self, n=1, **kw):
        return self.store.add_stream(f"https://x/s{n}", "url", {"Cookie": "a=b"}, dict(SETTINGS, **kw), title=f"s{n}",
                                     stream={"audio_url": f"https://x/a{n}"})

    def test_stream_item_create_and_hidden_from_queue(self):
        reg = self.add(1)
        s = self.add_stream()
        self.assertEqual((s.kind, s.state, s.title, s.headers), ("stream", "streaming", "s1", {"Cookie": "a=b"}))
        self.assertTrue(s.is_stream)
        self.assertFalse(reg.is_stream)
        self.assertEqual(s.stream, {"audio_url": "https://x/a1", "source_kind": "url"})
        self.assertEqual(s.cache_key, "")
        self.assertEqual([i.id for i in self.store.items()], [reg.id])
        self.assertEqual([i.id for i in self.store.stream_items()], [s.id])
        self.assertEqual(self.store.get(s.id).id, s.id)
        self.assertEqual(self.store.item_dir(s.id), self.root / "streams" / s.id)
        self.assertEqual(self.store.item_dir(reg.id), self.root / "items" / reg.id)
        self.assertTrue(self.store.item_dir(s.id).is_dir())
        self.assertEqual(self.add_stream().target, s.target)           # no dedupe: each stream is its own item
        self.assertEqual(len(self.store.stream_items()), 2)

    def test_stream_item_subtitle_block_only_with_subtitles(self):
        self.assertEqual(self.add_stream().subtitle, {})
        self.store.set_settings(subtitle_runtime_contract="legacy")   # a fresh store's cloud profile is unconfigured
        s = self.add_stream(2, subtitles=True)
        self.assertEqual(s.subtitle["status"], "queued")
        self.assertTrue(s.settings["subtitles"])                    # validated, not frozen: follows the global settings
        self.assertFalse([k for k in s.settings if k.startswith("subtitle_")])

    def test_stream_item_ignored_by_scheduling_and_capacity(self):
        s = self.add_stream()
        self.store.update(s.id, state="queued")                       # even if it looked runnable
        self.assertIsNone(self.store.next_runnable("queued"))
        self.assertIsNone(self.store.next_runnable("waiting"))
        write_file(self.store.item_dir(s.id) / "audio.wav", 4096)
        self.assertEqual(self.store.usage_bytes(refresh=True), 0)
        self.store.set_settings(cap_gb=0.000001)                      # ~1 KiB cap: only regular items could be evicted
        self.assertFalse(self.store.ensure_space(10_000))
        self.assertEqual(len(self.store.stream_items()), 1)
        self.assertFalse(self.store.mark_seen(s.id))
        self.assertEqual(self.store.mark_all_seen(), 0)
        with self.assertRaises(KeyError):
            self.store.move(s.id, 0)
        self.assertFalse(self.store.remove(s.id))                     # regular remove never touches streams
        self.assertEqual(len(self.store.stream_items()), 1)

    def test_stream_item_update_is_memory_only_and_keeps_kind(self):
        self.store.set_settings(subtitle_runtime_contract="legacy")
        s = self.add_stream(2, subtitles=True)
        before = (self.root / "queue.json").read_bytes()
        calls = []
        self.store.on_change.append(lambda: calls.append(1))
        got = self.store.update(s.id, subtitle={"status": "asr", "percent": 10})
        self.assertEqual(got.subtitle["status"], "asr")
        self.assertEqual(self.store.get(s.id).subtitle["percent"], 10)
        self.assertEqual(calls, [1])                                  # the UI still hears about it
        self.assertEqual((self.root / "queue.json").read_bytes(), before)
        with self.assertRaises(ValueError):
            self.store.update(s.id, kind="url")
        self.assertNotIn(s.id, (self.root / "queue.json").read_text(encoding="utf-8"))

    def test_stream_item_remove_deletes_folder(self):
        s, t = self.add_stream(1), self.add_stream(2)
        write_file(self.store.item_dir(s.id) / "sub" / "a.srt", 10)
        calls = []
        self.store.on_change.append(lambda: calls.append(1))
        self.assertTrue(self.store.remove_stream(s.id))
        self.assertFalse((self.root / "streams" / s.id).exists())
        self.assertIsNone(self.store.get(s.id))
        self.assertFalse(self.store.remove_stream(s.id))
        self.assertFalse(self.store.remove_stream(self.add(1).id))    # a regular id is not a stream item
        self.assertEqual(len(self.store.items()), 1)
        self.assertEqual(calls, [1, 1])                               # remove_stream + the add() above
        self.assertTrue((self.root / "streams" / t.id).is_dir())
        self.assertEqual(self.store.clear_streams(), 1)
        self.assertEqual(self.store.stream_items(), [])
        self.assertFalse((self.root / "streams").exists())

    def test_stream_leftovers_cleared_on_load(self):
        reg = self.add(1)
        s = self.add_stream()
        write_file(self.store.item_dir(s.id) / "audio.wav", 100)
        write_file(self.root / "streams" / "orphan" / "x.srt", 10)    # a crashed run's residue, unknown to memory
        write_file(self.store.item_dir(reg.id) / "source.mp4", 100)
        again = QueueStore(self.root)                                 # a restart
        self.assertEqual(again.stream_items(), [])
        self.assertFalse((self.root / "streams").exists())
        self.assertIsNone(again.get(s.id))
        self.assertEqual([i.id for i in again.items()], [reg.id])
        self.assertTrue((self.root / "items" / reg.id / "source.mp4").exists())   # regular item data untouched

    def test_stream_kind_in_queue_json_is_dropped_on_load(self):
        reg = self.add(1)
        data = json.loads((self.root / "queue.json").read_text(encoding="utf-8"))
        data["items"].append({"id": "ghost001", "kind": "stream", "target": "u", "order": 1})
        (self.root / "queue.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual([i.id for i in QueueStore(self.root).items()], [reg.id])


if __name__ == "__main__":
    unittest.main()
