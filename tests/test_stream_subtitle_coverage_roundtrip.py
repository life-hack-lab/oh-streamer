"""P3 coverage contract using real SRT publication, queue persistence and player parsers (CPU only)."""
from pathlib import Path
from unittest import mock

from oh_streamer.player import progressive_stream as ps, queue_bridge as qb
from oh_streamer.queue import queue_store, subtitle_host as sh
from oh_streamer.subtitles import slice_plan as sp, slice_results as sr
from tests.test_queue_service import ServiceTestBase


class CoverageRoundtripTests(ServiceTestBase):
    def test_real_publication_survives_store_and_both_player_snapshot_paths(self):
        item_id = self.ready()
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        self.svc._subtitle_update(item_id, status="queued", audio_offset=2.0)
        plan = sp.new_slice_plan(3600.0, media_id="coverage-roundtrip")
        req = sp.seek_cut_request(plan, 2400.0)
        self.assertTrue(sp.apply_seek_cuts(plan, req, [], []))
        segment = plan.slices[0]
        result = sr.publish(folder, plan, {segment.index: [
            {"start": 1.0, "end": 3.0, "text": "original", "zh": "translated"}
        ]}, japanese=True)
        self.assertIsInstance(result["srt"], dict)   # real producer shape, unlike old fake IPC tests
        self.assertEqual(result["covered_s"], 0.0)
        expected = [[segment.start, segment.end]]
        self.assertEqual(result["covered_ranges"], expected)
        self.player_reply = {"ok": True}
        host = sh.QueueSubtitleHost(self.svc, item_id, probe_offset=False)
        with mock.patch.object(self.svc, "_maybe_play_after_subtitles"):
            host.publish(result)
        item = queue_store.QueueStore(self.svc.store.root).get(item_id)
        self.assertEqual(item.subtitle["covered_ranges"], expected)
        self.assertEqual(item.subtitle["audio_offset"], 2.0)
        sent = [p for p in self.players if p.get("cmd") == "subtitle_update"]
        self.assertEqual(len(sent), 1)
        update = qb.subtitle_update_from_message(sent[0])
        self.assertIsNotNone(update)
        self.assertEqual(update.covered_ranges, expected)
        self.assertEqual(update.subtitle_offset, 2.0)
        self.assertEqual([t["path"] for t in update.subtitles], [t["path"] for t in result["subtitles"]])
        # Initial open must carry the same discontinuous coverage after reopening a queue item.
        payload = dict(progressive=True, input=item.source_path, work_dir=str(folder), queue_id=item_id)
        self.svc._add_external_subtitles(payload, item)
        opened = ps.progressive_from_message(payload, probe=lambda _: {
            "duration": 3600.0, "fps": 25, "video_start": 0, "frames": 90000, "has_audio": True})
        self.assertIsNotNone(opened)
        self.assertEqual(opened.subtitle_covered_ranges, expected)
        self.assertEqual(opened.subtitle_offset, update.subtitle_offset)
        self.assertFalse(opened.subtitle_complete)
        # Range growth alone matters even when the continuous prefix / path / publication key is unchanged.
        grown = {**result, "covered_ranges": [[segment.start, segment.end + 10.0]]}
        self.svc._subtitle_publish_update(item, grown)
        sent = [p for p in self.players if p.get("cmd") == "subtitle_update"]
        self.assertEqual(len(sent), 2)
        self.assertEqual(qb.subtitle_update_from_message(sent[-1]).covered_ranges, grown["covered_ranges"])

    def test_rebuilding_a_slice_table_invalidates_its_old_coverage_ranges(self):
        from oh_streamer.queue.subtitle_recovery import without_legacy
        original = {"status": "partial", "covered_s": 30.0, "covered_ranges": [[0, 30], [2400, 2430]],
                    "block_srt": ["old.srt"], "block_subtitles": [], "audio_offset": 2.0}
        self.assertEqual(without_legacy(original, publish=True), {"status": "partial"})
        self.assertEqual(without_legacy(original)["covered_ranges"], original["covered_ranges"])

    def test_language_map_without_track_metadata_is_still_a_valid_dynamic_snapshot(self):
        item_id = self.ready()
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        paths = {lang: str(folder / f"{lang}.srt") for lang in ("zh-TW", "full", "ja")}
        for path in paths.values():
            Path(path).write_text("", encoding="utf-8")
        self.player_reply = {"ok": True}
        self.svc._subtitle_publish_update(self.svc.store.get(item_id), {
            "srt": paths, "covered_s": 0, "covered_ranges": [[2400, 2430]], "complete": False})
        sent = [p for p in self.players if p.get("cmd") == "subtitle_update"]
        self.assertEqual(len(sent), 1)
        update = qb.subtitle_update_from_message(sent[0])
        self.assertIsNotNone(update)
        self.assertEqual([t["path"] for t in update.subtitles], [paths["zh-TW"], paths["ja"]])
