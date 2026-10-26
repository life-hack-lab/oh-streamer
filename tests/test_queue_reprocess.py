"""Reprocessing version and source ownership tests; no GPU or network."""
import subprocess
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from test_queue_service import ServiceTestBase
from oh_streamer.queue.queue_store import QueueStore


class ReprocessTests(ServiceTestBase):
    def finished(self, mode="copy"):
        item_id = self.ready(state="done")
        out = self.svc.store.item_dir(item_id) / "output.mkv"
        out.write_bytes(b"old-output")
        self.svc.store.update(item_id, source_path="", output_path=str(out),
                              settings={"compare": "on", "compare_original": "copy"})
        info = {"streams": [{"index": 0, "codec_type": "video", "title": "oh:processed"},
                            {"index": 2, "codec_type": "video", "title": "oh:original"},
                            {"index": 1, "codec_type": "audio"}],
                "tags": {"OH_ORIGINAL": mode}, "duration": 8.0}
        return item_id, out, info

    def test_cached_source_is_independent_and_settings_stay_pinned(self):
        old_id = self.ready(state="processing", headers={"Referer": "https://example.test"})
        old = self.svc.store.get(old_id)
        result = self.svc.reprocess(old_id, {"encoder": "h264", "save_compare": True,
                                            "compare_original": "encode", "compare_quality": "tiny"})
        self.assertTrue(result["ok"], result)
        new = self.svc.store.get(result["id"])
        self.assertNotEqual(old_id, new.id)
        self.assertEqual(new.parent_id, old_id)
        self.assertEqual(new.target, old.target)
        self.assertEqual(new.headers, old.headers)
        self.assertEqual(new.state, "queued")
        self.assertNotEqual(new.source_path, old.source_path)
        Path(old.source_path).unlink()
        self.assertEqual(Path(new.source_path).read_bytes(), b"s" * 100)
        self.svc.store.set_settings(encoder="hevc", save_compare=False, compare_original="copy")
        job = self.svc._build_job(new)
        self.assertEqual(job["settings"]["encoder"], "h264")
        self.assertEqual(job["settings"]["compare"], "on")
        self.assertEqual(job["settings"]["compare_original"], "encode")
        self.assertEqual(job["settings"]["compare_quality"], "tiny")
        loaded = QueueStore(self.svc.store.root).get(new.id)
        self.assertEqual(loaded.parent_id, old_id)

    def test_same_settings_still_create_new_version(self):
        old_id = self.ready(state="done")
        result = self.svc.reprocess(old_id, {})
        self.assertTrue(result["ok"], result)
        self.assertNotEqual(result["id"], old_id)

    def test_waiting_item_edits_in_place(self):
        self.svc.pause_all()
        old_id = self.add("https://example.test/video")
        result = self.svc.reprocess(old_id, {"encoder": "h264"})
        self.assertEqual(result["id"], old_id)
        self.assertEqual(self.svc.store.get(old_id).settings["pinned"]["encoder"], "h264")
        self.assertEqual(len(self.svc.store.items()), 1)

    def test_stop_current_only_after_success(self):
        old_id = self.ready(state="processing")
        result = self.svc.reprocess(old_id, {}, {"stop_current": True})
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.state(old_id), "paused")
        self.assertEqual(self.state(result["id"]), "queued")

    def test_failed_preparation_keeps_old_and_discards_new(self):
        old_id = self.ready(state="processing")
        with patch("oh_streamer.queue.queue_service.shutil.copy2", side_effect=OSError("copy failed")):
            result = self.svc.reprocess(old_id, {}, {"stop_current": True})
        self.assertFalse(result["ok"])
        self.assertEqual(self.state(old_id), "processing")
        self.assertEqual(len(self.svc.store.items()), 1)
        self.assertFalse(self.svc._reprocessing)

    def test_quality_change_or_redownload_does_not_reuse_source(self):
        old_id = self.ready(state="done")
        for changes, options in (({"quality": 1080}, {}), ({}, {"redownload": True})):
            result = self.svc.reprocess(old_id, changes, options)
            new = self.svc.store.get(result["id"])
            self.assertEqual(new.state, "waiting")
            self.assertEqual(new.source_path, "")

    def test_compare_uses_original_track_and_effective_encode_mode(self):
        old_id, out, info = self.finished(mode="encode")
        commands = []
        def remux(cmd):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b"extracted-original")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        self.svc._run_cmd = remux
        self.svc._probe_tracks = lambda path: info if path == str(out) else {
            "streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "duration": 8.0}
        view = self.svc.reprocess_options(old_id)
        self.assertTrue(view["source"]["lossy"])
        result = self.svc.reprocess(old_id, {})
        self.assertTrue(result["ok"], result)
        new = self.svc.store.get(result["id"])
        self.assertEqual(new.source_origin, "compare_encode")
        self.assertIn("0:2", commands[0])
        self.assertIn("0:a?", commands[0])
        self.assertEqual(out.read_bytes(), b"old-output")

    def test_compare_copy_and_duration_validation(self):
        old_id, out, info = self.finished()
        self.svc._probe_tracks = lambda path: info if path == str(out) else {
            "streams": [{"codec_type": "video"}], "duration": 2.0}
        def remux(cmd):
            Path(cmd[-1]).write_bytes(b"incomplete")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        self.svc._run_cmd = remux
        self.assertFalse(self.svc.reprocess_options(old_id)["source"]["lossy"])
        self.assertFalse(self.svc.reprocess(old_id, {})["ok"])
        self.assertEqual(len(self.svc.store.items()), 1)
        self.assertEqual(out.read_bytes(), b"old-output")

    def test_processed_track_never_used_as_original(self):
        old_id, out, info = self.finished()
        info["streams"] = info["streams"][:1]
        self.svc._probe_tracks = lambda path: info
        self.assertEqual(self.svc.reprocess_options(old_id)["source"]["mode"], "download")
        result = self.svc.reprocess(old_id, {})
        self.assertEqual(self.state(result["id"]), "waiting")

    def test_missing_local_file_can_be_relocated(self):
        self.svc.pause_all()
        old_id = self.add(str(self.tmp / "missing.mp4"), kind="file")
        self.svc.store.update(old_id, state="done")
        self.assertFalse(self.svc.reprocess(old_id, {})["ok"])
        replacement = self.tmp / "relocated.mp4"
        replacement.write_bytes(b"original")
        result = self.svc.reprocess(old_id, {}, {"source_path": str(replacement)})
        self.assertTrue(result["ok"], result)
        new = self.svc.store.get(result["id"])
        self.assertEqual(new.target, str(replacement.resolve()))
        self.assertEqual(new.source_path, new.target)
        self.assertTrue(replacement.is_file())

    def test_invalid_settings_do_not_change_old(self):
        old_id = self.ready(state="done")
        before = self.svc.store.get(old_id).to_dict()
        for changes in ({"encoder": "bogus"}, {"quality": -1}, {"quality": True},
                        {"save_compare": "yes"}, {"detection_model": "not-installed"}):
            self.assertFalse(self.svc.reprocess(old_id, changes)["ok"])
        self.assertEqual(self.svc.store.get(old_id).to_dict(), before)
        self.assertEqual(len(self.svc.store.items()), 1)

    def test_space_limit_preserves_old_version(self):
        old_id = self.ready(state="done")
        self.svc.store.update(old_id, watched=True)
        self.svc.store.set_settings(cap_gb=0)
        result = self.svc.reprocess(old_id, {})
        self.assertFalse(result["ok"])
        self.assertIsNotNone(self.svc.store.get(old_id))
        self.assertEqual(len(self.svc.store.items()), 1)

    def test_source_cannot_be_deleted_during_preparation(self):
        old_id = self.ready(state="done")
        copy = shutil.copy2
        def copying(src, dest):
            self.assertFalse(self.svc.remove(old_id)["ok"])
            self.svc._watched_ids()
            self.assertIn(old_id, self.svc.store.protected_ids)
            return copy(src, dest)
        with patch("oh_streamer.queue.queue_service.shutil.copy2", side_effect=copying):
            self.assertTrue(self.svc.reprocess(old_id, {})["ok"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg/ffprobe")
    def test_real_compare_extraction_keeps_original_pixels_and_audio(self):
        old_id, out, _info = self.finished()
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=32x32:r=5:d=1",
               "-f", "lavfi", "-i", "color=blue:s=32x32:r=5:d=1",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
               "-map", "0:v", "-map", "1:v", "-map", "2:a", "-c:v", "ffv1", "-c:a", "pcm_s16le",
               "-metadata:s:v:0", "title=oh:processed", "-metadata:s:v:1", "title=oh:original",
               "-metadata", "OH_ORIGINAL=copy", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        before = out.read_bytes()
        result = self.svc.reprocess(old_id, {})
        self.assertTrue(result["ok"], result)
        new = self.svc.store.get(result["id"])
        def pixels(path, track):
            return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", track,
                                   "-f", "framemd5", "-"], check=True, capture_output=True).stdout
        self.assertEqual(pixels(out, "0:v:1"), pixels(new.source_path, "0:v:0"))
        tracks = self.svc._probe_tracks(new.source_path)["streams"]
        self.assertEqual([t["codec_type"] for t in tracks], ["video", "audio"])
        self.assertEqual(new.source_origin, "compare_copy")
        self.assertEqual(out.read_bytes(), before)
