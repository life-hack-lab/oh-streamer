"""Pure-logic checks for player/subtitle_compare.py (no mpv, no GPU)."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oh_streamer.player import subtitle_compare as sc  # noqa: E402

SRT = "\ufeff1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>你好</i>\r\n世界\r\n\r\n2\r\n00:01:00.250 --> 00:01:02.000\r\n第二句\r\n"


EXTRA_CUE = chr(10) + "3" + chr(10) + "00:02:00,000 --> 00:02:01,000" + chr(10) + "第三句" + chr(10)


def manifest(combos, **extra):
    data = {"format": sc.MANIFEST_FORMAT, "schema_version": 1, "video": {}, "combos": combos, "mux": None}
    data.update(extra)
    return data


def entry(cid, label=None, status="done", zh=None, ja=None):
    return {"id": cid, "label": label or cid, "status": status,
            "files": {"zh": zh if zh is not None else f"{cid}.zh-TW.srt", "ja": ja}, "cue_count": 2}


class TextHelperTests(unittest.TestCase):
    def test_sanitize_label(self):
        self.assertEqual(sc.sanitize_label("  a\tb\n c  "), "a b c")
        self.assertEqual(sc.sanitize_label("x{\\an8}y"), "x(／an8)y")
        self.assertEqual(sc.sanitize_label(None, "fallback"), "fallback")
        self.assertEqual(sc.sanitize_label("\n\x00 ", "id{1}"), "id(1)")
        long = sc.sanitize_label("字" * 100)
        self.assertEqual(len(long), sc.MAX_LABEL_CHARS)
        self.assertTrue(long.endswith("…"))
        self.assertEqual(sc.sanitize_label("anime-whisper＋Sakura v1.0"), "anime-whisper＋Sakura v1.0")

    def test_escape_ass_text(self):
        self.assertEqual(sc.escape_ass_text("a\nb"), "a\\Nb")
        self.assertEqual(sc.escape_ass_text("a\r\nb\rc"), "a\\Nb\\Nc")
        self.assertEqual(sc.escape_ass_text("{\\an8}hi"), "\\{＼an8\\}hi")
        self.assertNotIn("\\N", sc.escape_ass_text("literal \\N here").replace("\\N", "", 0).split("literal")[1][:0] or "")
        self.assertEqual(sc.escape_ass_text("back\\Nslash"), "back＼Nslash")
        self.assertEqual(sc.escape_ass_text("  x \n\n y  "), "x\\N\\Ny")
        self.assertEqual(sc.escape_ass_text(None), "")
        self.assertIn(",", sc.escape_ass_text("a,b,c"))   # commas are fine in the last field

    def test_time_and_colour(self):
        self.assertEqual(sc.format_ass_time(0), "0:00:00.00")
        self.assertEqual(sc.format_ass_time(3661.456), "1:01:01.46")
        self.assertEqual(sc.format_ass_time(-3), "0:00:00.00")
        self.assertEqual(sc.format_ass_time(59.999), "0:01:00.00")
        self.assertEqual(sc.ass_color((255, 0, 0)), "&H000000FF")     # BGR order
        self.assertEqual(sc.ass_color((1, 2, 3), alpha=0x80), "&H80030201")
        self.assertEqual(sc.ass_color((300, -5, 0)), "&H000000FF")
        dim = sc.dim_color((255, 255, 255))
        self.assertTrue(all(150 < c < 255 for c in dim))
        self.assertEqual(sc.dim_color((10, 20, 30), 0), (10, 20, 30))

    def test_palette_distinct_and_wraps(self):
        colours = [sc.combo_color(i)[1] for i in range(len(sc.PALETTE))]
        self.assertEqual(len(set(colours)), len(colours))
        self.assertEqual(sc.combo_color(len(sc.PALETTE)), sc.combo_color(0))

    def test_style_names_are_unique_ascii_without_commas(self):
        names = [sc.safe_style_name(x, i) for i, x in enumerate(["a b", "a,b", "字幕", "a b"])]
        self.assertEqual(len(set(names)), 4)
        for n in names:
            self.assertRegex(n, r"^[A-Za-z0-9_.\-]+$")


class SrtTests(unittest.TestCase):
    def test_parse_basic(self):
        cues = sc.parse_srt(SRT)
        self.assertEqual([(c.start, c.end, c.text) for c in cues],
                         [(1.0, 2.5, "你好\n世界"), (60.25, 62.0, "第二句")])

    def test_tolerates_garbage(self):
        text = "x\n\n3\n00:00:05,000 --> 00:00:04,000\nbackwards\n\n00:00:01,5 --> 00:00:02,0\nno index\n\n4\nbad --> time\ntext\n\n5\n00:00:03,000 --> 00:00:04,000\n   \n"
        cues = sc.parse_srt(text)
        self.assertEqual([(c.start, c.end, c.text) for c in cues], [(1.5, 2.0, "no index")])

    def test_sorted_and_empty(self):
        self.assertEqual(sc.parse_srt(""), [])
        self.assertEqual(sc.parse_srt(None), [])
        cues = sc.parse_srt("1\n00:00:09,000 --> 00:00:10,000\nb\n\n2\n00:00:01,000 --> 00:00:02,000\na\n")
        self.assertEqual([c.text for c in cues], ["a", "b"])


class LayoutTests(unittest.TestCase):
    def test_bands_stack_bottom_up_with_own_layers(self):
        for n in range(1, sc.MAX_COMBOS + 1):
            bands = sc.layout_bands(n)
            self.assertEqual(len(bands), n)
            margins = [b.margin_v for b in bands]
            self.assertEqual(margins, sorted(margins))
            self.assertEqual(len(set(margins)), n)
            self.assertEqual(len({b.layer for b in bands}), n)
            top = bands[-1].margin_v + bands[-1].font_size * 1.2
            self.assertLessEqual(top, sc.PLAY_RES[1] * 0.92, n)
            self.assertGreaterEqual(min(b.font_size for b in bands), 16)

    def test_bands_edge_counts(self):
        self.assertEqual(sc.layout_bands(0), [])
        self.assertEqual(len(sc.layout_bands(50)), sc.MAX_COMBOS)
        self.assertGreater(sc.layout_bands(1)[0].font_size, sc.layout_bands(8)[0].font_size)


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.tracks = [("a", "組合A", sc.parse_srt(SRT)), ("b", "組合B{x}", sc.parse_srt(SRT)[:1])]
        self.ass = sc.build_composite_ass(self.tracks)
        self.lines = self.ass.splitlines()

    def test_sections_and_styles(self):
        for header in ("[Script Info]", "[V4+ Styles]", "[Events]"):
            self.assertIn(header, self.lines)
        styles = [l for l in self.lines if l.startswith("Style:")]
        self.assertEqual(len(styles), 2)
        fmt = next(l for l in self.lines if l.startswith("Format:") and "Fontname" in l)
        self.assertEqual(len(fmt.split(",")), len(styles[0].split(",")))
        self.assertIn("PlayResY: 720", self.ass)
        margins = [int(s.split(",")[-2]) for s in styles]
        self.assertLess(margins[0], margins[1])       # first combo is the lowest band
        self.assertNotEqual(styles[0].split(",")[3], styles[1].split(",")[3])   # distinct colours

    def test_dialogue_lines(self):
        dlg = [l for l in self.lines if l.startswith("Dialogue:")]
        self.assertEqual(len(dlg), 3)
        starts = [l.split(",")[1] for l in dlg]
        self.assertEqual(starts, sorted(starts, key=lambda s: (len(s), s)))
        for l in dlg:
            parts = l.split(",", 9)
            self.assertEqual(len(parts), 10)
            self.assertEqual(parts[3], parts[4])            # Style == Name, used by Lua to hide a band
        first = dlg[0]
        self.assertIn("【組合A】", first)
        self.assertTrue(first.endswith("你好\\N世界"))
        self.assertIn("【組合B(x)】", "\n".join(dlg))     # braces in labels never reach ASS
        self.assertEqual({l.split(",")[0] for l in dlg}, {"Dialogue: 0", "Dialogue: 1"})

    def test_label_colour_dimmer_than_text(self):
        dlg = next(l for l in self.lines if l.startswith("Dialogue:"))
        m = re.search(r"\\1c&H([0-9A-F]{6})&", dlg)
        self.assertIsNotNone(m)
        style = next(l for l in self.lines if l.startswith("Style:"))
        self.assertNotEqual("&H00" + m.group(1), style.split(",")[3])
        self.assertIn("{\\r}", dlg)

    def test_no_override_injection_from_text(self):
        ass = sc.build_composite_ass([("a", "A", [sc.Cue(0, 1, "{\\fs200}big")])])
        body = next(l for l in ass.splitlines() if l.startswith("Dialogue:")).split(",", 9)[9]
        self.assertNotIn("{\\fs200}", body)
        self.assertIn("\\{", body)

    def test_empty_and_capped(self):
        self.assertIn("[Events]", sc.build_composite_ass([]))
        many = [(f"c{i}", f"L{i}", sc.parse_srt(SRT)) for i in range(20)]
        ass = sc.build_composite_ass(many)
        self.assertEqual(len([l for l in ass.splitlines() if l.startswith("Style:")]), sc.MAX_COMBOS)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "clip.01.mkv"
        self.video.write_bytes(b"x" * 10)
        self.folder = self.root / "clip.01.subtitle-combos"
        self.folder.mkdir()

    def write(self, data, srts=("a", "b")):
        for name in srts:
            (self.folder / f"{name}.zh-TW.srt").write_text(SRT, encoding="utf-8")
        (self.folder / "manifest.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_folder_rule(self):
        self.assertEqual(sc.default_folder(self.video), self.folder)
        self.assertIsNone(sc.find_manifest(self.video))
        self.assertIsNone(sc.load(self.video))

    def test_queue_item_folder_resolution(self):
        item = self.root / "items" / "q1"
        qfolder = item / "subtitles" / "combos"
        qfolder.mkdir(parents=True)
        out = item / "output.mkv"
        out.write_bytes(b"x")
        self.assertIsNone(sc.queue_item_folder(out))              # no manifest yet
        self.assertEqual(sc.resolve_folder(out), sc.default_folder(out))
        (qfolder / "a.zh-TW.srt").write_text(SRT, encoding="utf-8")
        (qfolder / "manifest.json").write_text(json.dumps(manifest([entry("a")])), encoding="utf-8")
        self.assertEqual(sc.queue_item_folder(out), qfolder)
        self.assertEqual(sc.find_manifest(out), qfolder / "manifest.json")
        self.assertEqual([c.id for c in sc.load(out).combos], ["a"])
        # an explicit folder wins; one without manifest is ignored; other file names keep the sibling rule
        other = item / "x.mkv"
        other.write_bytes(b"x")
        self.assertIsNone(sc.find_manifest(other))
        self.assertEqual(sc.find_manifest(other, qfolder), qfolder / "manifest.json")
        self.assertIsNone(sc.find_manifest(other, self.root / "empty"))
        self.assertEqual(sc.prepare(out, self.root / "work").combos[0]["id"], "a")
        # the sibling folder keeps working for ordinary videos (self.video / self.folder of this class)
        self.write(manifest([entry("a")]), srts=("a",))
        self.assertEqual(sc.find_manifest(self.video), self.folder / "manifest.json")

    def test_load_keeps_done_with_existing_files_in_order(self):
        self.write(manifest([entry("b", "B"), entry("a", "A"), entry("c", status="failed"),
                             entry("d", "D", status="done")]), srts=("a", "b"))
        found = sc.load(self.video)
        self.assertEqual([c.id for c in found.combos], ["b", "a"])
        self.assertEqual(found.combos[0].label, "B")
        self.assertEqual(found.combos[0].zh_path, self.folder / "b.zh-TW.srt")
        self.assertTrue(any(s.startswith("c:") for s in found.skipped))
        self.assertTrue(any("d:" in s and "missing" in s for s in found.skipped))

    def test_rejects_bad_documents(self):
        for data in ({"format": "other", "schema_version": 1, "combos": []},
                     manifest([], schema_version=2),
                     manifest([], schema_version="1"),
                     manifest([], schema_version=True),
                     {"format": sc.MANIFEST_FORMAT, "schema_version": 1, "combos": {}}):
            with self.assertRaises(sc.ManifestError):
                sc.parse_manifest(data, self.folder)

    def test_load_never_raises(self):
        (self.folder / "manifest.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(sc.load(self.video))
        self.write(manifest([], schema_version=9))
        self.assertIsNone(sc.load(self.video))
        self.write(manifest([entry("a", status="failed")]))
        self.assertIsNone(sc.load(self.video))

    def test_paths_cannot_leave_the_folder(self):
        (self.root / "evil.srt").write_text(SRT, encoding="utf-8")
        found = sc.parse_manifest(manifest([entry("a", zh="../evil.srt"), entry("b", zh=str(self.root / "evil.srt")),
                                            entry("c", zh="sub/x.srt"), entry("d", zh=None)]), self.folder)
        self.assertEqual(found.combos, ())

    def test_duplicate_ids_and_bad_entries(self):
        self.write(manifest([entry("a"), entry("a"), "junk", {"label": "no id"}, entry("b")]))
        found = sc.load(self.video)
        self.assertEqual([c.id for c in found.combos], ["a", "b"])

    def test_missing_ja_is_none_and_label_fallback(self):
        e = entry("a", ja="a.ja.srt")
        e["label"] = ""
        self.write(manifest([e]))
        c = sc.load(self.video).combos[0]
        self.assertIsNone(c.ja_path)
        self.assertEqual(c.label, "a")

    def test_running_is_accepted_only_with_zh_file(self):
        self.write(manifest([entry("a", status="running"), entry("b", status="running"), entry("c", status="failed")],
                            managed_by="slice_scheduler", updated_at="2026-10-21T01:00:00"), srts=("a", "c"))
        found = sc.load(self.video)
        self.assertEqual([c.id for c in found.combos], ["a"])
        self.assertTrue(any(x.startswith("b:") and "missing" in x for x in found.skipped))
        self.assertTrue(any(x.startswith("c:") for x in found.skipped))

    def test_scheduler_video_block_without_size_is_not_a_change(self):
        self.write(manifest([entry("a", status="running")], video={"path": str(self.video), "media_id": "m1"}), srts=("a",))
        self.assertFalse(sc.load(self.video).video_changed)

    def test_fingerprint_tracks_progress(self):
        self.assertIsNone(sc.manifest_fingerprint(self.video))
        self.write(manifest([entry("a", status="running")], updated_at="t1"), srts=("a",))
        f1 = sc.manifest_fingerprint(self.video)
        self.assertEqual(f1, sc.manifest_fingerprint(self.video))
        e = entry("a", status="running")
        e["cue_count"] = 7
        self.write(manifest([e], updated_at="t1"), srts=("a",))
        self.assertNotEqual(f1, sc.manifest_fingerprint(self.video))
        self.write(manifest([entry("a", status="done")], updated_at="t1"), srts=("a",))
        self.assertNotEqual(f1, sc.manifest_fingerprint(self.video))
        self.write(manifest([entry("a", status="running")], updated_at="t2"), srts=("a",))
        self.assertNotEqual(f1, sc.manifest_fingerprint(self.video))
        (self.folder / "manifest.json").write_text("{bad", encoding="utf-8")
        self.assertIsNone(sc.manifest_fingerprint(self.video))

    def test_legacy_done_manifest_unchanged(self):
        self.write(manifest([entry("a"), entry("b", status="queued")]), srts=("a", "b"))
        found = sc.load(self.video)
        self.assertEqual([c.id for c in found.combos], ["a"])
        self.assertTrue(any(x.startswith("b:") for x in found.skipped))

    def test_video_changed_flag(self):
        st = self.video.stat()
        self.write(manifest([entry("a")], video={"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}))
        self.assertFalse(sc.load(self.video).video_changed)
        self.write(manifest([entry("a")], video={"size_bytes": st.st_size + 1, "mtime_ns": st.st_mtime_ns}))
        self.assertTrue(sc.load(self.video).video_changed)

    def test_prepare_writes_composite_and_json(self):
        self.write(manifest([entry("a", "A"), entry("b", "B")]))
        work = self.root / "work"
        prep = sc.prepare(self.video, work)
        self.assertEqual(prep.composite, work / "compare.ass")
        text = prep.composite.read_text(encoding="utf-8")
        self.assertEqual(len([l for l in text.splitlines() if l.startswith("Dialogue:")]), 4)
        self.assertEqual([r["id"] for r in prep.combos], ["a", "b"])
        styles = {l.split(",")[0][len("Style: "):] for l in text.splitlines() if l.startswith("Style:")}
        self.assertEqual({r["style"] for r in prep.combos}, styles)
        payload = json.loads(sc.publish_json(prep))
        self.assertEqual(payload["title"], sc.COMPOSITE_TITLE)
        self.assertEqual(payload["combos"][0]["file"], str(self.folder / "a.zh-TW.srt"))
        self.assertRegex(payload["combos"][0]["rgb"], r"^[0-9A-F]{6}$")

    def test_prepare_none_without_manifest(self):
        self.assertIsNone(sc.prepare(self.video, self.root / "work"))
        self.assertFalse((self.root / "work").exists())


class ControllerGlueTests(unittest.TestCase):
    """publish_sub_combos / clear_sub_combos with a fake IPC (no mpv)."""

    def setUp(self):
        import os
        import shutil
        from types import SimpleNamespace
        from oh_streamer.player import controller
        self.controller = controller
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "v.mp4"
        self.video.write_bytes(b"x")
        self.sets = []
        self.logs = []
        ipc = SimpleNamespace(set=lambda k, v: self.sets.append((k, v)))
        deps = SimpleNamespace(os=os, tempfile=tempfile, shutil=shutil, log=self.logs.append, MpvError=RuntimeError)
        self.ctl = SimpleNamespace(ipc=ipc, _deps=deps, cur_open=(str(self.video), {}), combo_dir=None, combos_published=False)
        self.ctl.publish_sub_combos = lambda refresh=False: controller.PlayerController.publish_sub_combos(self.ctl, refresh)
        self.addCleanup(lambda: self.ctl.combo_dir and shutil.rmtree(self.ctl.combo_dir, ignore_errors=True))

    def call(self, name):
        getattr(self.controller.PlayerController, name)(self.ctl)

    def test_nothing_published_without_manifest(self):
        self.call("publish_sub_combos")
        self.call("clear_sub_combos")
        self.assertEqual(self.sets, [])
        self.assertIsNone(self.ctl.combo_dir)

    def test_urls_and_missing_input_are_ignored(self):
        for src in ("https://example.com/a.mp4", None):
            self.ctl.cur_open = (src, {}) if src else None
            self.call("publish_sub_combos")
        self.assertEqual(self.sets, [])

    def test_publish_then_clear(self):
        folder = self.root / "v.subtitle-combos"
        folder.mkdir()
        (folder / "a.zh-TW.srt").write_text(SRT, encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(manifest([entry("a", "A")])), encoding="utf-8")
        self.call("publish_sub_combos")
        self.assertEqual(self.sets[0][0], "user-data/oh/sub-combos")
        self.assertEqual(json.loads(self.sets[0][1])["combos"][0]["id"], "a")
        self.assertTrue(self.ctl.combos_published)
        self.call("clear_sub_combos")
        self.assertEqual(self.sets[-1], ("user-data/oh/sub-combos", ""))
        self.call("clear_sub_combos")
        self.assertEqual(len(self.sets), 2)

    def _growing(self, status="running", count=2, updated="t1", cids=("a",)):
        folder = self.root / "v.subtitle-combos"
        folder.mkdir(exist_ok=True)
        rows = []
        for cid in cids:
            (folder / f"{cid}.zh-TW.srt").write_text(SRT, encoding="utf-8")
            e = entry(cid, status=status)
            e["cue_count"] = count
            rows.append(e)
        (folder / "manifest.json").write_text(json.dumps(manifest(rows, updated_at=updated)), encoding="utf-8")

    def test_poll_rebuilds_only_when_fingerprint_changes(self):
        self._growing()
        self.call("publish_sub_combos")
        self.assertEqual(len(self.sets), 1)
        composite = Path(self.ctl.combo_dir) / sc.COMPOSITE_FILENAME
        mtime = composite.stat().st_mtime_ns
        for _ in range(3):                                   # unchanged manifest: nothing rebuilt, nothing published
            self.controller.PlayerController.poll_sub_combos(self.ctl, 0.0)
        self.assertEqual(len(self.sets), 1)
        self.assertEqual(composite.stat().st_mtime_ns, mtime)
        # the scheduler appends cues: same combos -> rev bump, composite rebuilt
        folder = self.root / "v.subtitle-combos"
        self._growing(count=3, updated="t2")
        (folder / "a.zh-TW.srt").write_text(SRT + EXTRA_CUE, encoding="utf-8")
        nxt = self.controller.PlayerController.poll_sub_combos(self.ctl, 10.0)
        self.assertEqual(nxt, 10.0 + sc.POLL_INTERVAL_S)
        self.assertEqual(self.sets[-1], ("user-data/oh/sub-combos-rev", 1))
        self.assertIn("第三句", composite.read_text(encoding="utf-8"))
        self.controller.PlayerController.poll_sub_combos(self.ctl, 20.0)     # unchanged again
        self.assertEqual(len(self.sets), 2)

    def test_poll_republishes_list_when_a_combo_appears(self):
        self._growing(cids=("a",))
        self.call("publish_sub_combos")
        self._growing(cids=("a", "b"), updated="t2")
        self.controller.PlayerController.poll_sub_combos(self.ctl, 0.0)
        self.assertEqual(self.sets[-1][0], "user-data/oh/sub-combos")
        self.assertEqual([c["id"] for c in json.loads(self.sets[-1][1])["combos"]], ["a", "b"])

    def test_poll_picks_up_a_manifest_whose_first_combo_had_no_file_yet(self):
        folder = self.root / "v.subtitle-combos"
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(manifest([entry("a", status="running")], updated_at="t0")), encoding="utf-8")
        self.call("publish_sub_combos")                       # no zh yet: nothing published, but watched
        self.assertEqual(self.sets, [])
        self._growing(updated="t1")
        self.controller.PlayerController.poll_sub_combos(self.ctl, 0.0)
        self.assertEqual(self.sets[-1][0], "user-data/oh/sub-combos")

    def test_poll_idle_without_manifest_or_after_clear(self):
        self.controller.PlayerController.poll_sub_combos(self.ctl, 0.0)
        self.assertEqual(self.sets, [])
        self._growing()
        self.call("publish_sub_combos")
        self.call("clear_sub_combos")
        n = len(self.sets)
        self._growing(updated="t9")
        self.controller.PlayerController.poll_sub_combos(self.ctl, 0.0)
        self.assertEqual(len(self.sets), n)

    def _queue_item(self, name="output.mkv"):
        item = self.root / "items" / "q1"
        (item / "subtitles" / "combos").mkdir(parents=True)
        out = item / name
        out.write_bytes(b"x")
        folder = item / "subtitles" / "combos"
        (folder / "a.zh-TW.srt").write_text(SRT, encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(manifest([entry("a", "A")])), encoding="utf-8")
        return out, folder

    def test_processed_queue_output_publishes_by_layout(self):
        from types import SimpleNamespace
        out, _folder = self._queue_item()
        self.ctl.cur_open = None
        self.ctl.processed = SimpleNamespace(path=str(out), combos_dir=None)
        self.call("publish_sub_combos")
        self.assertEqual(self.sets[0][0], "user-data/oh/sub-combos")
        self.assertEqual(json.loads(self.sets[0][1])["combos"][0]["id"], "a")

    def test_processed_explicit_folder_wins_for_any_file_name(self):
        from types import SimpleNamespace
        out, folder = self._queue_item("renamed.mkv")
        self.ctl.cur_open = None
        self.ctl.processed = SimpleNamespace(path=str(out), combos_dir=str(folder))
        self.call("publish_sub_combos")
        self.assertEqual(len(self.sets), 1)

    def test_processed_without_manifest_publishes_nothing(self):
        from types import SimpleNamespace
        self.ctl.cur_open = None
        self.ctl.processed = SimpleNamespace(path=str(self.video), combos_dir=str(self.root / "nope"))
        self.call("publish_sub_combos")
        self.assertEqual(self.sets, [])

    def test_errors_never_propagate(self):
        folder = self.root / "v.subtitle-combos"
        folder.mkdir()
        (folder / "a.zh-TW.srt").write_text(SRT, encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(manifest([entry("a")])), encoding="utf-8")
        def boom(k, v):
            raise OSError("pipe closed")
        self.ctl.ipc.set = boom
        self.call("publish_sub_combos")
        self.assertTrue(any("skipped" in l for l in self.logs))


if __name__ == "__main__":
    unittest.main()
