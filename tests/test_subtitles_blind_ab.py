"""Offline blind-review alignment, anonymity, vote validation and scoring."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copy import deepcopy
import hashlib
import json
from tempfile import TemporaryDirectory
import unittest

from oh_streamer.subtitles.blind_ab import (
    VOTE_FORMAT, build_blind_review, main, score_votes, write_blind_review,
)


class BlindReviewTests(unittest.TestCase):
    def setUp(self):
        self.source = [{"id": f"source-{i}", "start": i, "end": i + .8, "text": f"日本語{i}"}
                       for i in range(6)]
        self.translations = {
            name: [{"id": row["id"], "start": row["start"], "end": row["end"],
                    "text_ja": row["text"], "text": f"譯文{suffix}{i}"}
                   for i, row in enumerate(self.source)]
            for name, suffix in (("secret_model_one", "甲"), ("secret_model_two", "乙"))
        }

    def votes(self, key, choices):
        return {"format": VOTE_FORMAT, "dataset_id": key["dataset_id"],
                "votes": [{"cue_id": cue["id"], "choice": choice}
                          for cue, choice in zip(key["cues"], choices)]}

    def test_reproducible_shuffled_and_anonymous_page(self):
        page, key = build_blind_review(self.source, self.translations, seed=4)
        self.assertEqual((page, key), build_blind_review(self.source, self.translations, seed=4))
        self.assertNotEqual(key, build_blind_review(self.source, self.translations, seed=5)[1])
        self.assertEqual({c["options"]["A"]["model"] for c in key["cues"]}, set(self.translations))
        for name in (*self.translations, "source-0"):
            self.assertNotIn(name, page)
        self.assertIn("dataset_id:datasetId", page)
        self.assertIn("blind_votes.json", page)

    def test_three_models_and_all_cues_kept(self):
        self.translations["third"] = deepcopy(self.translations["secret_model_one"])
        _, key = build_blind_review(self.source, self.translations, seed=1)
        self.assertEqual(len(key["cues"]), 6)
        for cue in key["cues"]:
            self.assertEqual(set(cue["options"]), set("ABC"))
            self.assertEqual({x["model"] for x in cue["options"].values()}, set(self.translations))

    def test_review_uses_same_taiwan_postprocessing_as_subtitle_display(self):
        from oh_streamer.subtitles.postprocess import display_segments
        from oh_streamer.subtitles.sentences import wrap_two_lines
        row = self.translations['secret_model_one'][0]
        row['text'] = '請結賬,你好?'
        _, key = build_blind_review(self.source, self.translations, seed=4)
        option = next(o for o in key['cues'][0]['options'].values()
                      if o['model'] == 'secret_model_one')
        self.assertEqual(option['text'], wrap_two_lines(display_segments([row])[0].text))
        self.assertEqual(option['text'], '請結帳，你好？')

    def test_alignment_by_id_accepts_reordering_and_preserves_missing_hidden(self):
        self.translations["secret_model_one"].reverse()
        del self.translations["secret_model_two"][1]
        self.translations["secret_model_two"][2]["preserved_original"] = True
        page, key = build_blind_review(self.source, self.translations, seed=2)
        for i, cue in enumerate(key["cues"]):
            by_model = {o["model"]: o for o in cue["options"].values()}
            self.assertEqual(by_model["secret_model_one"]["text"], f"譯文甲{i}")
        self.assertIn("（缺少譯文）", page)
        self.assertIn("（未顯示譯文）", page)
        self.assertNotIn("譯文乙3", page)
        self.assertEqual(key["cues"][1]["source_id"], "source-1")

    def test_existing_idless_json_aligns_by_full_identity(self):
        for row in self.source:
            del row["id"]
        for rows in self.translations.values():
            for row in rows:
                del row["id"]
            rows.reverse()
        _, key = build_blind_review({"segments": self.source}, self.translations, seed=3)
        self.assertEqual({o["text"] for o in key["cues"][0]["options"].values()}, {"譯文甲0", "譯文乙0"})

    def test_duplicate_ids_mismatch_and_extra_cues_rejected(self):
        for mutation in (
            lambda rows: rows.append(deepcopy(rows[0])),
            lambda rows: rows[0].update(start=.2),
            lambda rows: rows[0].update(text_ja="其他日文"),
            lambda rows: rows[0].update(id="unknown"),
        ):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.translations)
                mutation(changed["secret_model_one"])
                with self.assertRaises(ValueError):
                    build_blind_review(self.source, changed)
        self.source.append(deepcopy(self.source[0]))
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            build_blind_review(self.source, self.translations)

    def test_ambiguous_idless_alignment_rejected(self):
        self.source = [{k: v for k, v in self.source[0].items() if k != "id"}] * 2
        self.translations = {name: [{k: v for k, v in rows[0].items() if k != "id"}]
                             for name, rows in self.translations.items()}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            build_blind_review(self.source, self.translations)

    def test_html_injection_escaped(self):
        self.source[0]["text"] = '</script><img src=x onerror="evil()">&'
        for rows in self.translations.values():
            rows[0]["text_ja"] = self.source[0]["text"]
            rows[0]["text"] = '<script>evil()</script>"&'
        page, _ = build_blind_review(self.source, self.translations)
        self.assertNotIn('<img src=x', page)
        self.assertNotIn('<script>evil()', page)
        self.assertIn('&lt;script&gt;evil()', page)
        self.assertIn('&lt;/script&gt;&lt;img', page)

    def test_ties_skips_unanswered_and_exact_denominators(self):
        _, key = build_blind_review(self.source, self.translations, seed=4)
        winner = "secret_model_one"
        choices = [next(k for k, v in c["options"].items() if v["model"] == winner)
                   for c in key["cues"][:2]] + ["tie", "skip"]
        report = score_votes(key, self.votes(key, choices))
        counts = report["models"][winner]
        self.assertEqual((counts["wins"], counts["losses"], counts["ties"], counts["abstentions"]), (2, 0, 1, 3))
        self.assertEqual(counts["denominator"], 3)
        self.assertAlmostEqual(counts["win_rate"], 2 / 3)
        self.assertEqual(report["models"]["secret_model_two"]["losses"], 2)
        self.assertEqual((report["skipped"], report["unanswered"]), (1, 2))

    def test_no_votes_has_null_rate(self):
        _, key = build_blind_review(self.source, self.translations)
        for counts in score_votes(key, self.votes(key, []))["models"].values():
            self.assertIsNone(counts["win_rate"])
            self.assertEqual(counts["denominator"], 0)

    def test_votes_bound_to_exact_dataset(self):
        _, key = build_blind_review(self.source, self.translations, seed=1)
        _, other = build_blind_review(self.source, self.translations, seed=2)
        with self.assertRaisesRegex(ValueError, "binding"):
            score_votes(other, self.votes(key, ["A"]))
        changed = deepcopy(key)
        changed["cues"][0]["text_ja"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash"):
            score_votes(changed, self.votes(key, ["A"]))

    def test_duplicate_unknown_and_invalid_votes_rejected(self):
        _, key = build_blind_review(self.source, self.translations)
        for rows in (
            [{"cue_id": "q000001", "choice": "A"}] * 2,
            [{"cue_id": "q999999", "choice": "A"}],
            [{"cue_id": "q000001", "choice": "C"}],
            [{"cue_id": "q000001", "choice": []}],
        ):
            votes = self.votes(key, [])
            votes["votes"] = rows
            with self.assertRaises(ValueError):
                score_votes(key, votes)

    def test_key_mapping_validation_even_if_hash_recomputed(self):
        _, key = build_blind_review(self.source, self.translations)
        key["cues"][0]["options"]["B"]["model"] = key["cues"][0]["options"]["A"]["model"]
        content = {k: v for k, v in key.items() if k != "dataset_id"}
        key["dataset_id"] = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            score_votes(key, self.votes(key, []))

    def test_timestamp_and_model_validation(self):
        for timestamp in (float("nan"), float("inf"), True, -1, "0"):
            source = deepcopy(self.source)
            source[0]["start"] = timestamp
            with self.assertRaises(ValueError):
                build_blind_review(source, self.translations)
        with self.assertRaises(ValueError):
            build_blind_review(self.source, {"single": []})

    def test_cli_generate_score_and_file_apis(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            source.write_text(json.dumps(self.source), "utf-8")
            arguments = ["generate", str(source), "--out", str(root), "--seed", "4"]
            for i, (name, rows) in enumerate(self.translations.items()):
                path = root / f"model{i}.json"
                path.write_text(json.dumps({"segments": rows}), "utf-8")
                arguments += ["--translation", f"{name}={path}"]
            self.assertEqual(main(arguments), 0)
            key = json.loads((root / "answer_key.json").read_text("utf-8"))
            self.assertTrue((root / "blind_ab.html").is_file())
            votes = root / "votes.json"
            votes.write_text(json.dumps(self.votes(key, ["A", "tie"])), "utf-8")
            output = root / "score.json"
            self.assertEqual(main(["score", str(root / "answer_key.json"), str(votes), "--out", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text("utf-8"))["total_cues"], 6)
            self.assertEqual(write_blind_review(source, self.translations, root, seed=4), key)

    def test_duplicate_json_fields_rejected(self):
        with TemporaryDirectory() as folder:
            file = Path(folder) / "invalid.json"
            file.write_text('{"segments": [], "segments": []}', "utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                build_blind_review(file, self.translations)


if __name__ == "__main__":
    unittest.main()
