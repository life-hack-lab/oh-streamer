"""CPU coverage of exact V3 span fusion and parent-level F7 agreement rescue."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles.asr import segment_from_record
from oh_streamer.subtitles.core import Segment, clean_segments
from oh_streamer.subtitles.fusion import fuse_segments, fusion_identity, select_candidate
from oh_streamer.subtitles.sentences import sentence_cues


def row(text, *, score=-.5, ratio=.8, start=0, end=8, **more):
    return dict(start=start, end=end, text=text, avg_logprob=score,
                compression_ratio=ratio, no_speech_prob=.01, gap=True, **more)


class FusionTests(unittest.TestCase):
    def test_normalized_identical_and_high_agreement_always_select_a1(self):
        self.assertEqual(select_candidate(row("ココに来て。", score=-.9), row("ここに来て!", score=-.1)), ("A1", "same"))
        self.assertEqual(select_candidate(row("ここに来てください。", score=-.9), row("ここに来てくださいね。", score=-.1)), ("A1", "agree"))

    def test_one_empty_both_empty_and_empty_records_are_preserved(self):
        self.assertEqual(select_candidate(row(""), row("向こうへ行こう")), ("A2", "one_empty"))
        self.assertEqual(select_candidate(row("向こうへ行こう"), row("")), ("A1", "one_empty"))
        output = fuse_segments([row("")], [row("")])
        self.assertEqual(output[0]["text"], "")
        self.assertEqual(output[0]["selected_model"], "A1")

    def test_loop_and_compression_demotion_precede_logprob(self):
        self.assertEqual(select_candidate(row("そうそうそうそう", score=-.1), row("向こうへ行こう", score=-.9)), ("A2", "demote"))
        self.assertEqual(select_candidate(row("向こうへ行こう", score=-.9), row("今日は何をしますか", score=-.1, ratio=1.2)), ("A1", "demote"))
        self.assertEqual(select_candidate(row("向こうへ行こう", score=-.9, ratio=1.2), row("今日は何をしますか", score=-.1, ratio=1.2)), ("A2", "alp"))

    def test_ties_choose_a1_and_only_strictly_higher_a2_score_wins(self):
        for left, right, chosen in ((-.5, -.5, "A1"), (-.5, -.499, "A2"),
                                    (-.5, -.501, "A1"), (None, None, "A1"),
                                    (None, -.5, "A2"), (0, -.1, "A1")):
            with self.subTest(left=left, right=right):
                self.assertEqual(select_candidate(row("向こうへ行こう", score=left),
                                                 row("今日は何をしますか", score=right)), (chosen, "alp"))

    def test_chosen_record_metadata_and_parent_alt_are_preserved_and_deterministic(self):
        first, second = row("向こうへ行こう", score=-.9), row("今日は何をしますか", score=-.1, ratio=.6)
        result = fuse_segments([first], [second])
        self.assertEqual(result, fuse_segments([first], [second]))
        self.assertEqual(result[0]["text"], second["text"])
        self.assertEqual(result[0]["alt_text"], first["text"])
        self.assertEqual(result[0]["compression_ratio"], .6)
        self.assertEqual(result[0]["avg_logprob"], -.1)
        self.assertNotIn("alt_text", first)
        self.assertEqual(fusion_identity(), fusion_identity())

    def test_exact_span_sets_required_and_order_is_stable(self):
        first, second = row("言葉"), row("言葉", start=8, end=9)
        output = fuse_segments([second, first], [first, second])
        self.assertEqual([item["start"] for item in output], [0, 8])
        for left, right in (([first], [row("言葉", start=.00001)]), ([first, first], [first]), ([first], [])):
            with self.assertRaises(ValueError):
                fuse_segments(left, right)

    def test_parent_agreement_rescues_each_sentence_but_single_disables_rescue(self):
        text = "ここへ来てください。向こうには行かないで。"
        first = row(text, score=-.9)
        second = row("ここへ来てください。向こうには行かないでね。", score=-.8)
        fused = fuse_segments([first], [second])[0]
        parent = segment_from_record(fused)
        cues = sentence_cues(parent)
        self.assertGreater(len(cues), 1)
        self.assertGreaterEqual(parent.asr_agreement, .8)
        self.assertEqual(len(clean_segments(cues, show_interjections=True)), len(cues))
        single = segment_from_record(fuse_segments([dict(first, alt_text=text, asr_agreement=1)], mode="single")[0])
        self.assertEqual(single.alt_text, "")
        self.assertEqual(clean_segments(sentence_cues(single)), [])
        self.assertTrue(all(cue.low_confidence for cue in clean_segments(sentence_cues(single), include_low_confidence=True)))
        # Agreement never rescues an exact hard hallucination.
        bad = replace(parent, text="おはようございます", alt_text="おはようございます", asr_agreement=1)
        self.assertEqual(clean_segments([bad], include_low_confidence=True), [])

    def test_legacy_stored_low_marker_survives_without_confidence(self):
        source = Segment(0, 1, "ここへ来て", low_confidence=True)
        self.assertEqual(clean_segments([source]), [])
        self.assertTrue(clean_segments([source], include_low_confidence=True)[0].low_confidence)


if __name__ == "__main__":
    unittest.main()
