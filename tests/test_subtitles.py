"""Pure subtitle helper tests; no speech models or third-party packages required."""

from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import unittest

from oh_streamer.subtitles.core import (
    Segment,
    clean_segments,
    format_timestamp,
    inspect_hallucinations,
    is_interjection,
    make_translation_batches,
    parse_numbered_translations,
    parse_srt,
    parse_timestamp,
    render_srt,
    to_taiwan_traditional,
    translate_segments,
)


class SubtitleTimecodeTests(unittest.TestCase):
    def test_timestamp_round_trip_and_srt_parse(self):
        self.assertEqual(parse_timestamp("00:01:02,345"), 62.345)
        self.assertEqual(parse_timestamp("00:00:01.2"), 1.2)
        self.assertEqual(format_timestamp(62.345), "00:01:02,345")
        cues = parse_srt(
            "\ufeff1\r\n00:00:01,200 --> 00:00:02,000\r\n第一行\r\n第二行\r\n\r\n"
            "2\r\n00:00:03,000 --> 00:00:04,000\r\n下一句\r\n"
        )
        self.assertEqual(cues, [
            Segment(1.2, 2.0, "第一行\n第二行"),
            Segment(3.0, 4.0, "下一句"),
        ])

    def test_render_wraps_18_chars_and_splits_long_text_with_bounded_durations(self):
        text = "這是一段很長的中文字幕內容需要依照指定字數換行並在超長時拆成多個字幕片段以便觀眾閱讀理解"
        rendered = render_srt([Segment(0.0, 18.0, text)])
        cues = parse_srt(rendered)
        self.assertGreater(len(cues), 1)
        self.assertEqual("".join(cue.text.replace("\n", "") for cue in cues), text)
        for cue in cues:
            self.assertLessEqual(max(map(len, cue.text.splitlines())), 18)
            self.assertLessEqual(cue.end - cue.start, 7.0)
            self.assertGreaterEqual(cue.end - cue.start, 0.7)

    def test_short_cue_is_held_for_minimum_duration(self):
        cue = parse_srt(render_srt([Segment(1.0, 1.2, "短句")]))[0]
        self.assertAlmostEqual(cue.end - cue.start, 0.7)

    def test_short_cue_respects_next_sentence_boundary(self):
        cues = parse_srt(render_srt([
            Segment(0.0, 0.1, "第一句"),
            Segment(0.5, 1.0, "下一句"),
        ]))
        self.assertEqual(cues[0].end, cues[1].start)
        self.assertAlmostEqual(cues[0].end - cues[0].start, 0.5)
        self.assertAlmostEqual(cues[1].end - cues[1].start, 0.7)

    def test_simultaneous_starts_merge_instead_of_emitting_zero_length_cues(self):
        cues = parse_srt(render_srt([
            Segment(0.0, 2.0, "同時發生甲"),
            Segment(0.0, 1.0, "同時發生乙"),
            Segment(2.0, 3.0, "下一句"),
        ]))
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "同時發生甲 同時發生乙")
        self.assertEqual(cues[0].end, cues[1].start)

    def test_split_cues_fill_only_the_available_interval(self):
        long_text = "這是一段很長的中文字幕內容需要分成多個字幕片段並且每個片段都必須保持正確時間"
        cues = parse_srt(render_srt([
            Segment(0.0, 18.0, long_text),
            Segment(5.0, 6.0, "下一句"),
        ]))
        first_sentence_cues = [cue for cue in cues if cue.start < 5.0]
        self.assertGreater(len(first_sentence_cues), 1)
        self.assertEqual("".join(cue.text.replace("\n", "") for cue in first_sentence_cues), long_text)
        self.assertEqual(first_sentence_cues[-1].end, 5.0)
        for earlier, later in zip(cues, cues[1:]):
            self.assertLessEqual(earlier.end, later.start)
            self.assertLessEqual(earlier.end - earlier.start, 7.0)

    def test_unsplittable_short_text_never_emits_a_long_cue(self):
        cue = parse_srt(render_srt([Segment(0.0, 30.0, "嗯")]))[0]
        self.assertEqual(cue.end - cue.start, 7.0)

    def test_invalid_srt_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_timestamp("00:60:00,000")
        with self.assertRaises(ValueError):
            parse_srt("1\nnot-a-time --> 00:00:01,000\ntext")
        with self.assertRaises(ValueError):
            parse_srt("1\n --> 00:00:01,000\ntext")

    def test_segments_reject_nonfinite_timing_and_confidence_values(self):
        invalid_values = (
            {"start": float("nan"), "end": 1.0},
            {"start": 0.0, "end": float("inf")},
            {"start": 0.0, "end": 1.0, "no_speech_prob": float("nan")},
            {"start": 0.0, "end": 1.0, "avg_logprob": -float("inf")},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Segment(text="bad", **values)


class SubtitleCleanupTests(unittest.TestCase):
    def test_subtitle_group_credit_pattern_is_blacklisted(self):
        cues = [
            Segment(0, 1, "字幕：翻訳協力 山田"),
            Segment(1, 2, "字幕組スタッフが来ました"),
            Segment(2, 3, "今日は字幕の話をします"),
        ]
        self.assertEqual([cue.text for cue in clean_segments(cues)], ["今日は字幕の話をします"])

    def test_blacklist_vad_duplicates_ngram_and_default_hidden_interjections(self):
        cues = [
            Segment(0, 1, "ご視聴ありがとうございました"),
            Segment(2, 3, "あっ"),
            Segment(4, 5, "こんにちは"),
            Segment(5, 6, "こんにちは"),
            Segment(7, 8, "同じ文同じ文同じ文"),
            Segment(9, 10, "無音区間"),
        ]
        clean = clean_segments(cues, vad_intervals=[(0, 8.5)])
        self.assertEqual([cue.text for cue in clean], ["こんにちは", "無音区間"])
        self.assertEqual(clean_segments([Segment(0, 1, "あっ")]), [])
        self.assertEqual(clean_segments([Segment(0, 1, "あっ、んっ、はぁ")]), [])
        shown = clean_segments([Segment(0, 1, "あっ")], show_interjections=True)
        self.assertEqual(shown[0].flags, ("interjection",))
        mixed = clean_segments([Segment(0, 1, "あっ、んっ、はぁ")], show_interjections=True)
        self.assertEqual(mixed[0].flags, ("interjection",))
        self.assertTrue(is_interjection("はぁ…"))
        self.assertTrue(is_interjection("あっ、んっ、はぁ"))

    def test_non_speech_only_blacklist_entry_respects_vad(self):
        phrase = Segment(1.0, 2.0, "お疲れ様でした")
        self.assertEqual(clean_segments([phrase], vad_intervals=[(1.0, 2.0)]), [phrase])
        self.assertEqual(clean_segments([phrase], vad_intervals=[(3.0, 4.0)]), [])
        self.assertEqual(clean_segments([phrase]), [phrase])

    def test_whisper_confidence_thresholds_are_applied_only_when_present(self):
        cues = [
            Segment(0, 1, "低信心", no_speech_prob=0.9),
            Segment(1, 2, "低對數", avg_logprob=-2.0),
            Segment(2, 3, "無分數"),
        ]
        self.assertEqual(len(clean_segments(cues)), 3)
        self.assertEqual([cue.text for cue in clean_segments(cues, no_speech_threshold=.6,
                         avg_logprob_threshold=-1.0)], ["無分數"])
        self.assertEqual(len(clean_segments(cues, no_speech_threshold=None, avg_logprob_threshold=None)), 3)

    def test_hallucination_report_counts_categories_and_unique_union(self):
        cues = [
            Segment(0, 1, "チャンネル登録お願いします"),
            Segment(1, 2, "重複"),
            Segment(2, 3, "重複"),
            Segment(3, 4, "重複"),
            Segment(8, 9, "非語音"),
        ]
        counts = inspect_hallucinations(cues, vad_intervals=[(0, 4)])
        self.assertEqual(counts, {
            "blacklist": 1,
            "consecutive_repeat": 2,
            "frequent_repeat": 3,
            "outside_vad": 1,
            "repeated_ngram": 0,
            "unique_suspect": 5,
        })

    def test_repetition_inside_one_long_asr_span_is_counted(self):
        counts = inspect_hallucinations([Segment(0, 600, '先頭です。繰返繰返繰返後の台詞')])
        self.assertEqual(counts['repeated_ngram'], 1)
        self.assertEqual(counts['unique_suspect'], 1)


class TranslationLogicTests(unittest.TestCase):
    def test_batches_have_20_to_40_items_and_prior_five_as_context(self):
        cues = [Segment(float(i), float(i + 1), f"句子{i}") for i in range(25)]
        batches = make_translation_batches(cues, batch_size=20)
        self.assertEqual([len(batch.items) for batch in batches], [20, 5])
        self.assertEqual(batches[0].expected_ids, tuple(range(1, 21)))
        self.assertEqual([cue.text for cue in batches[1].context], [f"句子{i}" for i in range(15, 20)])
        self.assertIn("翻譯為簡體中文", batches[1].prompt)
        self.assertIn("只輸出編號與簡體中文譯文", batches[1].prompt)
        self.assertIn("句子19", batches[1].prompt)
        self.assertNotIn("1. 句子15", batches[1].prompt)
        with self.assertRaises(ValueError):
            make_translation_batches(cues, batch_size=10)

    def test_numbered_output_must_match_expected_order_and_be_complete(self):
        self.assertEqual(
            parse_numbered_translations("1. 你好\n2) 再見", 2),
            ["你好", "再見"],
        )
        for invalid in (
            "1. 你好",
            "1. 你好\n1. 重複",
            "2. 再見\n1. 你好",
            "說明\n1. 你好",
            "1. 你好\n以上為翻譯",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_numbered_translations(invalid, 2)

    def test_mismatch_falls_back_per_sentence_and_preserves_failed_source(self):
        cues = [Segment(float(i), float(i + 1), f"source-{i}") for i in range(21)]
        batch_calls = []
        single_calls = []

        def translate_batch(batch):
            batch_calls.append(batch)
            if batch_calls[0] is batch:
                return "1. 這批編號不完整"
            return "1. 完成"

        def translate_one(text):
            single_calls.append(text)
            if text == "source-2":
                raise RuntimeError("single sentence failed")
            return f"譯-{text}"

        result = translate_segments(
            cues,
            translate_batch,
            translate_one,
            batch_size=20,
            opencc=lambda text: "繁:" + text,
        )
        self.assertEqual(len(result), 21)
        self.assertEqual(result.metrics, {
            "mismatch_batches": 1,
            "fallback_sentences": 20,
            "untranslated": 1,
            "suspected_refusals": 0,
            "residue_retry_sentences": 0,
            "residue_unresolved_sentences": 0,
        })
        self.assertEqual(len(single_calls), 20)
        self.assertTrue(result.results[2].preserved_original)
        self.assertEqual(result.results[2].text, "[原文] source-2")
        self.assertEqual(result.results[-1].text, "繁:完成")

    def test_japanese_residue_retries_only_affected_cues(self):
        calls = []
        def single(text):
            calls.append(text)
            return '振作起来'
        run = translate_segments([Segment(0, 1, 'こんにちは'), Segment(1, 2, 'しっかりしろ')],
                                 lambda batch: '1. 你好\n2. しっかりしろ', single,
                                 opencc=lambda value: value)
        self.assertEqual(calls, ['しっかりしろ'])
        self.assertEqual(run.metrics['residue_retry_sentences'], 1)
        self.assertEqual(run.metrics['untranslated'], 0)
        self.assertEqual(run.results[1].text, '振作起来')
        failed = translate_segments([Segment(0, 1, 'しっかりしろ')],
                                    lambda batch: '1. しっかりしろ', lambda text: text,
                                    opencc=lambda value: value)
        self.assertTrue(failed.results[0].preserved_original)
        self.assertEqual(failed.metrics['residue_unresolved_sentences'], 1)

    def test_opencc_converter_object_and_refusal_heuristic_metrics(self):
        class MockOpenCC:
            def __init__(self):
                self.calls = []

            def convert(self, value):
                self.calls.append(value)
                return "TW:" + value

        converter = MockOpenCC()
        cue = Segment(0, 1, "本文")
        result = translate_segments(
            [cue],
            lambda batch: "1. 抱歉，無法協助這個請求",
            lambda text: self.fail("valid batch should not retry"),
            opencc=converter,
        )
        self.assertEqual(converter.calls, ["抱歉，無法協助這個請求"])
        self.assertEqual(result.results[0].text, "TW:抱歉，無法協助這個請求")
        self.assertEqual(result.metrics["suspected_refusals"], 1)
        self.assertEqual(result.metrics["mismatch_batches"], 0)

    def test_real_opencc_s2twp_conversion_when_installed(self):
        try:
            import opencc  # noqa: F401
        except ImportError:
            self.skipTest("OpenCC is optional outside the subtitle PoC environment")
        self.assertEqual(
            to_taiwan_traditional("软件、数据库、鼠标、出租车"),
            "軟體、資料庫、滑鼠、計程車",
        )


if __name__ == "__main__":
    unittest.main()
