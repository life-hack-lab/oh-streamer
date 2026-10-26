"""CPU tests for the opt-in short-reply rule (sentences.filter_short policy='replies')."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles.core import Segment
from oh_streamer.subtitles.sentences import filter_short
from oh_streamer.subtitles.short_replies import (
    classify_short, filter_short_replies, is_known_reply, normalize)


class ShortReplyTests(unittest.TestCase):
    def test_normalize_folds_prolongation_katakana_and_reduplication(self):
        self.assertEqual(normalize('うーん…'), 'うん')
        self.assertEqual(normalize('ウンウン'), 'うん')
        self.assertEqual(normalize('はいっ！'), 'はい')

    def test_known_replies_are_kept(self):
        for text in ('わかりました。', 'あ、はい。', 'えっ…', 'なるほど', 'うーん…', 'よいしょ', 'ありがとうございます'):
            self.assertTrue(is_known_reply(text), text)
            self.assertEqual(classify_short(text, .9), (True, 'known_reply'), text)

    def test_hard_rejects(self):
        self.assertEqual(classify_short('…', .8), (False, 'empty'))
        self.assertEqual(classify_short('ご視聴ありがとうございました', .5), (False, 'blacklist'))
        self.assertEqual(classify_short('ほんのほんのほんのほんの', .9), (False, 'repetition'))
        self.assertEqual(classify_short('OK thanks', .9), (False, 'foreign_script'))
        self.assertEqual(classify_short('等待', .8), (False, 'han_only'))
        self.assertEqual(classify_short('そうなんですよねえ本当に', .5), (False, 'implausible_rate'))

    def test_unknown_text_needs_signals(self):
        self.assertEqual(classify_short('ちょめちょめ', .9, .3), (True, 'signals'))
        self.assertEqual(classify_short('ちょめちょめ', .9, 5.0, min_signals=3), (True, 'signals'))  # 3 of 4 signals
        self.assertEqual(classify_short('ちょめちょめ', .9, 5.0), (False, 'weak_signals'))  # default needs 4
        self.assertEqual(classify_short('ちょめちょめ', .9, 5.0, min_signals=4), (False, 'weak_signals'))
        self.assertEqual(classify_short('…ちょめちょめ', .9, 5.0, min_signals=3), (False, 'weak_signals'))  # leading ellipsis
        self.assertEqual(classify_short('ちょめちょめ', .9, None, min_signals=4), (False, 'weak_signals'))

    def test_rate_check_applies_before_lexicon(self):
        self.assertEqual(classify_short('はい', .4), (True, 'known_reply'))
        self.assertEqual(classify_short('はい', .1), (False, 'implausible_rate'))

    def test_filter_contract_and_default_unchanged(self):
        rows = [Segment(0, 3, '長い発話です'), Segment(3.5, 4.2, 'わかりました'),
                Segment(10, 10.8, '…'), Segment(20, 20.9, 'ご視聴ありがとうございました'),
                Segment(30, 30.9, 'ちょめちょめ'), Segment(40.0, 43.0, '次の発話です')]
        kept, rejected = filter_short_replies(rows, 1., min_signals=4)
        self.assertEqual(kept, [rows[0], rows[1], rows[5]])
        self.assertEqual(rejected, [rows[2], rows[3], rows[4]])
        self.assertIn(rows[4], filter_short_replies(rows, 1., min_signals=3)[0])  # 3 of 4 signals
        near = rows[:4] + [Segment(30, 30.9, 'ちょめちょめ'), Segment(31., 34., '次です')]
        self.assertIn(near[4], filter_short_replies(near, 1., min_signals=4)[0])  # neighbour within 1 s
        self.assertEqual(filter_short(rows, 1., 'replies'), filter_short_replies(rows, 1., 4))
        # the default policy is unchanged: only the old whitelist survives
        self.assertEqual(filter_short(rows, 1.)[0], [rows[0], rows[5]])
        with self.assertRaises(ValueError):
            filter_short(rows, 1., 'nope')


if __name__ == '__main__':
    unittest.main()
