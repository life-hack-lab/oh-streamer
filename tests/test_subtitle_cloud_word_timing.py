"""Cloud ASR (Groq Whisper) returns whole-second segment times; word timestamps refill the boundaries."""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from oh_streamer.subtitles import model_adapters as adapters


def _w(word, start, end):
    return {'word': word, 'start': start, 'end': end}


# Shape of a real Groq reply (語音測試01, 420 s+): segments on whole seconds, words at 10 ms precision.
SEGMENTS = [
    {'id': 0, 'start': 0.0, 'end': 2.0, 'text': ' 負けても貰えるんだ'},
    {'id': 1, 'start': 2.0, 'end': 4.0, 'text': ' 一応貰えます'},
    {'id': 2, 'start': 4.0, 'end': 6.0, 'text': ' やるだけで'},
]
WORDS = [_w('負', 0.0, 1.3), _w('けて', 1.3, 1.42), _w('も', 1.42, 1.66), _w('貰', 1.66, 1.92), _w('えるんだ', 1.92, 2.22),
         _w('一', 2.26, 2.32), _w('応', 2.32, 2.46), _w('貰', 2.46, 2.66), _w('え', 2.66, 2.8), _w('ます', 2.8, 2.86),
         _w('やる', 4.12, 4.3), _w('だけ', 4.3, 4.52), _w('で', 4.52, 4.76)]


class RefineSegmentsTests(unittest.TestCase):
    def test_whole_second_boundaries_follow_word_times(self):
        out = adapters.refine_segments_with_words(SEGMENTS, WORDS)
        self.assertEqual([(s['start'], s['end']) for s in out],
                         [(0.94, 2.26), (2.26, 3.11), (4.12, 5.01)])      # stretched first word capped, 0.25 s tail pad, no overlap
        self.assertEqual([s['text'] for s in out], [s['text'] for s in SEGMENTS])

    def test_precise_segment_times_are_replaced_by_word_times_too(self):
        segs = [dict(SEGMENTS[0], start=0.04, end=1.98)]
        out = adapters.refine_segments_with_words(segs, WORDS[:5])
        self.assertEqual((round(out[0]['start'], 3), round(out[0]['end'], 3)), (0.94, 2.47))

    def test_no_words_or_text_mismatch_keeps_segments(self):
        for words in (None, [], [_w('違', 0.1, 0.3)]):
            out = adapters.refine_segments_with_words(SEGMENTS, words)
            self.assertEqual([(s['start'], s['end']) for s in out], [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)])

    def test_words_missing_some_text_still_align(self):
        segs = [dict(SEGMENTS[0]), {'id': 1, 'start': 2.5, 'end': 4.5, 'text': ' 幻覚の繰り返し'}, dict(SEGMENTS[2], start=5.0, end=7.0)]
        words = WORDS[:5] + [_w('やる', 5.12, 5.3), _w('だけ', 5.3, 5.52), _w('で', 5.52, 5.76)]
        out = adapters.refine_segments_with_words(segs, words)
        self.assertEqual((round(out[0]['start'], 3), round(out[0]['end'], 3)), (0.94, 2.47))
        self.assertEqual((out[1]['start'], out[1]['end']), (2.5, 4.5))      # nothing matched: untouched
        self.assertEqual((out[2]['start'], round(out[2]['end'], 3)), (5.12, 6.01))

    def test_never_overlaps_previous_segment(self):
        words = [_w('負けても貰えるんだ', 0.0, 2.3), _w('一応貰えます', 2.1, 3.0)]
        out = adapters.refine_segments_with_words(SEGMENTS[:2], words)
        self.assertGreaterEqual(out[1]['start'], out[0]['end'])
        self.assertGreater(out[1]['end'], out[1]['start'])

    def test_pulls_back_earlier_cue_instead_of_overlapping_precise_one(self):
        segs = [dict(SEGMENTS[0]), dict(SEGMENTS[1], start=2.05, end=3.5)]
        out = adapters.refine_segments_with_words(segs, WORDS[:10])
        self.assertLessEqual(out[0]['end'], out[1]['start'])
        self.assertEqual(out[1]['start'], 2.26)

    def test_late_words_move_a_contiguous_segment_even_far_away(self):
        # Groq segments are contiguous: this one starts at 44.3 s but its words are spoken at 60-65 s (music before)
        segs = [{'id': 0, 'start': 44.32, 'end': 65.16, 'text': ' 負けても貰えるんだ'}]
        words = [_w('負けても', 60.08, 62.0), _w('貰えるんだ', 62.0, 64.94)]
        out = adapters.refine_segments_with_words(segs, words)
        self.assertEqual((out[0]['start'], round(out[0]['end'], 3)), (60.56, 65.19))

    def test_stretched_first_word_is_capped(self):
        segs = [{'id': 0, 'start': 79.0, 'end': 90.0, 'text': ' なんでやねん'}]
        words = [_w('なん', 79.06, 84.36), _w('でやねん', 84.36, 85.5)]
        out = adapters.refine_segments_with_words(segs, words)
        self.assertAlmostEqual(out[0]['start'], 84.36 - 0.72, places=3)


class RefineLoggingAndWordsTests(unittest.TestCase):
    def test_matched_segments_keep_their_words(self):
        out = adapters.refine_segments_with_words(SEGMENTS, WORDS)
        self.assertEqual([w['word'] for w in out[2]['words']], ['やる', 'だけ', 'で'])
        self.assertEqual(out[1]['words'][0], _w('一', 2.26, 2.32))
        self.assertEqual(sum(len(s['words']) for s in out), len(WORDS))

    def test_unmatched_segment_has_no_words(self):
        out = adapters.refine_segments_with_words(SEGMENTS[:1], [_w('違', 0.1, 0.3)])
        self.assertNotIn('words', out[0])

    def test_one_info_summary_per_call_and_reasons_at_debug(self):
        words = WORDS[:5] + [_w('一応貰えます', 9.0, 9.5)]
        with self.assertLogs(adapters._log, 'DEBUG') as logs:
            adapters.refine_segments_with_words(SEGMENTS[:3], words, label='chunk 4 (m)')
        info = [r for r in logs.records if r.levelname == 'INFO']
        self.assertEqual(len(info), 1)
        summary = info[0].getMessage()
        for part in ('chunk 4 (m)', '3 segments', '2 refined', '1 no word match'):
            self.assertIn(part, summary)
        debug = ' '.join(r.getMessage() for r in logs.records if r.levelname == 'DEBUG')
        self.assertIn('words match', debug)

    def test_missing_words_are_one_info_line(self):
        with self.assertLogs(adapters._log, 'INFO') as logs:
            adapters.refine_segments_with_words(SEGMENTS, None, label='chunk 0')
        self.assertEqual(len(logs.records), 1)
        self.assertIn('no word timestamps', logs.records[0].getMessage())

    def test_refinement_never_makes_a_valid_segment_degenerate(self):
        # Brute force: refinement itself cannot produce end <= start (the overlap clamp in
        # clamp_and_validate_segments is what can, see test_subtitle_cloud_chunks).
        import random
        rng = random.Random(7)
        for _ in range(300):
            segs, words, t = [], [], 0.0
            for i in range(rng.randint(1, 6)):
                t += rng.choice([0.0, 0.0, 0.3, 1.0])
                length = rng.choice([0.04, 0.1, 1.0, 2.0])
                start = t if rng.random() < 0.5 else float(round(t))
                end = start + length if rng.random() < 0.5 else float(round(start + length)) or start + 1.0
                if end <= start:
                    end = start + 1.0
                text = ' ' + 'あいうえお'[: rng.randint(1, 5)] + str(i)
                segs.append({'id': i, 'start': start, 'end': end, 'text': text})
                ws = start + rng.uniform(-0.3, 1.0)
                for ch in text.strip():
                    words.append(_w(ch, ws, ws + rng.choice([0.0, 0.02, 0.2])))
                    ws += 0.1
                t = max(t, end)
            out = adapters.refine_segments_with_words(segs, words)
            for before, after in zip(segs, out):
                self.assertGreater(after['end'], after['start'], (segs, words, out))


class CloudAsrRequestTests(unittest.TestCase):
    def _run(self, reply):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / 'source.wav'
            with wave.open(str(source), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b'\0\0' * 16000 * 8)
            row = {'remote_model': 'whisper', 'base_url': 'https://example.org/v1'}
            payloads = []

            def fake(row_arg, route, payload, key, content_type):
                payloads.append(payload)
                return reply

            with patch.object(adapters, '_request', side_effect=fake):
                result = adapters._cloud_asr(row, source, out)
            return result, payloads, out

    def test_request_asks_for_segment_and_word(self):
        _, payloads, _ = self._run({'segments': SEGMENTS, 'words': WORDS})
        body = payloads[0]
        self.assertEqual(body.count(b'name="timestamp_granularities[]"'), 2)
        self.assertIn(b'\r\n\r\nword\r\n', body)
        self.assertIn(b'\r\n\r\nsegment\r\n', body)

    def test_result_uses_word_times(self):
        result, _, _ = self._run({'segments': SEGMENTS, 'words': WORDS})
        self.assertEqual([(s['start'], s['end']) for s in result['segments']],
                         [(0.94, 2.26), (2.26, 3.11), (4.12, 5.01)])

    def test_timing_version_in_checkpoint_identity_and_slice_cache(self):
        from oh_streamer.queue import subtitle_scheduler
        self.assertNotEqual(subtitle_scheduler.asr_cache_params()['pipeline'], 'slice-asr-v1')
        self.assertGreaterEqual(adapters.ASR_TIMING_VERSION, 2)


def _abs_seg(start, end, text, words=None):
    seg = {'start': start, 'end': end, 'text': text}
    if words is not None:
        seg['words'] = words
    return seg


class PrepareSliceCuesWordTimingTests(unittest.TestCase):
    def _prep(self, segs):
        from oh_streamer.subtitles.slice_results import _prepare_slice_cues
        return _prepare_slice_cues(segs, config={'sentence': {'short_policy': 'keep'}})

    def test_contiguous_segments_start_at_their_late_words(self):
        segs = [_abs_seg(0.0, 44.32, 'こんにちは。', [_w('こんにちは', 1.0, 1.9)]),
                _abs_seg(44.32, 65.16, 'お疲れさまでした。', [_w('お疲れ', 60.08, 61.0), _w('さまでした', 61.0, 64.94)])]
        cues = self._prep(segs)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0]['start'], 1.0, places=2)
        self.assertAlmostEqual(cues[0]['end'], 2.15, places=2)
        self.assertAlmostEqual(cues[1]['start'], 60.08, places=2)       # not 44.32 (music)
        self.assertAlmostEqual(cues[1]['end'], 65.19, places=2)

    def test_stretched_first_word_is_capped(self):
        cues = self._prep([_abs_seg(79.0, 90.0, 'なんでやねん', [_w('なん', 79.06, 84.36), _w('でやねん', 84.36, 85.5)])])
        self.assertAlmostEqual(cues[0]['start'], 84.36 - 0.72, places=2)

    def test_word_cue_is_clipped_to_next_cue_start(self):
        segs = [_abs_seg(0.0, 3.0, 'あいうえお', [_w('あいうえお', 1.0, 2.0)]),
                _abs_seg(3.0, 4.0, 'かきくけこ', [_w('かきくけこ', 2.1, 3.0)])]
        cues = self._prep(segs)
        self.assertLessEqual(cues[0]['end'], cues[1]['start'] + 1e-9)
        self.assertGreaterEqual(cues[0]['end'] - cues[0]['start'], 0.3)

    def test_long_segment_splits_at_spaces_with_own_word_times(self):
        text = '今日はとても良い天気ですね みなさんこんにちは 今から料理を始めます よろしくお願いします'
        words = [_w('今日はとても良い天気ですね', 10.0, 12.0), _w('みなさんこんにちは', 12.2, 13.5),
                 _w('今から料理を始めます', 20.0, 22.0), _w('よろしくお願いします', 22.2, 24.0)]
        cues = self._prep([_abs_seg(0.0, 24.0, text, words)])
        self.assertGreater(len(cues), 1)
        self.assertEqual(''.join(c['text'] for c in cues).replace(' ', ''), text.replace(' ', ''))
        for c in cues:                                       # no cut inside a word group
            self.assertNotIn(c['text'].strip()[:1], ' ')
        self.assertAlmostEqual(cues[0]['start'], 10.0, places=1)
        later = [c for c in cues if '料理' in c['text']][0]
        self.assertAlmostEqual(later['start'], 20.0, places=1)         # gap kept: second half is not at 10+x
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a['end'], b['start'] + 1e-6)

    def test_segment_without_words_is_unchanged(self):
        from oh_streamer.subtitles.sentences import sentence_cues
        from oh_streamer.subtitles.core import Segment
        seg = _abs_seg(5.0, 9.0, 'こんにちは。お元気ですか。')
        cues = self._prep([seg])
        want = sentence_cues(Segment(start=5.0, end=9.0, text=seg['text']))
        self.assertEqual([(c['start'], c['end'], c['text']) for c in cues],
                         [(round(w.start, 3), round(w.end, 3), w.text) for w in want])

    def test_words_that_do_not_spell_the_text_fall_back(self):
        cues = self._prep([_abs_seg(5.0, 9.0, 'こんにちは。', [_w('違う', 20.0, 21.0)])])
        self.assertEqual((cues[0]['start'], cues[0]['end']), (5.0, 9.0))


class HanSafeSplitTests(unittest.TestCase):
    def test_long_cue_is_never_cut_between_han_characters(self):
        from oh_streamer.subtitles.core import Segment, render_srt
        text = 'AI機器人假新聞' * 3
        srt = render_srt([Segment(start=0.0, end=20.0, text=text)], max_line_chars=40, max_duration=7.0)
        body = ''.join(line for line in srt.splitlines() if line and not line.isdigit() and '-->' not in line)
        self.assertEqual(body, text)
        import re
        blocks = [b.splitlines()[2:] for b in srt.strip().split(chr(10) * 2)]
        for lines in blocks:
            joined = ''.join(lines)
            self.assertNotRegex(joined[:1] + joined[-1:], r'^$')
        for a, b in zip(blocks, blocks[1:]):
            left, right = ''.join(a)[-1], ''.join(b)[0]
            self.assertFalse('一' <= left <= '鿿' and '一' <= right <= '鿿', (left, right))

    def test_unsplittable_run_is_capped_not_cut(self):
        from oh_streamer.subtitles.core import Segment, render_srt
        srt = render_srt([Segment(start=0.0, end=30.0, text='機器人假新聞真相')], max_line_chars=40, max_duration=7.0)
        self.assertEqual(srt.count('-->'), 1)
        self.assertIn('00:00:07,000', srt)

    def test_splits_at_punctuation_when_available(self):
        from oh_streamer.subtitles.core import Segment, render_srt
        srt = render_srt([Segment(start=0.0, end=12.0, text='機器人假新聞，真相是什麼')], max_line_chars=40, max_duration=7.0)
        self.assertEqual(srt.count('-->'), 2)
        self.assertIn('機器人假新聞，', srt)


class CreditPatternTests(unittest.TestCase):
    def test_by_credit_prefix_is_blacklisted(self):
        from oh_streamer.subtitles import core
        for text in ('by H.おやすみなさい', 'By Someone.'):
            self.assertTrue(core._blacklisted(text, speech_present=True, blacklist=core._DEFAULT_BLACKLIST), text)
        for text in ('ありがとうby me', 'おやすみなさい'):
            self.assertFalse(core._blacklisted(text, speech_present=True, blacklist=core._DEFAULT_BLACKLIST), text)


class MtCacheParamsTests(unittest.TestCase):
    def test_cue_timing_version_is_part_of_the_cache_params(self):
        from oh_streamer.queue.subtitle_scheduler import mt_cache_params
        self.assertEqual(mt_cache_params()['cue_timing'], 'words-v4')


class PrepareCueFilterTests(unittest.TestCase):
    def _w(self, text, start, end):
        return {'word': text, 'start': start, 'end': end}

    def test_short_word_timed_reply_is_kept_by_its_reply_span(self):
        from oh_streamer.subtitles.slice_results import _prepare_slice_cues
        # the stored ASR times are already word-tight: 「失礼」 spans 0.68 s, measured from the previous cue it stays
        segs = [{'start': 65.9, 'end': 67.2, 'text': '気持ちよくないですか',
                 'words': [self._w('気持ちよく', 65.9, 66.6), self._w('ないですか', 66.6, 67.0)]},
                {'start': 68.92, 'end': 69.6, 'text': '失礼',
                 'words': [self._w('失', 68.92, 69.42), self._w('礼', 69.42, 69.60)]}]
        cues = _prepare_slice_cues(segs)
        self.assertEqual([c['text'] for c in cues], ['気持ちよくないですか', '失礼'])
        self.assertGreater(cues[1]['start'], 68.9)

    def test_isolated_short_non_whitelisted_cue_is_still_filtered(self):
        from oh_streamer.subtitles.slice_results import _prepare_slice_cues
        segs = [{'start': 10.0, 'end': 10.3, 'text': 'ぷ', 'words': [self._w('ぷ', 10.0, 10.3)]}]
        self.assertEqual(_prepare_slice_cues(segs), [])

    def test_credit_line_with_words_is_dropped(self):
        from oh_streamer.subtitles.slice_results import _prepare_slice_cues
        segs = [{'start': 240.0, 'end': 247.0, 'text': 'by H.おやすみなさい',
                 'words': [self._w('by', 240.9, 241.1), self._w('H.', 241.1, 241.5),
                           self._w('おやすみなさい', 241.5, 247.3)]}]
        self.assertEqual(_prepare_slice_cues(segs), [])


class PauseSplitTests(unittest.TestCase):
    def _cues(self, text, words):
        from oh_streamer.subtitles.slice_results import _prepare_slice_cues
        segs = [{'start': words[0][1], 'end': words[-1][2], 'text': text,
                 'words': [{'word': w, 'start': a, 'end': b} for w, a, b in words]}]
        return _prepare_slice_cues(segs, config={'sentence': {'short_policy': 'keep'}})

    def test_long_pause_between_speakers_splits_the_cue(self):
        cues = self._cues('あと5時間かどうされましたか', [
            ('あと', 85.6, 85.7), ('5', 85.7, 86.0), ('時間', 86.0, 86.3), ('か', 86.3, 86.7),
            ('どう', 89.5, 89.7), ('され', 89.7, 89.9), ('ましたか', 89.9, 90.5)])
        self.assertEqual([c['text'] for c in cues], ['あと5時間か', 'どうされましたか'])
        self.assertLess(cues[0]['end'], cues[1]['start'])     # the screen is empty during the pause

    def test_short_side_stays_attached(self):
        cues = self._cues('私新中署の捜査一課です', [
            ('私', 171.0, 171.1), ('新中署', 172.0, 172.6), ('の', 172.6, 172.7),
            ('捜査一課', 172.7, 173.4), ('です', 173.4, 173.7)])
        self.assertEqual([c['text'] for c in cues], ['私新中署の捜査一課です'])

    def test_short_reply_may_stand_alone(self):
        cues = self._cues('はい分かりました', [
            ('はい', 10.0, 10.2), ('分かり', 11.2, 11.6), ('ました', 11.6, 12.0)])
        self.assertEqual([c['text'] for c in cues], ['はい', '分かりました'])

    def test_short_pause_does_not_split(self):
        cues = self._cues('嘘かのことか見分けのつかぬ', [
            ('嘘かのことか', 7.5, 8.6), ('見分けのつかぬ', 9.2, 10.0)])
        self.assertEqual(len(cues), 1)


if __name__ == '__main__':
    unittest.main()
