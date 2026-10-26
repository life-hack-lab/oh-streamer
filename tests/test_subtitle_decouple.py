"""Unit tests for Auto Clause & Speaker Turn Decoupling (decouple.py)."""

import unittest
import numpy as np

from oh_streamer.subtitles.core import Segment
from oh_streamer.subtitles.decouple import (
    AutoClauseDecoupler,
    DecoupleSegment,
    decouple_segments,
)


def synthesize_audio_with_pause(
    total_duration: float,
    pause_start: float,
    pause_end: float,
    sample_rate: int = 16000,
    signal_amp: float = 0.5,
    pause_amp: float = 0.001,
) -> np.ndarray:
    """Helper to synthesize test audio with tone bursts separated by a dip/pause."""
    n_samples = int(total_duration * sample_rate)
    t = np.linspace(0, total_duration, n_samples, endpoint=False)
    # 440Hz sine wave tone
    audio = signal_amp * np.sin(2 * np.pi * 440.0 * t)
    
    # Silence/dip mask
    pause_mask = (t >= pause_start) & (t <= pause_end)
    audio[pause_mask] = pause_amp * np.random.randn(np.sum(pause_mask))
    return audio.astype(np.float32)


class AutoClauseDecouplerTests(unittest.TestCase):
    def setUp(self):
        self.decoupler = AutoClauseDecoupler(
            sample_rate=16000,
            min_cue_duration=0.8,
            min_fast_pace_duration=0.65,
            max_cps=15.0,
            min_chars=3,
        )

    def test_case_a_emergency_shout_decoupling(self):
        """Case A: Panicked shout followed by pleading line.

        Text: 「救急車、救急車、救急車だ！ あなたの水流力をもう一度貸してほしい」
        The shout occupies ~1.5s, pause is ~300ms (1.5s ~ 1.8s), followed by pleading line to 5.2s.
        """
        text = "救急車、救急車、救急車だ！ あなたの水流力をもう一度貸してほしい"
        start = 0.0
        end = 5.2
        audio = synthesize_audio_with_pause(
            total_duration=end,
            pause_start=1.5,
            pause_end=1.8,
            sample_rate=16000,
        )
        seg = Segment(start=start, end=end, text=text)

        result = self.decoupler.decouple(seg, audio)
        self.assertEqual(len(result), 2)
        part1, part2 = result[0], result[1]

        # Part 1 should contain the panicked shout
        self.assertEqual(part1.text, "救急車、救急車、救急車だ！")
        self.assertAlmostEqual(part1.start, 0.0, places=2)
        # Split time should be near center of dip (1.65s)
        self.assertGreaterEqual(part1.end, 1.5)
        self.assertLessEqual(part1.end, 1.85)

        # Part 2 should contain the pleading sentence
        self.assertEqual(part2.text, "あなたの水流力をもう一度貸してほしい")
        self.assertGreaterEqual(part2.start, part1.end)
        self.assertAlmostEqual(part2.end, 5.2, places=2)

    def test_case_b_izakaya_rapid_dialogue_turn(self):
        """Case B: Rapid speaker alternation in izakaya.

        Nakata: 「いやいやいや社長の人柄が私を動かしたんですよ」
        Osaka President: 「ほんま人たらしやわ！ははは！」
        Total duration: 6.0s. First speaker ends around 3.2s, 250ms breathing pause (3.2s ~ 3.45s).
        """
        text = "いやいやいや社長の人柄が私を動かしたんですよ。 ほんま人たらしやわ！ははは！"
        start = 0.0
        end = 6.0
        audio = synthesize_audio_with_pause(
            total_duration=end,
            pause_start=3.2,
            pause_end=3.45,
            sample_rate=16000,
        )
        seg = DecoupleSegment(start=start, end=end, text=text)

        result = self.decoupler.decouple(seg, audio)
        self.assertEqual(len(result), 2)

        part1, part2 = result[0], result[1]
        self.assertEqual(part1.text, "いやいやいや社長の人柄が私を動かしたんですよ。")
        self.assertEqual(part2.text, "ほんま人たらしやわ！ははは！")
        self.assertAlmostEqual(part1.start, 0.0, places=2)
        self.assertGreaterEqual(part1.end, 3.1)
        self.assertLessEqual(part1.end, 3.5)
        self.assertAlmostEqual(part2.end, 6.0, places=2)

    def test_case_c_sokuon_and_compound_atomicity_protection(self):
        """Case C: Strict protection against cutting sokuon (促音) and compound words.

        Text with sokuon: 「やっぱり喫茶店に行こう」
        Even if there's an acoustic silence inside 'やっぱり' or between '喫' and '茶',
        cutting between word boundaries must be strictly forbidden!
        """
        text = "やっぱり喫茶店に行こう"
        # Verify word boundary rejects cutting at 'やっ|ぱり' or '喫|茶'
        char_idx_yappari = 2  # after 'っ'
        self.assertFalse(self.decoupler._is_valid_word_boundary(text, char_idx_yappari))
        self.assertFalse(self.decoupler._is_valid_word_boundary(text, 1))  # before 'っ'

        # Kanji compound atomicity: between 喫 and 茶
        char_idx_kissaten = text.index("茶")
        self.assertFalse(self.decoupler._is_valid_word_boundary(text, char_idx_kissaten))

        # Overall decouple should refuse to cut inside words even if audio has a dip
        audio = synthesize_audio_with_pause(
            total_duration=3.0,
            pause_start=0.5,
            pause_end=0.8,
            sample_rate=16000,
        )
        seg = DecoupleSegment(start=0.0, end=3.0, text=text)
        result = self.decoupler.decouple(seg, audio)
        # Cannot cut without a valid linguistic boundary!
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, text)

    def test_case_d_suppress_short_sentence_cutting(self):
        """Case D: Ultra-short sentence (< 0.8s) must suppress cutting."""
        text = "はい！ わかりました！"
        seg = Segment(start=1.0, end=1.6, text=text)  # duration = 0.6s (< 0.8s)
        audio = synthesize_audio_with_pause(
            total_duration=2.0,
            pause_start=1.2,
            pause_end=1.4,
            sample_rate=16000,
        )
        result = self.decoupler.decouple(seg, audio)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, text)

    def test_boundary_conditions_no_dip_suppression(self):
        """If there is no acoustic dip, keep original sentence intact (safety fallback)."""
        text = "救急車、救急車、救急車だ！ あなたの水流力をもう一度貸してほしい"
        # Audio with constant high amplitude (no silence / no dip)
        n_samples = int(5.0 * 16000)
        t = np.linspace(0, 5.0, n_samples, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        seg = Segment(start=0.0, end=5.0, text=text)
        result = self.decoupler.decouple(seg, audio)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, text)

    def test_boundary_conditions_excessive_cps_suppression(self):
        """If cutting would create excessive reading speed (CPS > 15), suppress cutting."""
        # 30 chars in 1.0s -> CPS = 30 > 15
        text = "あ" * 15 + "！" + "い" * 15
        audio = synthesize_audio_with_pause(
            total_duration=2.0,
            pause_start=0.9,
            pause_end=1.1,
            sample_rate=16000,
        )
        # Even with dip, 16 chars in ~1s exceeds max_cps=15
        decoupler = AutoClauseDecoupler(max_cps=10.0)
        seg = DecoupleSegment(start=0.0, end=2.0, text=text)
        result = decoupler.decouple(seg, audio)
        self.assertEqual(len(result), 1)

    def test_convenience_decouple_segments(self):
        """Test decouple_segments on a list of segments."""
        segs = [
            Segment(start=0.0, end=5.2, text="救急車だ！ 助けてくれ！"),
            Segment(start=6.0, end=7.0, text="了解です。"),
        ]
        audio = synthesize_audio_with_pause(
            total_duration=8.0,
            pause_start=1.8,
            pause_end=2.2,
            sample_rate=16000,
        )
        decoupled = decouple_segments(segs, audio)
        # First segment should be decoupled into 2, second stays 1 -> total 3
        self.assertEqual(len(decoupled), 3)
        self.assertEqual(decoupled[0].text, "救急車だ！")
        self.assertEqual(decoupled[1].text, "助けてくれ！")
        self.assertEqual(decoupled[2].text, "了解です。")

    def test_multichannel_audio_support(self):
        """Verify that 2D multichannel audio (both (N, C) and (C, N)) is handled cleanly."""
        text = "救急車だ！ 助けてくれ！"
        mono_audio = synthesize_audio_with_pause(5.2, 1.8, 2.2, sample_rate=16000)

        # Shape (N, 2)
        stereo_nc = np.stack([mono_audio, mono_audio], axis=-1)
        res_nc = self.decoupler.decouple(Segment(0.0, 5.2, text), stereo_nc)
        self.assertEqual(len(res_nc), 2)
        self.assertEqual(res_nc[0].text, "救急車だ！")

        # Shape (2, N)
        stereo_cn = np.stack([mono_audio, mono_audio], axis=0)
        res_cn = self.decoupler.decouple(Segment(0.0, 5.2, text), stereo_cn)
        self.assertEqual(len(res_cn), 2)
        self.assertEqual(res_cn[0].text, "救急車だ！")

    def test_short_clip_rms_envelope_robustness(self):
        """Verify that ultra-short clips (< 5 frames) don't trigger dimension mismatch in RMS."""
        short_audio = np.zeros(480, dtype=np.float32)  # 30ms -> 2 frames
        rms, times = self.decoupler._compute_rms_envelope(short_audio, 0.0, 0.03)
        self.assertEqual(len(rms), len(times))


if __name__ == "__main__":
    unittest.main()
