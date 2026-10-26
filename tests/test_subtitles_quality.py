"""CPU quality regression tests; no model downloads or CUDA required."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles.asr import _WhisperConfidence, gap_inference_groups, segment_from_record
from oh_streamer.subtitles.core import Segment, clean_segments, _segments_for_srt
from oh_streamer.subtitles.quality import is_onomatopoeia, key_text, quality_fingerprint
from oh_streamer.subtitles.sentences import sentence_cues


class QualityTests(unittest.TestCase):
    def test_cache_reclassifies_f7_without_reloading_asr_and_gap_changes_raw_identity(self):
        from oh_streamer.subtitles import asr, runtime
        keys = []
        cached = {"segments": [{"start": 0, "end": 2, "text": "ここに来て", "gap": True,
                                 "avg_logprob": -.8, "compression_ratio": .8}],
                  "raw_segments": [], "speech_intervals": [], "metrics": {}}
        def cache(path, key):
            keys.append(key)
            return cached
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(runtime, "local_model_fingerprint", return_value="model"), \
                patch.object(runtime, "source_identity", return_value=("source", {})), \
                patch.object(asr, "_cache_matches", side_effect=cache), \
                patch.object(asr, "_make_pipeline") as loader:
            first = asr.run_asr("fixture.wav", folder, cache_dir=Path(folder) / "cache")
            second = asr.run_asr("fixture.wav", folder, cache_dir=Path(folder) / "cache",
                                quality_config={"f7": {"avg_logprob_min": -.9}})
            asr.run_asr("fixture.wav", folder, cache_dir=Path(folder) / "cache", gap_fill=False)
            loader.assert_not_called()
        self.assertTrue(first["segments"][0]["low_confidence"])
        self.assertFalse(second["segments"][0]["low_confidence"])
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])

    def test_f7_boundaries_missing_confidence_and_vad_pass(self):
        base = Segment(0, 2, "ここに来てください", gap=True,
                       avg_logprob=-.7, no_speech_prob=.0999, compression_ratio=1.1999)
        self.assertEqual(clean_segments([base]), [base])
        for change in ({"avg_logprob": -.7001}, {"no_speech_prob": .1}, {"compression_ratio": 1.2}):
            cue = replace(base, **change)
            self.assertEqual(clean_segments([cue]), [])
            full = clean_segments([cue], include_low_confidence=True)
            self.assertTrue(full[0].low_confidence)
            self.assertEqual(clean_segments([replace(cue, gap=False)])[0].text, cue.text)
        self.assertEqual(len(clean_segments([replace(base, avg_logprob=None, no_speech_prob=None,
                                                   compression_ratio=None)])), 1)
        self.assertEqual(clean_segments([base], vad_intervals=[]), [base])

    def test_hard_drops_apply_to_both_tracks_and_exact_sentence_only(self):
        for text in ("おはようございます。", "作詞・作曲・編曲 初音ミク", "ご視聴ありがとうございました",
                     "こく、こく", "そこに行く行く行く行くよ", "同じ文同じ文同じ文"):
            for gap in (False, True):
                self.assertEqual(clean_segments([Segment(0, 2, text, gap=gap)],
                                                include_low_confidence=True, show_interjections=True), [])
        self.assertEqual(len(clean_segments([Segment(0, 2, "おはようございます、今日は暑いですね")])), 1)
        self.assertEqual(key_text("コク、コク（説明）"), "こくこく")

    def test_metadata_survives_sentence_and_display_splitting(self):
        source = Segment(0, 8, "ここに来て。次はあそこへ。", gap=True, low_confidence=True,
                         compression_ratio=.8, avg_logprob=-.8, alt_text="代替")
        for cue in sentence_cues(source) + _segments_for_srt([source], max_line_chars=4,
                                                            min_duration=.7, max_duration=7):
            self.assertTrue(cue.gap)
            self.assertTrue(cue.low_confidence)
            self.assertEqual(cue.compression_ratio, .8)
            self.assertEqual(cue.alt_text, "代替")
        restored = segment_from_record({"start": 1, "end": 2, "text": "言葉", "src": "gap",
                                        "avg_logprob": -.9, "compression_ratio": .7})
        self.assertTrue(restored.gap)
        self.assertEqual(restored.compression_ratio, .7)

    def test_simple_onomatopoeia_configuration_and_fingerprint(self):
        self.assertTrue(is_onomatopoeia("ちゅっ、ちゅぷっ……"))
        self.assertFalse(is_onomatopoeia("ちゅっ、好きだよ"))
        self.assertFalse(is_onomatopoeia(""))
        self.assertNotEqual(quality_fingerprint(), quality_fingerprint({"version": "changed"}))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy runtime unavailable")
    def test_gap_windows_rms_coverage_and_short_tail_match_funnel(self):
        import numpy as np
        audio = np.full(26 * 16000, .01, dtype=np.float32)
        groups = gap_inference_groups([{"start": 3, "end": 5}], audio)
        self.assertEqual(groups, [(0, 3), (5, 15), (15, 25)])
        self.assertEqual(gap_inference_groups([], np.zeros(5 * 16000, dtype=np.float32)), [])
        self.assertEqual(gap_inference_groups([], audio,
                         config={"gap_fill": {"enabled": False}}), [])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch runtime unavailable")
    def test_confidence_scores_text_plus_eot_and_sot_no_speech_without_changing_generate(self):
        import torch
        from types import SimpleNamespace
        class Model:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.zeros(1))
                self.model = SimpleNamespace(encoder=lambda features: features)
            def parameters(self):
                return iter([self.parameter])
            def generate(self, input_features):
                return torch.tensor([[10, 11, 12, 13, 1, 2, 5]])
            def __call__(self, encoder_outputs, decoder_input_ids):
                logits = torch.zeros((1, decoder_input_ids.shape[1], 16))
                logits[0, 0, 14] = 1
                logits[0, 3, 1] = 2
                logits[0, 4, 2] = 3
                logits[0, 5, 5] = 4
                self.logits = logits
                return SimpleNamespace(logits=logits)
        model = Model()
        mapping = dict(zip(("<|startoftranscript|>", "<|ja|>", "<|transcribe|>",
                            "<|notimestamps|>", "<|nospeech|>"), (10, 11, 12, 13, 14)))
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda text: mapping.get(text),
                                    eos_token_id=5, unk_token_id=-1)
        original = model.generate
        confidence = _WhisperConfidence(torch, model, SimpleNamespace(tokenizer=tokenizer))
        output = model.generate(input_features=torch.zeros((1, 2, 3)))
        self.assertEqual(output[0].tolist(), [10, 11, 12, 13, 1, 2, 5])
        values = confidence.score()
        expected = model.logits[0].log_softmax(-1)
        self.assertAlmostEqual(values["avg_logprob"], float((expected[3, 1] + expected[4, 2] + expected[5, 5]) / 3))
        self.assertAlmostEqual(values["no_speech_prob"], float(expected[0, 14].exp()))
        confidence.close()
        self.assertEqual(model.generate, original)


if __name__ == "__main__":
    unittest.main()
