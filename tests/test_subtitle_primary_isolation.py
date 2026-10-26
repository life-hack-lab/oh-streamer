"""Comparison subtitles must not change the primary subtitle recipe or text."""
import unittest

from oh_streamer.subtitles import combos


class PrimaryIsolationTests(unittest.TestCase):

    def test_normalization_is_idempotent_and_preserves_single_model(self):
        row = combos.normalize_combo(dict(id="primary", label="主字幕",
                                         asr=dict(mode="single", model="anime-whisper")))
        self.assertEqual(row["asr"]["model"], "anime-whisper")
        self.assertEqual(combos.normalize_combo(row), row)




if __name__ == "__main__":
    unittest.main()
