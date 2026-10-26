"""Unit tests for Maximum Overlap Clamping and Multi-Chunk Joining (core.py & asr.py)."""

import unittest
from oh_streamer.subtitles.core import (
    Segment,
    assign_segments_to_intervals,
    join_assigned_segments,
)
from oh_streamer.subtitles.asr import map_asr_chunks_to_intervals


class MaxOverlapClampingTests(unittest.TestCase):
    def test_prevent_trailing_leakage_1_15_ticket_booking(self):
        """Preceding sentence ending 0.05s into next sentence interval must NOT leak.

        Interval 1: [70.0, 75.0] (Sentence 1: Previous dialogue)
        Interval 2: [75.1, 78.0] (Sentence 2: Flight/hotel booking line)

        ASR Chunk 1: [70.0, 75.15] "来週の大阪出張の件ですが" (overlaps Int1 by 5.0s, Int2 by 0.05s)
        ASR Chunk 2: [75.15, 78.0] "ホテルと飛行機の予約をお願いしていい？" (overlaps Int2 by 2.85s)
        """
        intervals = [
            (70.0, 75.0),
            (75.1, 78.0),
        ]
        segments = [
            {"start": 70.0, "end": 75.15, "text": "来週の大阪出張の件ですが"},
            {"start": 75.15, "end": 78.0, "text": "ホテルと飛行機の予約をお願いしていい？"},
        ]

        assigned = assign_segments_to_intervals(segments, intervals, snap_threshold=0.8)
        self.assertEqual(len(assigned), 2)

        # Interval 1 gets chunk 1 (5.0s overlap > 0.05s overlap)
        self.assertEqual(len(assigned[0]), 1)
        self.assertEqual(assigned[0][0]["text"], "来週の大阪出張の件ですが")

        # Interval 2 gets chunk 2 (2.85s overlap > 0s overlap)
        self.assertEqual(len(assigned[1]), 1)
        self.assertEqual(assigned[1][0]["text"], "ホテルと飛行機の予約をお願いしていい？")

    def test_multi_chunk_joining_preserves_all_chunks(self):
        """Whisper splitting a long sentence into multiple continuous chunks must keep all chunks.

        Interval: [100.0, 108.0]
        ASR chunks: 3 consecutive chunks
        """
        intervals = [(100.0, 108.0)]
        chunks = [
            {"start": 100.2, "end": 102.5, "text": "部長、私こないだ、"},
            {"start": 102.6, "end": 105.0, "text": "酒井さんと更衣室が一緒になったんですけど…"},
            {"start": 105.1, "end": 107.8, "text": "あの子、脱ぐとすごいんですよぉ。"},
        ]

        assigned = assign_segments_to_intervals(chunks, intervals)
        self.assertEqual(len(assigned), 1)
        # Strictly preserve all 3 chunks in chronological order, NEVER taking only chunks[0]!
        self.assertEqual(len(assigned[0]), 3)
        self.assertEqual(assigned[0][0]["text"], "部長、私こないだ、")
        self.assertEqual(assigned[0][1]["text"], "酒井さんと更衣室が一緒になったんですけど…")
        self.assertEqual(assigned[0][2]["text"], "あの子、脱ぐとすごいんですよぉ。")

        joined = join_assigned_segments(assigned[0])
        self.assertEqual(
            joined,
            "部長、私こないだ、 酒井さんと更衣室が一緒になったんですけど… あの子、脱ぐとすごいんですよぉ。",
        )

    def test_nearest_snapping_within_threshold(self):
        """If overlap is 0.0 but distance is within snap_threshold (< 0.8s), snap to nearest."""
        intervals = [
            (10.0, 14.0),
            (16.0, 20.0),
        ]
        # Segment from 15.6 to 15.9 (distance to Interval 2 is 0.1s < 0.8s)
        seg = Segment(start=15.6, end=15.9, text="そうですね")

        assigned = assign_segments_to_intervals([seg], intervals, snap_threshold=0.8)
        self.assertEqual(len(assigned[0]), 0)
        self.assertEqual(len(assigned[1]), 1)
        self.assertEqual(assigned[1][0].text, "そうですね")

    def test_rejection_beyond_threshold(self):
        """If overlap is 0.0 and distance exceeds snap_threshold (>= 0.8s), do not assign."""
        intervals = [
            (10.0, 14.0),
            (25.0, 30.0),
        ]
        # Segment from 18.0 to 20.0 (distance to closest interval is 4.0s > 0.8s)
        seg = Segment(start=18.0, end=20.0, text="孤立した音声")

        assigned = assign_segments_to_intervals([seg], intervals, snap_threshold=0.8)
        self.assertEqual(len(assigned[0]), 0)
        self.assertEqual(len(assigned[1]), 0)

    def test_center_proximity_tie_breaking(self):
        """When overlaps are equal, choose interval whose center is closer to segment center."""
        intervals = [
            (10.0, 14.0),  # center = 12.0
            (13.0, 17.0),  # center = 15.0
        ]
        # Segment [13.0, 14.0]: overlaps Interval 1 by 1.0s, Interval 2 by 1.0s
        # Segment center = 13.5. Distance to 12.0 is 1.5; distance to 15.0 is 1.5.
        # Shift slightly toward Interval 2: [13.2, 14.2]
        # Overlap with Int 1: min(14.0, 14.2) - max(10.0, 13.2) = 0.8s
        # Overlap with Int 2: min(17.0, 14.2) - max(13.0, 13.2) = 1.0s
        seg = {"start": 13.2, "end": 14.2, "text": "テスト"}
        assigned = assign_segments_to_intervals([seg], intervals)
        self.assertEqual(len(assigned[1]), 1)

    def test_map_asr_chunks_to_intervals_wrapper(self):
        """Verify map_asr_chunks_to_intervals exported in asr.py."""
        intervals = [(0.0, 5.0)]
        chunks = [
            {"start": 0.5, "end": 2.0, "text": "前半"},
            {"start": 2.2, "end": 4.5, "text": "後半"},
        ]
        mapped = map_asr_chunks_to_intervals(chunks, intervals)
        self.assertEqual(len(mapped[0]), 2)
        self.assertEqual(join_assigned_segments(mapped[0]), "前半 後半")

    def test_inverted_timestamps_normalization(self):
        """Intervals or segments with inverted timestamps must be safely normalized."""
        intervals = [(10.0, 5.0)]  # Inverted: start=10, end=5
        chunks = [{"start": 8.0, "end": 6.0, "text": "倒序切片"}]  # Inverted: start=8, end=6
        assigned = assign_segments_to_intervals(chunks, intervals)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(len(assigned[0]), 1)
        self.assertEqual(assigned[0][0]["text"], "倒序切片")


if __name__ == "__main__":
    unittest.main()
