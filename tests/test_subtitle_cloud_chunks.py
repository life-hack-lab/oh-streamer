"""Unit and contract tests for M2: cloud ASR chunk planning, silence snapping, clamping, and checkpoints."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from oh_streamer.subtitles import cloud_chunks, model_adapters as adapters
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles.cloud_chunks import (
    ChunkCheckpoint,
    clamp_and_validate_segments,
    deduplicate_segments,
    detect_silence_intervals,
    plan_chunks,
    reconstruct_timeline,
)
from oh_streamer.subtitles.model_adapters import ModelAdapterError


class CloudChunksPlanningTests(unittest.TestCase):
    def test_plan_chunks_short_duration_single_chunk(self):
        chunks = plan_chunks(180.0, target_chunk_sec=300.0)
        self.assertEqual(chunks, [(0, 0.0, 180.0)])

    def test_plan_chunks_snaps_to_silence(self):
        # Total 650s, target 300s. Silence is detected at [288.0, 292.0]
        silences = [(288.0, 292.0), (580.0, 585.0)]
        chunks = plan_chunks(650.0, target_chunk_sec=300.0, silence_intervals=silences)
        self.assertEqual(len(chunks), 3)

        # Chunk 0 should snap to midpoint of (288.0, 292.0) = 290.0
        self.assertEqual(chunks[0], (0, 0.0, 290.0))
        # Chunk 1 starts at 290.0. Ideal cut is 290 + 300 = 590. Silence at 580~585 has midpoint 582.5
        self.assertEqual(chunks[1], (1, 290.0, 582.5))
        # Chunk 2 finishes the rest
        self.assertEqual(chunks[2], (2, 582.5, 650.0))

    def test_plan_chunks_fallback_hard_cut_with_overlap(self):
        # No silences at all
        chunks = plan_chunks(650.0, target_chunk_sec=300.0, overlap_sec=2.0, silence_intervals=[])
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], (0, 0.0, 302.0))
        self.assertEqual(chunks[1], (1, 300.0, 602.0))
        self.assertEqual(chunks[2], (2, 600.0, 650.0))


class ClampingAndConfidenceTests(unittest.TestCase):
    def test_clamp_minor_overlap_and_end_overshoot(self):
        raw = [
            {"start": 0.0, "end": 1.0, "text": "你好", "avg_logprob": -0.3, "no_speech_prob": 0.01},
            # Starts at 0.95 (overlap 0.05s <= 0.1s tolerance) -> clamped to 1.0
            {"start": 0.95, "end": 2.0, "text": "世界", "avg_logprob": -1.5, "no_speech_prob": 0.7},
            # Overshoots chunk duration 3.0 by 0.04s -> clamped to 3.0
            {"start": 2.0, "end": 3.04, "text": "再見"},
        ]
        clamped = clamp_and_validate_segments(raw, chunk_duration=3.0)
        self.assertEqual(len(clamped), 3)
        self.assertEqual(clamped[0]["start"], 0.0)
        self.assertEqual(clamped[0]["end"], 1.0)
        self.assertFalse(clamped[0]["low_confidence"])

        self.assertEqual(clamped[1]["start"], 1.0)
        self.assertEqual(clamped[1]["end"], 2.0)
        self.assertTrue(clamped[1]["low_confidence"])  # avg_logprob < -1.0 or no_speech_prob > 0.6

        self.assertEqual(clamped[2]["start"], 2.0)
        self.assertEqual(clamped[2]["end"], 3.0)

    def test_severe_overlap_is_clamped_and_counted(self):
        raw = [
            {"start": 0.0, "end": 1.0, "text": "你好"},
            {"start": 0.5, "end": 2.0, "text": "世界"},  # 0.5s overlap >> 0.1s: clamped (sample video A), not fatal
        ]
        clamped = clamp_and_validate_segments(raw, chunk_duration=3.0)
        self.assertEqual([(s["start"], s["end"]) for s in clamped], [(0.0, 1.0), (1.0, 2.0)])
        # a response that overlaps everywhere is still rejected (repair budget)
        raw = [{"start": float(i), "end": i + 2.0, "text": "x"} for i in range(6)]
        with self.assertRaises(ModelAdapterError) as ctx:
            clamp_and_validate_segments(raw, chunk_duration=10.0)
        self.assertEqual(ctx.exception.kind, "bad_response")

    def test_deduplicate_segments_on_boundary(self):
        segs = [
            {"start": 0.0, "end": 10.0, "text": "第一段話"},
            {"start": 9.5, "end": 15.0, "text": "第一段話"},  # Duplicate from overlap chunk
            {"start": 15.2, "end": 20.0, "text": "第二段話"},
        ]
        deduped = deduplicate_segments(segs)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["text"], "第一段話")
        self.assertEqual(deduped[0]["end"], 15.0)
        self.assertEqual(deduped[1]["text"], "第二段話")


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


class DegenerateSegmentRepairTests(unittest.TestCase):
    """sample video A: one zero-length segment used to fail the whole 5-minute chunk (both Groq models, 3 attempts each)."""

    def test_overlap_clamp_can_empty_a_short_segment(self):
        # [9.98, 10.05] after a segment ending at 10.05: the minor-overlap clamp moves start to 10.05 = end.
        raw = [{"start": 0.0, "end": 10.05, "text": "前の文"},
               {"start": 9.98, "end": 10.05, "text": "ね"},
               {"start": 11.0, "end": 12.0, "text": "次の文"}]
        with self.assertLogs(cloud_chunks._logger, "WARNING") as logs:
            out = clamp_and_validate_segments(raw, chunk_duration=20.0, label="chunk 3 (m)")
        self.assertEqual([s["text"] for s in out], ["前の文", "ね", "次の文"])
        self.assertAlmostEqual(out[1]["start"], 10.05)
        self.assertGreater(out[1]["end"], out[1]["start"])
        self.assertLessEqual(out[1]["end"], 11.0)
        self.assertEqual(out[1]["timing_repair"], "span")
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        for part in ("chunk 3 (m)", "segment 1", "9.980-10.050", "10.050-10.350", "ね"):
            self.assertIn(part, message)

    def test_end_clamp_can_empty_a_segment(self):
        # start == chunk duration, end overshoots by 0.1: the duration clamp makes end == start.
        raw = [{"start": 0.0, "end": 1.0, "text": "あ"}, {"start": 10.0, "end": 10.1, "text": "いえ"}]
        out = clamp_and_validate_segments(raw, chunk_duration=10.0)
        self.assertEqual("".join(s["text"] for s in out), "あいえ")      # no room after 10.0: merged
        for s in out:
            self.assertGreater(s["end"], s["start"])
            self.assertLessEqual(s["end"], 10.0)

    def test_zero_length_segment_uses_its_word_timestamps(self):
        raw = [{"start": 5.0, "end": 5.0, "text": "ええ", "words": [_w("ええ", 5.1, 5.6)]},
               {"start": 8.0, "end": 9.0, "text": "次"}]
        with self.assertLogs(cloud_chunks._logger, "WARNING") as logs:
            out = clamp_and_validate_segments(raw, chunk_duration=20.0)
        self.assertEqual((out[0]["start"], out[0]["end"], out[0]["timing_repair"]), (5.0, 5.6, "words"))
        self.assertIn("word timestamps", logs.records[0].getMessage())
        self.assertEqual(out[0]["words"], [_w("ええ", 5.1, 5.6)])

    def test_word_repair_stops_at_the_next_segment(self):
        raw = [{"start": 5.0, "end": 5.0, "text": "ええ", "words": [_w("ええ", 5.1, 7.0)]},
               {"start": 5.5, "end": 6.0, "text": "次"}]
        out = clamp_and_validate_segments(raw, chunk_duration=20.0)
        self.assertEqual(out[0]["end"], 5.5)

    def test_zero_length_without_words_gets_a_bounded_span(self):
        raw = [{"start": 5.0, "end": 5.0, "text": "あ"}, {"start": 5.5, "end": 7.0, "text": "次"}]
        out = clamp_and_validate_segments(raw, chunk_duration=20.0)
        self.assertEqual((out[0]["start"], out[0]["end"]), (5.0, 5.3))

        raw = [{"start": 5.0, "end": 5.0, "text": "あ" * 40}, {"start": 6.0, "end": 7.0, "text": "次"}]
        out = clamp_and_validate_segments(raw, chunk_duration=20.0)
        self.assertEqual(out[0]["end"], 6.0)                        # bounded by the next start, not 5 + 2 s

    def test_no_room_merges_text_into_the_next_segment(self):
        raw = [{"start": 0.0, "end": 1.0, "text": "a"},
               {"start": 5.0, "end": 5.0, "text": "b"},
               {"start": 5.05, "end": 6.0, "text": "c"}]
        with self.assertLogs(cloud_chunks._logger, "WARNING") as logs:
            out = clamp_and_validate_segments(raw, chunk_duration=20.0)
        self.assertEqual([(s["start"], s["end"], s["text"]) for s in out], [(0.0, 1.0, "a"), (5.0, 6.0, "bc")])
        self.assertIn("merging into the next", logs.records[0].getMessage())

    def test_no_room_at_the_chunk_end_merges_into_the_previous_segment(self):
        raw = [{"start": 0.0, "end": 9.0, "text": "a"}, {"start": 10.0, "end": 10.0, "text": "b"}]
        out = clamp_and_validate_segments(raw, chunk_duration=10.0)
        self.assertEqual([(s["start"], s["end"], s["text"]) for s in out], [(0.0, 9.0, "ab")])

    def test_only_segment_is_placed_at_the_chunk_end(self):
        out = clamp_and_validate_segments([{"start": 10.0, "end": 10.0, "text": "あ"}], chunk_duration=10.0)
        self.assertEqual((out[0]["start"], out[0]["end"]), (9.7, 10.0))

    def test_a_few_degenerate_segments_are_tolerated_many_are_not(self):
        def reply(bad):
            return [{"start": 2.0 * i, "end": 2.0 * i + (0.0 if i < bad else 1.0), "text": f"t{i}"} for i in range(20)]
        with self.assertLogs(cloud_chunks._logger, "WARNING"):
            out = clamp_and_validate_segments(reply(cloud_chunks.MIN_REPAIR_BUDGET), chunk_duration=60.0)
        self.assertEqual(len(out), 20)
        with self.assertRaises(ModelAdapterError) as ctx:
            clamp_and_validate_segments(reply(cloud_chunks.MIN_REPAIR_BUDGET + 1), chunk_duration=60.0)
        self.assertEqual(ctx.exception.kind, "bad_response")
        self.assertIn("too many degenerate", str(ctx.exception))

    def test_word_repairs_do_not_count_against_the_threshold(self):
        raw = [{"start": 2.0 * i, "end": 2.0 * i, "text": f"t{i}", "words": [_w(f"t{i}", 2.0 * i, 2.0 * i + 0.5)]}
               for i in range(20)]
        with self.assertLogs(cloud_chunks._logger, "WARNING"):
            out = clamp_and_validate_segments(raw, chunk_duration=60.0)
        self.assertTrue(all(s["end"] > s["start"] for s in out))

    def test_valid_segments_keep_their_words_and_stay_unrepaired(self):
        raw = [{"start": 0.0, "end": 1.0, "text": "a", "words": [_w("a", 0.1, 0.9), {"word": "bad"}]}]
        out = clamp_and_validate_segments(raw, chunk_duration=3.0)
        self.assertEqual(out[0]["words"], [_w("a", 0.1, 0.9)])
        self.assertNotIn("timing_repair", out[0])

    def test_reconstruct_timeline_shifts_words(self):
        segs = clamp_and_validate_segments(
            [{"start": 1.0, "end": 2.0, "text": "a", "words": [_w("a", 1.1, 1.9)]}], chunk_duration=5.0)
        out = reconstruct_timeline([(1, 100.0, 105.0)], [segs])
        self.assertEqual((out[0]["start"], out[0]["words"][0]["start"], out[0]["words"][0]["end"]),
                         (101.0, 101.1, 101.9))


def _row(name, kind="asr", local=False):
    return dict(id="user-" + name, kind=kind, provider="local" if local else "openai-compatible",
                connection_id="connection-" + name, remote_model=name, base_url="https://example.org/v1",
                recipe="whisper" if kind == "asr" else "general-llm", model_path="m-" + name)


class ContentErrorClassificationTests(unittest.TestCase):
    BROKEN = [
        (None, 10.0),                                                                    # no segments at all
        ([{"start": 0.0, "text": "x"}], 10.0),                                           # missing field
        ([{"start": 0.0, "end": float("nan"), "text": "x"}], 10.0),                      # non-finite
    ]

    def test_large_overlap_overshoot_and_empty_text_are_clamped_not_fatal(self):
        raw = [{"start": 0.0, "end": 2.0, "text": "a"}, {"start": 1.0, "end": 3.0, "text": "b"},
               {"start": 4.0, "end": 5.0, "text": " "}, {"start": 8.0, "end": 12.0, "text": "c"}]
        with self.assertLogs("oh_streamer.subtitles.cloud_chunks", level="INFO"):
            out = clamp_and_validate_segments(raw, chunk_duration=10.0)
        self.assertEqual([(s["start"], s["end"], s["text"]) for s in out],
                         [(0.0, 2.0, "a"), (2.0, 3.0, "b"), (8.0, 10.0, "c")])

    def test_invalid_content_is_a_bad_response_not_a_server_error(self):
        for raw, duration in self.BROKEN:
            with self.subTest(raw=raw):
                with self.assertRaises(ModelAdapterError) as ctx:
                    clamp_and_validate_segments(raw, chunk_duration=duration)
                self.assertEqual(ctx.exception.kind, "bad_response")

    def test_default_adapter_error_kind_is_unchanged(self):
        self.assertEqual(ModelAdapterError("Cloud inference request failed").kind, "server")

    def test_bad_response_retries_then_falls_back_like_a_server_error(self):
        def decide(kind, attempt):
            plan = dict(models=dict(asr=_row("groq"), translation=_row("mt", "translation")),
                        fallback=dict(asr=[_row("gpu", local=True)], translation=[]),
                        policy=dict(max_attempts=3, retries=1, cooldown_seconds=60), whisper_policy="fixed",
                        whisper_models=[])
            error = ModelAdapterError("Cloud ASR segment end must be strictly greater than start", kind=kind)
            state = sp.TaskState(candidate="user-groq", attempt=attempt)
            return fb.decide_after_failure(plan, "asr", state, "user-groq", error, 1000.0)
        self.assertIn("bad_response", fb.RETRY_KINDS)
        for attempt in (0, 1, 2):
            a, b = decide("bad_response", attempt), decide("server", attempt)
            self.assertEqual((a.action, a.attempt, a.candidate.id), (b.action, b.attempt, b.candidate.id))
            self.assertEqual(a.kind, "bad_response")
        self.assertEqual(decide("bad_response", 0).action, fb.RETRY)
        self.assertEqual(decide("bad_response", 2).action, fb.SWITCH)

    def test_degenerate_chunk_no_longer_fails_the_cloud_asr_run(self):
        reply = {"segments": [{"start": 0.0, "end": 2.0, "text": "前"}, {"start": 3.0, "end": 3.0, "text": "ね"},
                              {"start": 4.0, "end": 5.0, "text": "後"}]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / "source.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000 * 8)
            row = {"remote_model": "whisper", "base_url": "https://example.org/v1"}
            with patch.object(adapters, "_request", return_value=reply), self.assertLogs(cloud_chunks._logger, "WARNING"):
                result = adapters._cloud_asr(row, source, out)
        self.assertEqual("".join(s["text"] for s in result["segments"]), "前ね後")


class ChunkCheckpointAndResumptionTests(unittest.TestCase):
    def test_checkpoint_saves_and_resumes_without_recalling_api(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            source = out / "source.wav"
            # Create a 350s WAV (will split into 2 chunks)
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000 * 350)

            row = {"remote_model": "whisper", "base_url": "https://example.org/v1"}
            call_count = [0]

            def mock_request(row_arg, route, payload, key, content_type):
                call_count[0] += 1
                if call_count[0] == 1:
                    # 1st chunk succeeds
                    return {"segments": [{"start": 0.0, "end": 2.0, "text": "區塊一"}]}
                # 2nd chunk fails with network error
                raise ModelAdapterError("Network failed during chunk 2", kind="network")

            # First run: chunk 1 succeeds, chunk 2 throws error
            with patch.object(adapters, "_request", side_effect=mock_request):
                with self.assertRaises(ModelAdapterError):
                    adapters._cloud_asr(row, source, out)

            self.assertEqual(call_count[0], 2)

            # Check that chunk 0 checkpoint was saved!
            checkpoint_dir = out / "chunks"
            self.assertTrue((checkpoint_dir / "chunk_0000.json").exists())
            self.assertFalse((checkpoint_dir / "chunk_0001.json").exists())

            # Second run: chunk 1 should be loaded from checkpoint, chunk 2 called and succeeds!
            second_run_calls = [0]

            def mock_request_resumed(row_arg, route, payload, key, content_type):
                second_run_calls[0] += 1
                return {"segments": [{"start": 0.0, "end": 2.0, "text": "區塊二"}]}

            with patch.object(adapters, "_request", side_effect=mock_request_resumed):
                result = adapters._cloud_asr(row, source, out)

            # Only chunk 2 was requested via API! Chunk 1 was skipped!
            self.assertEqual(second_run_calls[0], 1)
            self.assertEqual(len(result["segments"]), 2)
            self.assertEqual(result["segments"][0]["text"], "區塊一")
            self.assertEqual(result["segments"][1]["text"], "區塊二")
            self.assertTrue((checkpoint_dir / "chunk_0001.json").exists())


if __name__ == "__main__":
    unittest.main()
