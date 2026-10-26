"""CPU-only tests for the subtitle runtime's fail-closed GPU gate."""

from __future__ import annotations

import unittest
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles import runtime, asr


class GpuSafetyTests(unittest.TestCase):
    def test_idle_ui_exception_is_limited_to_approved_pythonw_pids(self):
        with patch.dict(os.environ, {'OH_STREAMER_SUBTITLE_IDLE_UI_PIDS': '28968,48588'}):
            self.assertTrue(runtime._ignore_idle_ui('pythonw.exe', 28968))
            self.assertFalse(runtime._ignore_idle_ui('pythonw.exe', 11111))
            self.assertFalse(runtime._ignore_idle_ui('mpv.exe', 28968))

    def setUp(self) -> None:
        self.power_patch = patch.object(
            runtime, "_on_ac_power", return_value={"on_ac_power": True, "source": "test"}
        )
        self.process_patch = patch.object(
            runtime, "_process_names", return_value=(["python.exe"], "mock", [])
        )
        self.compute_patch = patch.object(
            runtime, "_compute_processes_nvml", return_value=([], "mock-NVML")
        )
        self.power_patch.start()
        self.process_patch.start()
        self.compute_patch.start()
        self.addCleanup(self.power_patch.stop)
        self.addCleanup(self.process_patch.stop)
        self.addCleanup(self.compute_patch.stop)

    def test_sustained_utilization_without_visible_compute_pid_fails_closed(self) -> None:
        with patch.object(
            runtime, "_gpu_utilization_samples", return_value=([82.0, 81.0, 83.0], "mock", [])
        ):
            with self.assertRaisesRegex(runtime.GPUSafetyError, "unidentified workload"):
                runtime.assert_gpu_safe()

    def test_low_utilization_returns_probe_audit(self) -> None:
        with patch.object(
            runtime, "_gpu_utilization_samples", return_value=([0.0, 4.0, 1.0], "mock", [])
        ):
            audit = runtime.assert_gpu_safe()

        self.assertEqual(audit["utilization_check"]["samples_percent"], [0.0, 4.0, 1.0])
        self.assertEqual(audit["compute_check"]["other_processes"], 0)

    def test_idle_pythonw_mpv_and_desktop_names_do_not_block(self):
        with patch.object(runtime, '_process_names', return_value=(['pythonw.exe', 'mpv.exe', 'ChatGPT.exe'], 'mock', [])), \
                patch.object(runtime, '_gpu_utilization_samples', return_value=([0, 0, 0], 'mock', [])):
            self.assertEqual(runtime.assert_gpu_safe()['process_check']['blocked_names_found'], [])

    def test_idle_context_without_cuda_allocation_is_audited_and_allowed(self):
        jobs = [{'pid': 123, 'used_gpu_memory_bytes': None}, {'pid': 456, 'used_gpu_memory_bytes': 0},
                {'pid': 789, 'used_gpu_memory_bytes': (1 << 64) - 1}]
        with patch.object(runtime, '_compute_processes_nvml', return_value=(jobs, 'mock')), \
                patch.object(runtime, '_gpu_utilization_samples', return_value=([0, 0, 0], 'mock', [])):
            self.assertEqual(runtime.assert_gpu_safe()['compute_check']['idle_contexts'], jobs)

    def test_cuda_allocation_or_actual_context_activity_still_blocks(self):
        for memory, samples in ((1024, [0, 0, 0]), (None, [0, 2, 0])):
            with self.subTest(memory=memory, samples=samples), \
                    patch.object(runtime, '_compute_processes_nvml', return_value=([{'pid': 123, 'used_gpu_memory_bytes': memory}], 'mock')), \
                    patch.object(runtime, '_gpu_utilization_samples', return_value=(samples, 'mock', [])):
                with self.assertRaisesRegex(runtime.GPUSafetyError, 'CUDA workload'):
                    runtime.assert_gpu_safe()

    def test_gpu_probe_failure_still_fails_closed(self):
        with patch.object(runtime, '_compute_processes_nvml', side_effect=RuntimeError('denied')), \
                patch.object(runtime, '_compute_processes_smi', side_effect=RuntimeError('denied')):
            with self.assertRaisesRegex(runtime.RuntimeCheckError, 'could not confirm'):
                runtime.assert_gpu_safe()

    def test_missing_process_names_do_not_override_confirmed_idle_gpu(self):
        with patch.object(runtime, '_process_names', side_effect=runtime.RuntimeCheckError('denied')), \
                patch.object(runtime, '_gpu_utilization_samples', return_value=([0, 0, 0], 'mock', [])):
            self.assertEqual(runtime.assert_gpu_safe()['process_check']['source'], 'unavailable')

    def test_invalid_utilization_fails_closed(self):
        for samples in ([], [float('nan')], [float('inf')], [-1], [101]):
            with self.subTest(samples=samples), \
                    patch.object(runtime, '_gpu_utilization_samples', return_value=(samples, 'mock', [])):
                with self.assertRaises(runtime.RuntimeCheckError):
                    runtime.assert_gpu_safe()

    def test_smi_fallback_preserves_known_allocation_and_unknown_context(self):
        result = SimpleNamespace(returncode=0, stdout='123, python.exe, 2048\n456, ChatGPT.exe, [N/A]\n', stderr='')
        with patch.object(runtime, '_run_checked', return_value=result):
            jobs, source = runtime._compute_processes_smi()
        self.assertEqual(source, 'nvidia-smi')
        self.assertEqual(jobs[0]['used_gpu_memory_bytes'], 2048 * 1024 * 1024)
        self.assertIsNone(jobs[1]['used_gpu_memory_bytes'])

    def test_successful_utilization_fallback_keeps_primary_probe_error(self) -> None:
        with patch.object(
            runtime, "_gpu_util_samples_nvml", side_effect=RuntimeError("NVML sample denied")
        ), patch.object(
            runtime, "_gpu_util_samples_smi", return_value=([2.0, 1.0, 0.0], "mock-smi")
        ):
            samples, source, errors = runtime._gpu_utilization_samples()

        self.assertEqual(samples, [2.0, 1.0, 0.0])
        self.assertEqual(source, "mock-smi")
        self.assertTrue(any("NVML sample denied" in error for error in errors))


class AsrDataTests(unittest.TestCase):
    def test_queue_admission_replaces_idle_probe_but_standalone_stays_strict(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        for managed in (False, True):
            with self.subTest(gpu_managed=managed), tempfile.TemporaryDirectory() as folder:
                with patch.dict(sys.modules, {"torch": fake_torch}), \
                        patch.object(runtime, "local_model_fingerprint", return_value="fixture"), \
                        patch.object(runtime, "source_identity", return_value=("fixture", {})), \
                        patch.object(runtime, "assert_gpu_safe", side_effect=runtime.GPUSafetyError("mpv")) as safety, \
                        patch.object(runtime, "resample_to_wav", side_effect=RuntimeError("audio boundary")):
                    expected = "audio boundary" if managed else "mpv"
                    with self.assertRaisesRegex(RuntimeError, expected):
                        asr.run_asr("fixture.wav", folder, cache_dir=Path(folder) / "cache",
                                    gpu_managed=managed)
                    self.assertEqual(safety.call_count, 0 if managed else 1)

    def test_cached_asr_rebuilds_missing_srt_without_model_or_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {'segments': [{'start': 0, 'end': 2, 'text': 'こんにちは'}],
                    'raw_segments': [], 'speech_intervals': [[0, 2]],
                    'metrics': {'wall_seconds': 12.3, 'realtime_factor_wall': 48.8}}
            with patch.object(runtime, 'local_model_fingerprint', return_value='fixture'), \
                 patch.object(runtime, 'source_identity', return_value=('fixture', {})), \
                 patch.object(asr, '_cache_matches', return_value=data), \
                 patch.object(asr, '_make_pipeline') as model, \
                 patch.object(runtime, 'assert_gpu_safe') as safety:
                result = asr.run_asr('fixture.wav', root, cache_dir=root / 'cache')
            self.assertTrue(result['metrics']['cache_hit'])
            self.assertEqual(result['metrics']['wall_seconds'], 12.3)
            self.assertEqual(result['metrics']['realtime_factor_wall'], 48.8)
            self.assertIn('こんにちは', (root / 'ja.srt').read_text('utf-8'))
            model.assert_not_called()
            safety.assert_not_called()

    def test_chunk_offsets_and_missing_end_keep_original_audio_time(self):
        rows = asr._json_chunks({'chunks': [
            {'text': '先頭', 'timestamp': (0.5, 1.5)},
            {'text': '末尾', 'timestamp': (2, None)}]}, offset=100, piece_duration=3)
        self.assertEqual([(r['start'], r['end']) for r in rows], [(100.5, 101.5), (102, 103)])

    def test_invalid_chunk_timestamps_are_clamped_without_nonfinite_values(self):
        rows = asr._json_chunks({'chunks': [
            {'text': '声', 'timestamp': (-2, float('nan'))}]}, offset=50, piece_duration=4)
        self.assertEqual((rows[0]['start'], rows[0]['end']), (50, 54))

    def test_missing_chunk_timestamps_have_explicit_fallback_marker(self):
        rows = asr._json_chunks({'text': '全文'}, offset=7, piece_duration=2)
        self.assertEqual(rows[0]['timestamp_source'], 'whole_input_fallback')
        self.assertEqual((rows[0]['start'], rows[0]['end']), (7, 9))

    def test_no_vad_keeps_complete_audio_while_vad_excludes_long_silence(self):
        intervals = [{'start': 1, 'end': 2}, {'start': 10, 'end': 11}]
        self.assertEqual(asr._inference_groups(intervals, audio_duration=20, use_vad=False), [(0, 20)])
        self.assertEqual(asr._inference_groups(intervals, audio_duration=20, use_vad=True), [(1, 2), (10, 11)])

    def test_model_fingerprint_ignores_downloader_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'fixture')
            before = runtime.local_model_fingerprint(root)
            (root / '.cache').mkdir()
            (root / '.cache' / 'download.lock').write_text('metadata')
            self.assertEqual(runtime.local_model_fingerprint(root), before)
            (root / 'model.safetensors').write_bytes(b'changed fixture')
            self.assertNotEqual(runtime.local_model_fingerprint(root), before)

    def test_incomplete_local_model_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'incomplete'):
                runtime.local_model_fingerprint(folder)


if __name__ == "__main__":
    unittest.main()
