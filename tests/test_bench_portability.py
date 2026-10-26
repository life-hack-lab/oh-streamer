"""Portable benchmark policy, metric and failure tests without torch imports."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

_sys.path.insert(0, str(_STREAMER_ROOT / "benchmarks"))
import bench_portability as bench


class Device:
    def __init__(self, name):
        self.type = name.split(":")[0]
        self.name = name

    def __str__(self):
        return self.name


class BenchmarkTests(unittest.TestCase):
    def args(self, *extra):
        return bench.parser().parse_args(["local.mkv", "--output", "report.json", *extra])

    def test_finite_bounds_and_lists(self):
        for value in ("nan", "inf", "-1", "0"):
            with self.assertRaises(Exception):
                bench.finite_number(value)
        self.assertEqual(bench.finite_number("0", positive=False), 0)
        for value in ("nan", "-0.1"):
            with self.assertRaises(Exception):
                bench.finite_number(value, positive=False)
        for value in ("0", "2,", "1.5", "-2"):
            with self.assertRaises(Exception):
                bench.clip_lengths(value)
        self.assertEqual(bench.clip_lengths("2, 4,2"), [2, 4])
        with self.assertRaises(Exception):
            bench.model_names("v4-fast,")

    def test_device_and_precision_selection(self):
        torch = NS(cuda=NS(is_available=lambda: False, device_count=lambda: 0),
                   backends=NS(mps=NS(is_available=lambda: True)), device=Device)
        device = bench.device_backend.select_device("auto", torch_module=torch)
        self.assertEqual(device.type, "mps")
        self.assertTrue(bench.select_fp16("auto", device, lambda d: d.type == "mps"))
        self.assertFalse(bench.select_fp16("off", device, lambda _: True))
        self.assertTrue(bench.select_fp16("on", Device("cpu"), lambda _: False))
        with self.assertRaises(RuntimeError):
            bench.device_backend.select_device("cuda:1", torch_module=torch)

    def test_cpu_and_mps_never_use_cuda_metric_or_sync(self):
        cuda = Mock()
        torch = NS(cuda=cuda, mps=NS(current_allocated_memory=Mock(side_effect=[100, 300, 200]),
                                   synchronize=Mock()))
        memory = bench.MemoryMetrics(torch, Device("mps"))
        for _ in range(3):
            memory.sample()
        report = memory.finish()
        self.assertEqual(report["sampled_max_current_allocated_bytes"], 300)
        self.assertNotIn("peak_allocated_bytes", report)
        self.assertIn("not a true", report["scope"])
        bench.synchronize(torch, Device("mps"))
        torch.mps.synchronize.assert_called_once()
        cpu = bench.MemoryMetrics(torch, Device("cpu"))
        cpu.start()
        self.assertEqual(cpu.finish()["status"], "unavailable")
        bench.synchronize(torch, Device("cpu"))
        self.assertEqual(cuda.mock_calls, [])

    def test_cuda_peak_is_for_selected_device(self):
        torch = NS(cuda=Mock())
        torch.cuda.max_memory_allocated.return_value = 4096
        device = Device("cuda:1")
        memory = bench.MemoryMetrics(torch, device)
        memory.start()
        self.assertEqual(memory.finish()["peak_allocated_bytes"], 4096)
        torch.cuda.reset_peak_memory_stats.assert_called_once_with(device)
        torch.cuda.max_memory_allocated.assert_called_once_with(device)

    def test_timing_excludes_imports_load_and_first_frame_from_steady(self):
        report = bench.timing_metrics(5, 100, 102, 104, 2)
        self.assertEqual(report["first_frame_s"], 2)
        self.assertEqual(report["processing_fps"], 1.25)
        self.assertEqual(report["steady_output_fps"], 2)
        self.assertIsNone(bench.timing_metrics(1, 100, 102, 104, 2)["steady_output_fps"])
        with self.assertRaisesRegex(RuntimeError, "no frames"):
            bench.timing_metrics(0, 100, None, 104, 2)

    def test_detection_counts_exclude_prefetched_frames(self):
        entries = iter([(10, 0), (11, 2), (12, 5), object()])
        restorer = NS(frame_detection_queue=NS(get=lambda: next(entries)))
        counts = bench.DetectionCounts(restorer)
        for _ in range(4):
            restorer.frame_detection_queue.get()
        counts.delivered(10)
        counts.delivered(11)
        report = counts.report()
        self.assertEqual(report["mosaics"], 2)
        self.assertEqual(report["frames_with_mosaic"], 1)
        self.assertEqual(report["observed_frames"], 2)
        with self.assertRaisesRegex(RuntimeError, "missing detection"):
            counts.delivered(13)

    def fake_ls(self, model_path, *, produce=True):
        class Restorer:
            start_frame = 0
            def __init__(self):
                self.frame_detection_queue = NS(get=lambda: (self.current, self.current % 2))
                self.mosaic_restoration_model = NS()
                self.stopped = False
            def start(self, start_ns=0):
                self.start_ns = start_ns
            def __iter__(self):
                for i in range(3 if produce else 0):
                    self.current = i
                    self.frame_detection_queue.get()
                    yield b"frame", i
            def stop(self):
                self.stopped = True
        restorer = Restorer()
        model = NS(path=str(model_path))
        class Factory:
            def __init__(self, device, clip):
                self.device, self.clip = device, clip
                self.det_model = self.res_model = NS(dtype="float32")
            def load(self, det, res, fp16):
                self.det_name, self.res_name, self.fp16 = det, res, fp16
            def make(self, source):
                return restorer
        ls = NS(torch=NS(cuda=Mock()), gpu_has_fp16_acceleration=lambda _: False,
                ModelFiles=NS(get_detection_model_by_name=lambda _: model,
                              get_restoration_model_by_name=lambda _: model),
                Source=lambda *a: NS(meta=NS(video_fps_exact=Fraction(2), video_width=256,
                                           video_height=144, duration=10)),
                RestorerFactory=Factory, STOP_MARKER=object(), ErrorMarker=type("ErrorMarker", (), {}))
        return ls, restorer

    def test_real_pipeline_interface_and_actual_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pt"
            model_path.touch()
            ls, restorer = self.fake_ls(model_path)
            result = {}
            with patch.object(bench.device_backend, "select_device", return_value=Device("cpu")), \
                    patch.object(bench.time, "perf_counter", side_effect=[100, 102, 103, 104, 110]):
                bench.run_configuration(self.args("--device", "cpu", "--duration", "1.5"), result, ls=ls)
            self.assertEqual(result["selected"]["device"], "cpu")
            self.assertEqual(result["selected"]["detection_model"], "v4-fast")
            self.assertFalse(result["selected"]["fp16"])
            self.assertEqual(result["model_load_s"], 2)
            self.assertEqual(result["timings"]["first_frame_s"], 1)
            self.assertEqual(result["timings"]["processing_s"], 7)
            self.assertEqual(result["frames"], 3)
            self.assertTrue(restorer.stopped)
            self.assertEqual(ls.torch.cuda.mock_calls, [])

    def test_missing_model_and_no_frames_are_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            ls, _ = self.fake_ls(path)
            with patch.object(bench.device_backend, "select_device", return_value=Device("cpu")):
                with self.assertRaisesRegex(RuntimeError, "missing or unknown"):
                    bench.run_configuration(self.args(), {}, ls=ls)
                path.touch()
                ls, restorer = self.fake_ls(path, produce=False)
                with self.assertRaisesRegex(RuntimeError, "no frames"):
                    bench.run_configuration(self.args(), {}, ls=ls)
                self.assertTrue(restorer.stopped)

    def test_child_timeout_kills_tree_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock()
            process.wait.side_effect = subprocess.TimeoutExpired("bench", 0.1)
            with patch.object(bench.subprocess, "Popen", return_value=process), \
                    patch.object(bench, "terminate_process_tree") as terminate:
                result = bench.run_child(self.args("--timeout", "0.1"), "v4-fast", 4, 0, Path(tmp))
            terminate.assert_called_once_with(process)
            self.assertEqual(result["status"], "fail")
            self.assertIn("timed out", result["error"])
            self.assertNotIn("timings", result)

    def test_child_exit_failure_cannot_claim_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            def launch(*args, **kwargs):
                kwargs["stdout"].write(bench.RESULT_PREFIX + json.dumps({"status": "pass"}) + "\n")
                kwargs["stdout"].flush()
                return NS(wait=lambda timeout: 1)
            with patch.object(bench.subprocess, "Popen", side_effect=launch):
                result = bench.run_child(self.args(), "v4-fast", 4, 0, Path(tmp))
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["exit_code"], 1)

    def test_crossproduct_isolated_and_report_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            with patch.object(bench, "run_child", side_effect=[{"status": "pass"}, {"status": "fail"},
                                                              {"status": "pass"}, {"status": "pass"}]) as run:
                code = bench.main(["local.mkv", "--output", str(output), "--models", "v4-fast,v4-accurate",
                                   "--clip-lengths", "4,8"])
            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 4)
            self.assertEqual(json.loads(output.read_text())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
