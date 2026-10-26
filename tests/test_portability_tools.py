"""Diagnostic and acceptance failure semantics without Mac/GUI/model requirements."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import portability_common as C
import portability_doctor as D
import portability_acceptance as A


class ReportTests(unittest.TestCase):
    def test_json_framing_and_atomic_report(self):
        self.assertEqual(C.last_json('noise\n{"type":"progress"}\n{"type":"done"}\n')["type"], "done")
        self.assertEqual(C.last_json('{\n "status": "pass"\n}')["status"], "pass")
        with self.assertRaises(ValueError):
            C.last_json("no JSON")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            C.write_json(path, {"text": "繁體中文"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["text"], "繁體中文")
            with self.assertRaises(ValueError):
                C.write_json(path, {"fps": float("nan")})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_missing_command_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "logs" / "missing.log"
            result = C.run_command(["this-command-does-not-exist-portability"], timeout=1, log_path=log)
            self.assertTrue(log.is_file())
            self.assertIn(result["stderr"], log.read_text(encoding="utf-8"))
        self.assertIsNone(result["returncode"])
        self.assertFalse(result["timed_out"])

    def test_child_timeout_is_a_failure(self):
        result = C.run_command([sys.executable, "-c", "import time; time.sleep(10)"], timeout=.1)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["returncode"], 0)

    def test_exited_parent_does_not_leave_pipe_holding_child(self):
        import psutil
        code = "import subprocess,sys,time; time.sleep(.2); p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True)"
        result = C.run_command([sys.executable, "-c", code], timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["elapsed_s"], 5)
        pid = int(result["stdout"].strip())
        try:
            child = psutil.Process(pid)
            child.wait(timeout=2)
        except psutil.NoSuchProcess:
            pass


class DiagnosticTests(unittest.TestCase):
    def test_timed_out_probe_cannot_pass_from_partial_json(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(D, "run_command", return_value={
                "returncode": 0, "timed_out": True, "stdout": '{"status":"pass","selected_device":"cpu"}', "stderr": ""}):
            report = D.build_report(root=directory)
        self.assertEqual(next(c for c in report["checks"] if c["id"] == "torch-device")["status"], "fail")

    def test_missing_environment_still_produces_report(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(D.importlib.metadata, "version", side_effect=D.importlib.metadata.PackageNotFoundError), \
                patch.object(D.shutil, "which", return_value=None), \
                patch.object(D, "run_command", return_value={"returncode": 1, "stdout": "", "stderr": "torch missing", "timed_out": False}), \
                patch.object(D.sys, "platform", "linux"):
            report = D.build_report(root=directory)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any(c["id"] == "torch-device" and c["status"] == "fail" for c in report["checks"]))
        self.assertTrue(any(c["id"] == "package:pyatv" and c["status"] == "skip" for c in report["checks"]))
        self.assertFalse(report["mac_validated"])

    def test_required_hardware_cannot_pass_with_fallback(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(D, "run_command", return_value={"returncode": 0, "stdout": '{"status":"pass","selected_device":"cpu","mps_available":false}', "stderr": "", "timed_out": False}), \
                patch.object(D.encoding_backend, "encoder_usable", return_value=False):
            report = D.build_report(root=directory, require_mps=True, require_videotoolbox=True)
        by_id = {c["id"]: c for c in report["checks"]}
        self.assertEqual(by_id["required:mps"]["status"], "fail")
        self.assertEqual(by_id["encoder:h264_videotoolbox"]["status"], "fail")


class AcceptanceTests(unittest.TestCase):
    def test_timeout_with_successful_parent_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = A.Acceptance(directory)
            with patch.object(A, "run_command", return_value={"returncode": 0, "timed_out": True, "stdout": "", "stderr": ""}):
                with self.assertRaises(RuntimeError):
                    runner.command("timeout", ["worker"])

    def test_failed_check_does_not_hide_other_results(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = A.Acceptance(directory)
            with patch.object(runner, "check_encode", side_effect=RuntimeError("encode broke")), \
                    patch.object(runner, "check_fixture", return_value={"audio": "aac"}):
                report = runner.run(("encode", "fixture"))
        self.assertEqual(report["status"], "fail")
        self.assertEqual([c["status"] for c in report["checks"]], ["fail", "pass"])

    def test_retry_runs_failed_only_and_records_non_retested_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.json"
            C.write_json(old, {"schema_version": 1, "kind": "portability-acceptance", "environment": C.environment(),
                "options": {"device": "cpu", "encoder": "libx264"},
                "checks": [{"id": "fixture", "status": "pass"}, {"id": "encode", "status": "fail"}]})
            with patch.object(A.Acceptance, "check_encode", return_value={"encoder": "libx264"}) as encode, \
                    patch.object(A.Acceptance, "check_fixture") as fixture:
                self.assertEqual(A.main(["--output-dir", directory, "--retry-failed", str(old)]), 0)
            report = json.loads((Path(directory) / "acceptance.json").read_text(encoding="utf-8"))
            encode.assert_called_once()
            fixture.assert_not_called()
            self.assertEqual(report["not_retested"], ["fixture"])
            self.assertEqual(report["options"]["device"], "cpu")

    def test_required_gpu_forces_diagnostic_even_for_selected_fixture(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(A.Acceptance, "check_diagnostic", side_effect=RuntimeError("no MPS")) as diagnostic, \
                patch.object(A.Acceptance, "check_fixture", return_value={}):
            self.assertEqual(A.main(["--output-dir", directory, "--checks", "fixture", "--require-mps"]), 1)
            diagnostic.assert_called_once()

    def test_retry_rejects_other_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.json"
            C.write_json(old, {"schema_version": 1, "kind": "portability-acceptance", "environment": {"os": "wrong"}, "options": {}, "checks": []})
            with self.assertRaises(SystemExit) as failure:
                A.main(["--output-dir", directory, "--retry-failed", str(old)])
            self.assertEqual(failure.exception.code, 2)

    def test_retry_rejects_malformed_report_shapes(self):
        valid = {"schema_version": 1, "kind": "portability-acceptance", "environment": C.environment(), "options": {}, "checks": []}
        malformed = [[], {**valid, "environment": []}, {**valid, "checks": None},
                     {**valid, "checks": [{"id": "bogus", "status": "pass"}]},
                     {**valid, "options": {"device": []}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for report in malformed:
                C.write_json(path, report)
                with self.assertRaises(ValueError):
                    A.load_retry(path)

    def test_interrupt_cleans_owned_command_and_propagates(self):
        from unittest.mock import Mock
        process = Mock()
        process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
        with patch.object(C.subprocess, "Popen", return_value=process), patch.object(C, "_kill_owned") as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                C.run_command(["worker"], timeout=1)
        cleanup.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
