"""CPU-only checks for phase 2 adapter safety, bounded server and cache isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.subtitles import cli, environment
from oh_streamer.subtitles.llm import LlamaServerClient, LlamaServerError
from oh_streamer.subtitles.core import Segment, translate_segments
_TEMP = Path(__file__).resolve().parents[1] / '.test-tmp'
_TEMP.mkdir(exist_ok=True)


class IgpuRuntimeTests(unittest.TestCase):
    def tearDown(self):
        environment._vulkan_devices.cache_clear()

    def test_mobile_discrete_radeon_is_not_mistaken_for_integrated(self):
        listing = (' Vulkan0: AMD Radeon RX 7600M XT (8192 MiB)\n'
                   ' Vulkan1: AMD Radeon 6850M XT (8192 MiB)\n'
                   ' Vulkan2: AMD Radeon(TM) 860M Graphics (16444 MiB)\n')
        with patch.object(environment.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=listing, stderr='')):
            self.assertEqual(environment._vulkan_devices('fixture', 1),
                             (('Vulkan2', 'AMD Radeon(TM) 860M Graphics'),))

    def test_enumeration_selects_integrated_adapter_even_when_not_zero(self):
        listing = ('Available devices:\n'
                   ' Vulkan0: NVIDIA GeForce RTX 5060 Laptop GPU (7899 MiB)\n'
                   ' Vulkan1: AMD Radeon(TM) 860M Graphics (16444 MiB)\n')
        with tempfile.TemporaryDirectory(dir=_TEMP) as directory:
            server = Path(directory) / "llama-server.exe"
            server.write_bytes(b"fixture")
            with patch.object(environment.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout="", stderr=listing)) as run:
                result = environment.igpu_runtime({"subtitle_igpu_server": str(server)})
            self.assertTrue(result["igpu_available"])
            self.assertEqual(result["igpu_device"], "Vulkan1")
            self.assertIn("860M", result["igpu_name"])
            self.assertEqual(run.call_args.args[0], [str(server.resolve()), "--list-devices"])

    def test_explicit_discrete_device_or_failed_probe_disables_igpu(self):
        with tempfile.TemporaryDirectory(dir=_TEMP) as directory:
            server = Path(directory) / "server.exe"
            server.write_bytes(b"fixture")
            settings = dict(subtitle_igpu_server=str(server), subtitle_igpu_device="Vulkan0")
            with patch.object(environment, "_vulkan_devices", return_value=(("Vulkan1", "AMD Radeon 860M"),)):
                self.assertFalse(environment.igpu_runtime(settings)["igpu_available"])
            with patch.object(environment.subprocess, "run", side_effect=subprocess.TimeoutExpired("server", 15)):
                self.assertEqual(environment._vulkan_devices(str(server), 1), ())

    def test_explicit_missing_binary_does_not_silently_discover_another(self):
        with patch.object(environment, "_vulkan_devices") as probe:
            result = environment.igpu_runtime({"subtitle_igpu_server": "missing/server.exe"})
        self.assertFalse(result["igpu_available"])
        probe.assert_not_called()

    def test_transport_failure_propagates_without_per_cue_fallback(self):
        batch = Mock(side_effect=LlamaServerError("server transport failed"))
        single = Mock(return_value="translation")
        with self.assertRaisesRegex(LlamaServerError, "transport failed"):
            translate_segments([Segment(0, 1, "こんにちは")], batch, single,
                               propagate_errors=(LlamaServerError,))
        batch.assert_called_once()
        single.assert_not_called()

    def test_vulkan_requires_explicit_device(self):
        for device in (None, "CUDA0", "--help", "Vulkan0 Vulkan1"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                LlamaServerClient("sakura", "model.gguf", "server", backend="vulkan", device=device)

    def test_vulkan_server_command_binds_device_and_limits_host_threads(self):
        with tempfile.TemporaryDirectory(dir=_TEMP) as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"fixture")
            client = LlamaServerClient("sakura", model, "server", backend="vulkan", device="Vulkan1")
            process = SimpleNamespace(pid=123, poll=lambda: None)
            response = SimpleNamespace(status=200)
            class Response:
                def __enter__(self):
                    return response
                def __exit__(self, *_):
                    pass
            with patch("oh_streamer.subtitles.llm.shutil.which", return_value="server"), \
                    patch("oh_streamer.subtitles.llm.subprocess.Popen", return_value=process) as popen, \
                    patch("oh_streamer.subtitles.llm.urlopen", return_value=Response()):
                client.start()
                command = popen.call_args.args[0]
            for key, value in (("--device", "Vulkan1"), ("--fit", "off"),
                               ("--threads", "4"), ("--threads-batch", "4")):
                self.assertEqual(command[command.index(key) + 1], value)
            client._process = None  # Fake process was never launched.

    def test_backend_device_and_server_stamp_separate_translation_cache(self):
        with tempfile.TemporaryDirectory(dir=_TEMP) as directory:
            root = Path(directory)
            source, model, server = root / "ja.json", root / "model.gguf", root / "server.exe"
            source.write_text(json.dumps({"segments": []}), encoding="utf-8")
            model.write_bytes(b"fixture")
            server.write_bytes(b"server")
            common = dict(model="sakura", model_path=model, llama_server=server, gpu_managed=True)
            cuda = cli.translate_file(source, root, **common)
            one = cli.translate_file(source, root, **common, backend="vulkan", device="Vulkan0")
            two = cli.translate_file(source, root, **common, backend="vulkan", device="Vulkan1")
            server.write_bytes(b"different server build")
            three = cli.translate_file(source, root, **common, backend="vulkan", device="Vulkan1")
            self.assertEqual(len({row["cache_key"] for row in (cuda, one, two, three)}), 4)
            self.assertNotIn("execution_backend", cuda["recipe_options"])
            self.assertEqual(one["recipe_options"]["execution_backend"]["device"], "Vulkan0")


if __name__ == "__main__":
    unittest.main()


