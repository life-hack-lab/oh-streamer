"""Player selection and child-only macOS bundle identity, without a real mpv."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oh_streamer.player import player_runtime as runtime
from oh_streamer.player import oh_player as player


class PlayerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.main = self.root / "Application With Spaces.app" / "Contents" / "MacOS" / "oh-streamer"
        self.player = self.main.parent.parent / "Helpers/oh-player.app/Contents/MacOS/mpv"

    def frozen_mac(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(sys, "platform", "darwin"))
        stack.enter_context(patch.object(sys, "frozen", True, create=True))
        stack.enter_context(patch.object(sys, "executable", str(self.main)))
        stack.enter_context(patch.object(sys, "_MEIPASS", str(self.root / "unrelated-resources"), create=True))
        return stack

    def test_bundled_player_wins_over_path_and_does_not_use_meipass(self):
        self.player.parent.mkdir(parents=True)
        self.player.touch()
        with self.frozen_mac(), patch.object(runtime.shutil, "which") as which:
            self.assertEqual(runtime.find_mpv(None, self.root), str(self.player))
        which.assert_not_called()

    def test_missing_packaged_player_fails_instead_of_using_system_mpv(self):
        with self.frozen_mac(), patch.object(runtime.shutil, "which") as which:
            with self.assertRaisesRegex(SystemExit, "Bundled mpv is missing"):
                runtime.find_mpv(None, self.root)
        which.assert_not_called()

    def test_explicit_override_wins_and_missing_override_is_reported(self):
        external = self.root / "custom-mpv"
        external.touch()
        with self.frozen_mac():
            self.assertEqual(runtime.find_mpv(str(external), self.root), str(external))
            with self.assertRaisesRegex(SystemExit, "Requested mpv executable not found"):
                runtime.find_mpv(str(self.root / "missing"), self.root)
            self.assertEqual(runtime.process_kwargs(str(external)), {})

    def test_bundle_environment_is_child_only_and_preserves_existing_variables(self):
        with self.frozen_mac(), patch.dict(os.environ, {"TEST_PLAYER_ENV": "preserved"}, clear=True):
            kwargs = runtime.process_kwargs(str(self.player))
            self.assertEqual(kwargs["env"]["MPVBUNDLE"], "true")
            self.assertEqual(kwargs["env"]["TEST_PLAYER_ENV"], "preserved")
            self.assertNotIn("MPVBUNDLE", os.environ)

    def test_source_mac_uses_path_without_claiming_bundle_identity(self):
        external = self.root / "mpv"
        external.touch()
        with patch.object(sys, "platform", "darwin"), patch.object(sys, "frozen", False, create=True), \
                patch.object(runtime.shutil, "which", return_value=str(external)):
            self.assertEqual(runtime.find_mpv(None, self.root), str(external))
            self.assertEqual(runtime.process_kwargs(str(external)), {})

    def test_packaged_vulkan_driver_overrides_external_driver_paths_only_in_child(self):
        manifest = self.player.parents[1] / "Resources/vulkan/icd.d/MoltenVK_icd.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}", encoding="utf-8")
        with self.frozen_mac(), patch.dict(os.environ, {
                "VK_DRIVER_FILES": "external-driver.json", "VK_ICD_FILENAMES": "old-driver.json",
                "VK_ADD_DRIVER_FILES": "extra-driver.json"}):
            environment = runtime.process_kwargs(str(self.player))["env"]
            self.assertEqual(environment["VK_DRIVER_FILES"], str(manifest))
            self.assertEqual(environment["VK_ICD_FILENAMES"], str(manifest))
            self.assertNotIn("VK_ADD_DRIVER_FILES", environment)
            self.assertEqual(os.environ["VK_DRIVER_FILES"], "external-driver.json")

    def test_windows_source_keeps_project_tool_fallback(self):
        external = self.root / "tools/mpv/mpv.exe"
        external.parent.mkdir(parents=True)
        external.touch()
        with patch.object(sys, "platform", "win32"), patch.object(runtime.shutil, "which", return_value=None), \
                patch.object(Path, "is_file", autospec=True, side_effect=lambda path: path == external):
            self.assertEqual(runtime.find_mpv(None, self.root), str(external))
            self.assertEqual(runtime.process_kwargs(str(external)), {})


class PlayerSpawnTests(unittest.TestCase):
    def test_spawn_applies_branding_to_exact_child_and_forwards_bundle_environment(self):
        environment = {"MPVBUNDLE": "true"}
        with patch.object(player.subprocess, "Popen") as popen, \
                patch.object(player.player_runtime, "process_kwargs", return_value={"env": environment}), \
                patch.object(player.player_branding, "apply_player_branding") as branding, \
                patch.object(player, "primary_display_height", return_value=None):
            child = player.start_mpv("/helper/with spaces/mpv", "ipc", "title", 1.0)
        self.assertIs(child, popen.return_value)
        self.assertEqual(popen.call_args.args[0][0], "/helper/with spaces/mpv")
        self.assertIs(popen.call_args.kwargs["env"], environment)
        branding.assert_called_once_with(child)


if __name__ == "__main__":
    unittest.main()
