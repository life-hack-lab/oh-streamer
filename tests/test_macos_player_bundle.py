"""Mach-O packaging checks use a mutable tool inventory, so run on any host."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("macos_player_bundle", ROOT / "packaging/macos_player_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def load_commands(dependencies=(), rpaths=(), dylib_id=None):
    lines = []
    for command, key, value in [("LC_ID_DYLIB", "name", dylib_id),
                                *[("LC_LOAD_DYLIB", "name", value) for value in dependencies],
                                *[("LC_RPATH", "path", value) for value in rpaths]]:
        if value is not None:
            lines += ["Load command 0", f"          cmd {command}", f"         {key} {value} (offset 24)"]
    return "\n".join(lines)


class ToolInventory:
    """Model copied/thinned binaries and install_name_tool's actual mutations."""
    def __init__(self):
        self.original = {}
        self.copied = {}
        self.calls = []

    def binary(self, path, dependencies=(), rpaths=(), dylib_id=None, architectures=("arm64",)):
        path.parent.mkdir(parents=True, exist_ok=True)
        key = str(path.resolve())
        path.write_text(key, encoding="utf-8")
        self.original[key] = {"dependencies": list(dependencies), "rpaths": list(rpaths),
                              "dylib_id": dylib_id, "architectures": architectures}
        return path.resolve()

    def info(self, path):
        path = Path(path)
        key = str(path.resolve())
        if key in self.original:
            return self.original[key]
        if key not in self.copied:
            original = self.original[path.read_text(encoding="utf-8")]
            self.copied[key] = {name: list(value) if isinstance(value, list) else value for name, value in original.items()}
        return self.copied[key]

    def __call__(self, *args, env=None):
        self.calls.append((args, env))
        tool = Path(args[0]).name
        if tool == "lipo":
            if args[1] == "-archs":
                return " ".join(self.info(args[2])["architectures"])
            Path(args[-1]).write_bytes(Path(args[1]).read_bytes())
            self.info(args[-1])["architectures"] = [args[3]]
        elif tool == "otool":
            info = self.info(args[-1])
            return load_commands(info["dependencies"], info["rpaths"], info["dylib_id"])
        elif tool == "install_name_tool":
            info = self.info(args[-1])
            if args[1] == "-id":
                info["dylib_id"] = args[2]
            elif args[1] == "-change":
                info["dependencies"] = [args[3] if value == args[2] else value for value in info["dependencies"]]
            elif args[1] == "-delete_rpath":
                info["rpaths"].remove(args[2])
        elif "--list-options" in args:
            return " --macos-app-activation-policy regular accessory prohibited"
        return ""


class PlayerBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tools = ToolInventory()
        self.mpv = self.tools.binary(self.root / "prepared/bin/mpv", rpaths=("@loader_path/../lib",))
        self.icon = self.root / "app.icns"
        self.icon.write_bytes(b"icns-shared-brand")
        self.notices = self.root / "notices"
        self.notices.mkdir()
        (self.notices / "COPYING").write_text("notices and source offer", encoding="utf-8")
        self.app = self.root / "Branded.app"
        self.app.mkdir()

    def build(self, **kwargs):
        with patch.object(bundle, "command", side_effect=self.tools):
            return bundle.bundle_player(self.app, self.mpv, self.icon, "Test Brand", self.notices,
                                        architecture="arm64", **kwargs)

    def test_parse_commands_preserves_spaces_and_skips_id_dependency(self):
        parsed = bundle.parse_load_commands(load_commands(("/path with spaces/lib.dylib",), ("@loader_path",), "@rpath/id.dylib"))
        self.assertEqual(parsed.dependencies, ("/path with spaces/lib.dylib",))
        self.assertEqual(parsed.rpaths, ("@loader_path",))
        self.assertEqual(parsed.dylib_id, "@rpath/id.dylib")

    def test_recursive_rpaths_rewrite_portably_keep_icon_and_sign_inside_out(self):
        lib_a = self.tools.binary(self.root / "prepared/lib/libA.dylib", ("@rpath/libB.dylib",), dylib_id="@rpath/libA.dylib")
        lib_b = self.tools.binary(self.root / "prepared/lib/libB.dylib", ("/usr/lib/libSystem.B.dylib",), dylib_id="@rpath/libB.dylib")
        self.tools.info(self.mpv)["dependencies"] = ["@rpath/libA.dylib"]
        originals = {path: path.read_bytes() for path in (self.mpv, lib_a, lib_b)}
        with patch.dict(os.environ, {"DYLD_LIBRARY_PATH": "/homebrew/lib", "VK_DRIVER_FILES": "/host/icd.json"}):
            helper = self.build()
        self.assertEqual(helper, self.app / "Contents/Helpers/oh-player.app")
        contents = helper / "Contents"
        info = plistlib.loads((contents / "Info.plist").read_bytes())
        self.assertEqual(info["CFBundleDisplayName"], "Test Brand Player")
        self.assertEqual(info["CFBundleIdentifier"], "local.oh.streamer.player")
        self.assertNotIn("LSUIElement", info)
        self.assertEqual((contents / "Resources/app.icns").read_bytes(), self.icon.read_bytes())
        self.assertTrue((contents / "Resources/ThirdPartyNotices/COPYING").is_file())
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)
        copied_mpv = self.tools.info(contents / "MacOS/mpv")
        self.assertEqual(copied_mpv["rpaths"], [])
        self.assertTrue(copied_mpv["dependencies"][0].startswith("@loader_path/../Frameworks/"))
        signs = [args[-1] for args, _ in self.tools.calls if args[0].endswith("codesign") and "--sign" in args]
        self.assertEqual(signs[-3:], [str(contents / "MacOS/mpv"), str(helper), str(self.app)])
        self.assertEqual(self.tools.calls[-1][0][1:4], ("--verify", "--deep", "--strict"))
        smoke = next((args, env) for args, env in self.tools.calls if "--version" in args)
        self.assertEqual(smoke[1]["PATH"], "/usr/bin:/bin")
        self.assertEqual(smoke[1]["MPVBUNDLE"], "true")
        self.assertNotIn("DYLD_LIBRARY_PATH", smoke[1])
        self.assertNotIn("VK_DRIVER_FILES", smoke[1])

    def test_missing_architecture_and_dependency_fail_before_creating_payload(self):
        self.tools.info(self.mpv)["architectures"] = ["x86_64"]
        with self.assertRaisesRegex(RuntimeError, "lacks required architecture"):
            self.build()
        self.tools.info(self.mpv)["architectures"] = ["arm64"]
        self.tools.info(self.mpv)["dependencies"] = ["@rpath/missing.dylib"]
        with self.assertRaisesRegex(RuntimeError, "Unresolved mpv dependency"):
            self.build()
        self.assertFalse((self.app / bundle.PLAYER_APP_RELATIVE).exists())

    def test_system_swift_rpath_becomes_absolute_before_rpaths_removed(self):
        self.tools.info(self.mpv).update(dependencies=["@rpath/libswiftCore.dylib"], rpaths=["/usr/lib/swift"])
        helper = self.build()
        dependencies = self.tools.info(helper / "Contents/MacOS/mpv")["dependencies"]
        self.assertEqual(dependencies, [str(Path("/usr/lib/swift/libswiftCore.dylib"))])

    def test_vulkan_requires_manifest_then_collects_driver_and_rewrites_json(self):
        vulkan = self.tools.binary(self.root / "prepared/lib/libvulkan.1.dylib", dylib_id="@rpath/libvulkan.1.dylib")
        self.tools.info(self.mpv)["dependencies"] = [str(vulkan)]
        with self.assertRaisesRegex(RuntimeError, "--mpv-vulkan-icd"):
            self.build()
        driver = self.tools.binary(self.root / "prepared/lib/libMoltenVK.dylib", dylib_id="@rpath/libMoltenVK.dylib")
        manifest = self.root / "MoltenVK_icd.json"
        manifest.write_text(json.dumps({"ICD": {"library_path": str(driver), "api_version": "1.3.0"}}), encoding="utf-8")
        helper = self.build(vulkan_icd=manifest)
        copied_manifest = helper / "Contents/Resources/vulkan/icd.d/MoltenVK_icd.json"
        data = json.loads(copied_manifest.read_text(encoding="utf-8"))
        self.assertTrue((copied_manifest.parent / data["ICD"]["library_path"]).resolve().is_file())
        smoke = next(env for args, env in self.tools.calls if "--version" in args)
        self.assertEqual(smoke["VK_DRIVER_FILES"], str(copied_manifest))
        self.assertEqual(smoke["VK_ICD_FILENAMES"], str(copied_manifest))

    def test_verify_rejects_absolute_build_machine_reference(self):
        framework = self.app / "Contents/Frameworks"
        framework.mkdir(parents=True)
        mpv = self.tools.binary(self.app / "Contents/MacOS/mpv", [str(self.mpv)])
        with patch.object(bundle, "command", side_effect=self.tools):
            with self.assertRaisesRegex(RuntimeError, "escapes its payload"):
                bundle.verify_payload(mpv, "arm64")

    def test_universal_source_is_thinned_without_mutating_original(self):
        self.tools.info(self.mpv)["architectures"] = ["arm64", "x86_64"]
        helper = self.build()
        self.assertEqual(self.tools.info(helper / "Contents/MacOS/mpv")["architectures"], ["arm64"])
        self.assertEqual(self.tools.info(self.mpv)["architectures"], ["arm64", "x86_64"])

    def test_cyclic_dylib_graph_and_loader_rpath_finish_with_one_copy_each(self):
        first = self.tools.binary(self.root / "prepared/lib/libA.dylib", ["@loader_path/libB.dylib"],
                                  rpaths=["@loader_path"], dylib_id="@rpath/libA.dylib")
        self.tools.binary(self.root / "prepared/lib/libB.dylib", ["@rpath/libA.dylib"], dylib_id="@rpath/libB.dylib")
        self.tools.info(self.mpv)["dependencies"] = [str(first)]
        helper = self.build()
        self.assertEqual(len(list((helper / "Contents/Frameworks").iterdir())), 2)

    def test_missing_vulkan_driver_and_framework_resources_are_rejected(self):
        manifest = self.root / "MoltenVK_icd.json"
        manifest.write_text(json.dumps({"ICD": {"library_path": "missing.dylib"}}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Missing Vulkan ICD library"):
            self.build(vulkan_icd=manifest)
        framework = self.tools.binary(self.root / "Foo.framework/Versions/A/Foo")
        self.tools.info(self.mpv)["dependencies"] = [str(framework)]
        with self.assertRaisesRegex(RuntimeError, "External framework resources"):
            self.build()

    def test_tool_failures_propagate_and_commands_have_timeout(self):
        with patch.object(bundle.subprocess, "run", return_value=type("Result", (), {"stdout": "mpv version"})()) as run:
            self.assertEqual(bundle.command("mpv", "--version", env={"PATH": "/usr/bin"}), "mpv version")
        self.assertEqual(run.call_args.kwargs["timeout"], 30)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.kwargs["env"], {"PATH": "/usr/bin"})
        with patch.object(bundle.subprocess, "run", side_effect=subprocess.TimeoutExpired("mpv", 30)):
            with self.assertRaises(subprocess.TimeoutExpired):
                bundle.command("mpv", "--version")
        error = subprocess.CalledProcessError(1, ["install_name_tool"], stderr="updated load commands do not fit")
        with patch.object(bundle.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "updated load commands do not fit"):
                bundle.command("install_name_tool")


if __name__ == "__main__":
    unittest.main()
