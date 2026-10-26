"""Focused checks for lazy migration of files from the old checkout layout."""
from pathlib import Path
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase, main
from unittest.mock import patch, MagicMock

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
for _path in (_STREAMER_ROOT, _STREAMER_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
_TEST_TMP = _STREAMER_ROOT / ".test-tmp" / "user-data-migration"
_TEST_TMP.mkdir(parents=True, exist_ok=True)


def temp_directory():
    return tempfile.TemporaryDirectory(dir=_TEST_TMP)

from oh_streamer import user_data  # noqa: E402
from oh_streamer.platform import airplay_cast as AC  # noqa: E402
from oh_streamer.sources import source_resolve as SR  # noqa: E402


class AtomicMigrationTest(TestCase):
    def test_concurrent_migration_publishes_one_complete_copy(self):
        with temp_directory() as directory:
            root = Path(directory)
            source = root / "legacy.json"
            destination = root / "data" / "settings.json"
            payload = b'{"max_clip_length": 90}\n'
            source.write_bytes(payload)

            def migrate(_):
                return user_data.migrate_legacy_file(source, destination, lambda value: value == payload)

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(migrate, range(24)))

            self.assertEqual(sum(copied for copied, _warning in results), 1)
            self.assertTrue(all(warning is None for _copied, warning in results))
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(list(destination.parent.glob("*.migrating")), [])

    def test_existing_destination_wins_without_overwrite(self):
        with temp_directory() as directory:
            root = Path(directory)
            source, destination = root / "legacy.json", root / "data.json"
            source.write_text("old", encoding="utf-8")
            destination.write_text("new", encoding="utf-8")

            copied, warning = user_data.migrate_legacy_file(source, destination)

            self.assertFalse(copied)
            self.assertIsNone(warning)
            self.assertEqual(destination.read_text(encoding="utf-8"), "new")
            self.assertEqual(source.read_text(encoding="utf-8"), "old")


class CredentialMigrationTest(TestCase):
    def test_mixed_legacy_devices_preserve_valid_pairing(self):
        with temp_directory() as directory:
            root = Path(directory)
            destination, legacy = root / "new.json", root / "old.json"
            valid = {"name": "TV", "airplay": "secret"}
            legacy.write_text(json.dumps({"devices": {"tv": valid, "stale": {"airplay": None}}}),
                              encoding="utf-8")
            with patch.multiple(AC, CRED_FILE=destination, LEGACY_CRED_FILE=legacy):
                self.assertEqual(AC.CredentialStore().load(), {"tv": valid})
            self.assertTrue(legacy.exists())

    def test_credentials_migrate_on_first_load(self):
        with temp_directory() as directory:
            root = Path(directory)
            destination = root / "data" / "airplay_credentials.json"
            legacy = root / "checkout" / "airplay_credentials.json"
            legacy.parent.mkdir()
            payload = {"version": 1, "devices": {"tv": {"name": "TV", "airplay": "secret"}}}
            legacy.write_text(json.dumps(payload), encoding="utf-8")
            with patch.multiple(AC, CRED_FILE=destination, LEGACY_CRED_FILE=legacy):
                store = AC.CredentialStore(destination)
                self.assertFalse(destination.exists())
                self.assertEqual(store.load(), payload["devices"])

            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), payload)
            self.assertTrue(legacy.exists())

    def test_invalid_credentials_are_not_copied(self):
        with temp_directory() as directory:
            root = Path(directory)
            destination = root / "data" / "airplay_credentials.json"
            legacy = root / "checkout" / "airplay_credentials.json"
            legacy.parent.mkdir()
            legacy.write_text('{"devices": {"tv": {"airplay":', encoding="utf-8")
            with patch.multiple(AC, CRED_FILE=destination, LEGACY_CRED_FILE=legacy):
                with self.assertLogs("oh_streamer.platform.airplay_cast", level="WARNING"):
                    self.assertEqual(AC.CredentialStore(destination).load(), {})

            self.assertFalse(destination.exists())
            self.assertTrue(legacy.exists())


class ResolverMigrationTest(TestCase):
    def test_default_override_migrates_lazily(self):
        with temp_directory() as directory:
            root = Path(directory)
            destination = root / "data" / "resolvers.json"
            legacy = root / "checkout" / "resolvers.json"
            fallback = root / "assets" / "resolvers.default.json"
            legacy.parent.mkdir()
            fallback.parent.mkdir()
            entry = {"match": "example.test", "command": ["python", "old_script.py"]}
            legacy.write_text(json.dumps([entry]), encoding="utf-8")
            fallback.write_text("[]", encoding="utf-8")
            with patch.multiple(SR, RESOLVERS_FILE=destination, LEGACY_RESOLVERS_FILE=legacy,
                                DEFAULT_RESOLVERS_FILE=fallback):
                result = SR.find_resolver("https://example.test/video")

            self.assertEqual(result, entry)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), [entry])
            self.assertTrue(legacy.exists())

    @patch("subprocess.run")
    def test_migrated_relative_script_resolves_from_checkout(self, mock_run):
        with temp_directory() as directory:
            root = Path(directory)
            data = root / "data"
            checkout = root / "checkout"
            data.mkdir()
            checkout.mkdir()
            script = checkout / "old_script.py"
            script.write_text("# legacy script", encoding="utf-8")
            config = data / "resolvers.json"
            mock_run.return_value = self._success_result()
            with patch.object(SR, "RESOLVERS_FILE", config), patch.object(SR, "checkout_root", return_value=checkout):
                SR.resolve_external("https://example.test/video", {},
                                    {"command": ["python", "old_script.py"]}, config)

            command = mock_run.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(Path(command[1]), script.resolve())

    @staticmethod
    def _success_result():
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"url": "https://cdn.example.test/video.m3u8"})
        result.stderr = ""
        return result

    @patch("subprocess.run")
    def test_browser_module_command_gets_source_path_and_frozen_route(self, mock_run):
        mock_run.return_value = self._success_result()
        resolver = {"command": ["{python}", "-m", "oh_streamer.sources.resolvers.resolve_browser"]}
        with patch.dict(os.environ, {"PYTHONPATH": "existing-path"}):
            SR.resolve_external("https://example.test/video", {}, resolver)
            command, kwargs = mock_run.call_args.args[0], mock_run.call_args.kwargs
            self.assertEqual(command[:3], [sys.executable, "-m", "oh_streamer.sources.resolvers.resolve_browser"])
            self.assertIn(str(Path(SR.__file__).resolve().parents[2]), kwargs["env"]["PYTHONPATH"])
            self.assertIn("existing-path", kwargs["env"]["PYTHONPATH"])

        with patch.object(SR.sys, "frozen", True, create=True):
            SR.resolve_external("https://example.test/video", {}, resolver)
            self.assertEqual(mock_run.call_args.args[0][:3], [sys.executable, "--resolver", "browser"])

    @patch("subprocess.run")
    def test_retired_builtin_script_migrates_but_custom_script_is_preserved(self, mock_run):
        mock_run.return_value = self._success_result()
        with temp_directory() as directory:
            config = Path(directory) / "resolvers.json"
            resolver = {"command": ["python", "resolvers/resolve_retired.py"]}
            with patch.object(SR, "RESOLVERS_FILE", config), patch.object(sys, "frozen", False, create=True):
                SR.resolve_external("https://example.test/video", {}, resolver, config)
                self.assertEqual(mock_run.call_args.args[0][:3],
                                 [sys.executable, "-m", "oh_streamer.sources.resolvers.resolve_browser"])
            custom = config.parent / "resolvers" / "resolve_retired.py"
            custom.parent.mkdir()
            custom.write_text("# user supplied resolver", encoding="utf-8")
            with patch.object(SR, "RESOLVERS_FILE", config):
                SR.resolve_external("https://example.test/video", {}, resolver, config)
                self.assertEqual(mock_run.call_args.args[0][1], str(custom.resolve()))


if __name__ == "__main__":
    main()
