import importlib.util
import unittest
from pathlib import Path
from unittest import mock

_SPEC = importlib.util.spec_from_file_location(
    "check_no_secrets", Path(__file__).resolve().parents[1] / "scripts" / "check_no_secrets.py"
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class CheckNoSecretsTests(unittest.TestCase):
    def test_github_tokens_match(self):
        for prefix in ("ghp", "gho", "ghu", "ghs", "ghr"):
            line = f'token = "{prefix}_' + "a1B2" * 9 + '"'
            self.assertTrue(mod.scan_text(line, "x.py"), prefix)
        self.assertTrue(mod.scan_text("github_pat_" + "A1b2C3d4" * 3 + "_xyz", "x.py"))

    def test_azure_keys_match(self):
        self.assertTrue(mod.scan_text("Ocp-Apim-Subscription-Key: " + "0123456789abcdef" * 2, "x.py"))
        self.assertTrue(mod.scan_text('AZURE_TRANSLATOR_KEY = "' + "Ab1" * 28 + '"', "x.py"))
        self.assertTrue(mod.scan_text("TRANSLATOR_KEY=" + "f" * 32, "x.py"))

    def test_non_secrets_do_not_match(self):
        self.assertFalse(mod.scan_text("commit = '" + "0123456789abcdef" * 2 + "'", "x.py"))
        self.assertFalse(mod.scan_text("AZURE_TRANSLATOR_KEY = os.environ['K']", "x.py"))
        self.assertFalse(mod.scan_text("ghp_short", "x.py"))

    def test_git_failure_is_nonzero(self):
        bad = mock.Mock(returncode=128, stdout="", stderr="fatal: not a git repository")
        with mock.patch.object(mod.subprocess, "run", return_value=bad):
            self.assertNotEqual(mod.main([]), 0)

    def test_empty_list_is_nonzero(self):
        ok = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(mod.subprocess, "run", return_value=ok):
            self.assertNotEqual(mod.main([]), 0)


if __name__ == "__main__":
    unittest.main()
