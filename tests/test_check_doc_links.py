import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_doc_links", Path(__file__).resolve().parents[1] / "scripts" / "check_doc_links.py"
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

EXISTING = {"docs/a.md", "docs/docs_a.md", "docs/sub/b.md", "tools/x.py", "scripts/s.py", "benchmarks/bm.py", "README.md"}


def refs(files):
    return [(f, r) for f, _, r in mod.find_missing(files, EXISTING)]


class FindMissingTests(unittest.TestCase):
    def test_valid_relative_link(self):
        self.assertEqual(refs({"docs/c.md": "[a](a.md) [b](sub/b.md#x) [r](../README.md)"}), [])

    def test_missing_link(self):
        self.assertEqual(refs({"docs/c.md": "[a](gone.md)"}), [("docs/c.md", "gone.md")])

    def test_ignores_urls_and_anchors(self):
        self.assertEqual(refs({"docs/c.md": "[a](https://x.y/z.md) [b](#top) [m](mailto:a@b.c)"}), [])

    def test_outside_repo_and_ellipsis(self):
        self.assertEqual(refs({"docs/c.md": "[a](../../other/x.md) [b](...)"}), [])

    def test_allowlist(self):
        self.assertEqual(refs({"docs/c.md": "[a](../lada/x.md) see testdata/foo.md and %LOCALAPPDATA%/x"}), [])

    def test_bare_mentions(self):
        text = "docs/gone.md docs_gone.md docs/nodir/ benchmarks/no.py tools/no.py scripts/no.py"
        self.assertEqual(len(refs({"README.md": text})), 6)

    def test_bare_existing(self):
        text = "docs/a.md docs_a.md docs/sub/ benchmarks/bm.py tools/x.py scripts/s.py"
        self.assertEqual(refs({"README.md": text}), [])

    def test_source_files_bare_only(self):
        self.assertEqual(refs({"src/m.py": "# see docs/gone.md [x](nothing.md)"}), [("src/m.py", "docs/gone.md")])


if __name__ == "__main__":
    unittest.main()
