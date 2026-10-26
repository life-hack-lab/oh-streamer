"""Install-root layout and current.json: paths, tolerant reads, atomic writes, READY / NEED_SETUP."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env import layout

LOCK_BYTES = b'version = 1\r\nrequires-python = "==3.13.*"\r\n'
LOCK_SHA = hashlib.sha256(LOCK_BYTES).hexdigest()


class LayoutTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve() / "install"
        self.root.mkdir()
        self.lock = Path(tmp.name).resolve() / "uv.lock"
        self.lock.write_bytes(LOCK_BYTES)

    def current(self, **overrides) -> layout.Current:
        fields = dict(lock_sha256=LOCK_SHA, flavor="cu128", verified=True, updated_at="2026-10-24T02:00:00+00:00")
        fields.update(overrides)
        return layout.Current(**fields)

    def make_env(self, sha: str = LOCK_SHA) -> Path:
        python = layout.python_in_env(layout.env_dir(self.root, sha))
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        return python

    def write_raw(self, text: str):
        (self.root / "current.json").write_text(text, encoding="utf-8")


class PathTests(LayoutTestCase):
    def test_lock_sha256_and_id(self):
        sha = layout.lock_sha256(self.lock)
        self.assertEqual(sha, LOCK_SHA)
        self.assertEqual(sha, sha.lower())
        self.assertEqual(len(sha), 64)
        self.assertEqual(layout.lock_id(sha), LOCK_SHA[:8])
        self.assertEqual(len(layout.lock_id(sha)), 8)

    def test_lock_sha256_is_over_raw_bytes(self):
        other = self.lock.with_name("other.lock")
        other.write_bytes(LOCK_BYTES.replace(b"\r\n", b"\n"))
        self.assertNotEqual(layout.lock_sha256(other), layout.lock_sha256(self.lock))

    def test_fixed_dirs(self):
        self.assertEqual(layout.app_dir(self.root), self.root / "app")
        self.assertEqual(layout.cache_dir(self.root), self.root / "uv-cache")
        self.assertEqual(layout.logs_dir(self.root), self.root / "logs")

    def test_env_dir(self):
        self.assertEqual(layout.env_dir(self.root, LOCK_SHA), self.root / "envs" / LOCK_SHA[:8])

    def test_python_in_env(self):
        env = Path("somewhere") / "env"
        expected = env / "Scripts" / "python.exe" if sys.platform == "win32" else env / "bin" / "python"
        self.assertEqual(layout.python_in_env(env), expected)
        with mock.patch.object(layout.sys, "platform", "win32"):
            self.assertEqual(layout.python_in_env(env), env / "Scripts" / "python.exe")
        with mock.patch.object(layout.sys, "platform", "linux"):
            self.assertEqual(layout.python_in_env(env), env / "bin" / "python")

    def test_accepts_str_paths(self):
        self.assertEqual(layout.env_dir(str(self.root), LOCK_SHA), layout.env_dir(self.root, LOCK_SHA))


class CurrentFileTests(LayoutTestCase):
    def test_missing_file(self):
        self.assertIsNone(layout.read_current(self.root))
        self.assertIsNone(layout.read_current(self.root / "does-not-exist"))

    def test_round_trip(self):
        for flavor in ("cu128", None):
            current = self.current(flavor=flavor, verified=flavor is not None)
            layout.write_current(self.root, current)
            self.assertEqual(layout.read_current(self.root), current)

    def test_write_creates_missing_root_and_leaves_no_temp(self):
        root = self.root / "not" / "yet"
        layout.write_current(root, self.current())
        self.assertEqual([p.name for p in root.iterdir()], ["current.json"])
        self.assertEqual(json.loads((root / "current.json").read_text(encoding="utf-8"))["schema"], 1)

    def test_write_overwrites(self):
        layout.write_current(self.root, self.current(flavor=None, verified=False))
        layout.write_current(self.root, self.current())
        self.assertEqual(layout.read_current(self.root), self.current())

    def test_bad_json(self):
        for text in ("", "{", "not json", "[1, 2]", "null", "42", '"x"', "[" * 100000):
            self.write_raw(text)
            self.assertIsNone(layout.read_current(self.root), text[:20])

    def test_non_utf8_bytes(self):
        (self.root / "current.json").write_bytes(b"\xff\xfe\x00bad")
        self.assertIsNone(layout.read_current(self.root))

    def test_missing_fields(self):
        good = {"schema": 1, "lock_sha256": LOCK_SHA, "flavor": "cu128", "verified": True, "updated_at": "t"}
        self.write_raw(json.dumps(good))
        self.assertIsNotNone(layout.read_current(self.root))
        for key in good:
            partial = {k: v for k, v in good.items() if k != key}
            self.write_raw(json.dumps(partial))
            self.assertIsNone(layout.read_current(self.root), key)

    def test_unknown_schema(self):
        good = {"schema": 1, "lock_sha256": LOCK_SHA, "flavor": "cu128", "verified": True, "updated_at": "t"}
        for schema in (2, 0, "1", 1.0, True, None):
            self.write_raw(json.dumps({**good, "schema": schema}))
            self.assertIsNone(layout.read_current(self.root), repr(schema))

    def test_wrong_field_types(self):
        good = {"schema": 1, "lock_sha256": LOCK_SHA, "flavor": "cu128", "verified": True, "updated_at": "t"}
        for key, value in (("lock_sha256", 5), ("flavor", 5), ("verified", "yes"), ("verified", 1), ("updated_at", None)):
            self.write_raw(json.dumps({**good, key: value}))
            self.assertIsNone(layout.read_current(self.root), f"{key}={value!r}")

    def test_extra_fields_are_ignored(self):
        good = {"schema": 1, "lock_sha256": LOCK_SHA, "flavor": None, "verified": False, "updated_at": "t", "note": "x"}
        self.write_raw(json.dumps(good))
        self.assertEqual(layout.read_current(self.root), self.current(flavor=None, verified=False, updated_at="t"))

    def test_current_is_frozen(self):
        with self.assertRaises(Exception):
            self.current().verified = False

    def test_atomic_write_keeps_old_file_when_replace_fails(self):
        old = self.current(flavor=None, verified=False)
        layout.write_current(self.root, old)
        before = (self.root / "current.json").read_bytes()
        with mock.patch.object(layout.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                layout.write_current(self.root, self.current())
        self.assertEqual((self.root / "current.json").read_bytes(), before)
        self.assertEqual(layout.read_current(self.root), old)
        self.assertEqual([p.name for p in self.root.iterdir()], ["current.json"])      # 暫存檔已清掉

    def test_atomic_write_keeps_old_file_when_serialising_fails(self):
        old = self.current()
        layout.write_current(self.root, old)
        with mock.patch.object(layout.json, "dump", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                layout.write_current(self.root, self.current(flavor=None, verified=False))
        self.assertEqual(layout.read_current(self.root), old)
        self.assertEqual([p.name for p in self.root.iterdir()], ["current.json"])

    def test_atomic_write_without_previous_file(self):
        with mock.patch.object(layout.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                layout.write_current(self.root, self.current())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIsNone(layout.read_current(self.root))

    def test_temp_file_is_in_the_same_directory(self):
        seen = []
        real_replace = layout.os.replace

        def spy(src, dst):
            seen.append((Path(src).parent, Path(dst)))
            return real_replace(src, dst)

        with mock.patch.object(layout.os, "replace", side_effect=spy):
            layout.write_current(self.root, self.current())
        self.assertEqual(seen, [(self.root, self.root / "current.json")])


class StatusTests(LayoutTestCase):
    def test_ready(self):
        self.make_env()
        layout.write_current(self.root, self.current())
        self.assertEqual(layout.status(self.root, self.lock), "READY")

    def test_no_current(self):
        self.make_env()
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_broken_current(self):
        self.make_env()
        self.write_raw("{")
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_lock_changed(self):
        self.make_env()
        layout.write_current(self.root, self.current())
        self.lock.write_bytes(LOCK_BYTES + b"# new dependency\n")
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_lock_changed_even_if_new_env_exists(self):
        new_lock = LOCK_BYTES + b"# new dependency\n"
        self.make_env(hashlib.sha256(new_lock).hexdigest())
        layout.write_current(self.root, self.current())                  # 仍指向舊 lock
        self.lock.write_bytes(new_lock)
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_base_env_only(self):
        self.make_env()
        layout.write_current(self.root, self.current(flavor=None, verified=True))
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_not_verified(self):
        self.make_env()
        layout.write_current(self.root, self.current(verified=False))
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_env_missing(self):
        layout.write_current(self.root, self.current())
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_env_without_python(self):
        layout.env_dir(self.root, LOCK_SHA).mkdir(parents=True)
        layout.write_current(self.root, self.current())
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_python_is_a_directory(self):
        layout.python_in_env(layout.env_dir(self.root, LOCK_SHA)).mkdir(parents=True)
        layout.write_current(self.root, self.current())
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_lock_missing(self):
        self.make_env()
        layout.write_current(self.root, self.current())
        self.lock.unlink()
        self.assertEqual(layout.status(self.root, self.lock), "NEED_SETUP")

    def test_status_writes_nothing(self):
        self.make_env()
        layout.write_current(self.root, self.current())
        listing = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*"))
        layout.status(self.root, self.lock)
        self.assertEqual(sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")), listing)


if __name__ == "__main__":
    unittest.main()
