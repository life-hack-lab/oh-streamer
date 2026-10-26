"""Run the Streamer unittest suite from any working directory."""

import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(ROOT)
    for source_path in (ROOT, ROOT / "src"):
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
