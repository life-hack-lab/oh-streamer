"""Test package: redirect the user data dir to a throw-away temp folder.

platform_support.data_dir() (and the subtitle/model helpers) resolve %LOCALAPPDATA% at import
time, so this must run before any test module imports the app. Without it the suite writes
player-queue-test-*.json, perf profiles, logs etc. into the user's real %LOCALAPPDATA%\oh_streamer.
"""

import atexit
import os
import shutil
import tempfile

if not os.environ.get("OH_STREAMER_TESTS_KEEP_REAL_DATA_DIR"):
    _TEST_APPDATA = tempfile.mkdtemp(prefix="oh_tests_appdata_")
    os.environ["LOCALAPPDATA"] = _TEST_APPDATA
    os.environ["APPDATA"] = _TEST_APPDATA
    os.environ["XDG_DATA_HOME"] = _TEST_APPDATA
    atexit.register(shutil.rmtree, _TEST_APPDATA, ignore_errors=True)
