from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform import player_branding as branding


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProcess:
    def __init__(self, pid, running_polls=2):
        self.pid = pid
        self.running_polls = running_polls
        self.poll_calls = 0
        self.terminate_calls = 0

    def poll(self):
        self.poll_calls += 1
        return None if self.poll_calls <= self.running_polls else 0

    def terminate(self):
        self.terminate_calls += 1


class FakeApi:
    def __init__(self, hwnd=0x123456789):
        self.hwnd = hwnd
        self.find_calls = []
        self.load_calls = []
        self.set_calls = []
        self.destroy_calls = []

    def find_window(self, pid):
        self.find_calls.append(pid)
        return self.hwnd

    def window_dpi(self, hwnd):
        self.dpi_hwnd = hwnd
        return 144

    def load_icon(self, path, width, height):
        self.load_calls.append((Path(path), width, height))
        return 0x100000001 if len(self.load_calls) == 1 else 0x200000002

    def set_icon(self, hwnd, which, icon):
        self.set_calls.append((hwnd, which, icon))
        return True

    def destroy_icon(self, icon):
        self.destroy_calls.append(icon)


class MissingWindowApi(FakeApi):
    def find_window(self, pid):
        self.find_calls.append(pid)
        return None


class PlayerBrandingTests(unittest.TestCase):
    def test_non_windows_is_noop(self):
        with patch.object(branding.sys, "platform", "darwin"), \
                patch.object(branding.threading, "Thread") as thread:
            branding.apply_player_branding(FakeProcess(42))
        thread.assert_not_called()

    def test_windows_starts_nonblocking_daemon_only_for_real_pid(self):
        process = FakeProcess(42)
        with patch.object(branding.sys, "platform", "win32"), \
                patch.object(branding.threading, "Thread") as thread:
            branding.apply_player_branding(process)
        thread.assert_called_once()
        args, kwargs = thread.call_args
        self.assertEqual(kwargs["target"], branding._run_branding_worker)
        self.assertEqual(args, ())
        self.assertEqual(kwargs["args"], (process,))
        self.assertTrue(kwargs["daemon"])
        thread.return_value.start.assert_called_once()

        with patch.object(branding.sys, "platform", "win32"), \
                patch.object(branding.threading, "Thread") as thread:
            branding.apply_player_branding(object())
        thread.assert_not_called()

    def test_worker_targets_pid_sets_dpi_sized_icons_and_releases_on_exit(self):
        process = FakeProcess(9123, running_polls=2)
        api = FakeApi()
        clock = FakeClock()

        with patch.object(branding, "set_window_app_id") as set_identity:
            branding._run_branding_worker(
                process,
                api=api,
                icon_path=Path("fake/app.ico"),
                clock=clock.monotonic,
                sleep=clock.sleep,
                process_poll_interval=0.05,
            )
        set_identity.assert_called_once_with(api.hwnd)

        self.assertTrue(api.find_calls)
        self.assertEqual(set(api.find_calls), {9123})
        self.assertEqual(api.dpi_hwnd, api.hwnd)
        self.assertEqual(api.load_calls, [
            (Path("fake/app.ico"), 24, 24),
            (Path("fake/app.ico"), 48, 48),
        ])
        self.assertEqual(api.set_calls, [
            (api.hwnd, branding._WindowsApi.ICON_SMALL, 0x100000001),
            (api.hwnd, branding._WindowsApi.ICON_BIG, 0x200000002),
        ])
        self.assertEqual(set(api.destroy_calls), {0x100000001, 0x200000002})
        self.assertEqual(process.terminate_calls, 0)

    def test_identity_is_applied_even_when_icons_cannot_load(self):
        api = FakeApi()
        api.load_icon = lambda *args: None
        with patch.object(branding, "set_window_app_id") as set_identity:
            branding._run_branding_worker(FakeProcess(42), api=api)
        set_identity.assert_called_once_with(api.hwnd)
        self.assertEqual(api.set_calls, [])

    def test_window_lookup_is_bounded_and_stops_when_process_exits(self):
        clock = FakeClock()
        api = MissingWindowApi()
        process = FakeProcess(77, running_polls=100)

        branding._run_branding_worker(
            process,
            api=api,
            icon_path=Path("unused.ico"),
            lookup_timeout=0.2,
            lookup_interval=0.1,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertEqual(api.find_calls, [77, 77, 77])
        self.assertEqual(clock.now, 0.2)
        self.assertEqual(api.load_calls, [])

        exited_process = FakeProcess(88, running_polls=0)
        exited_api = MissingWindowApi()
        branding._run_branding_worker(
            exited_process,
            api=exited_api,
            icon_path=Path("unused.ico"),
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(exited_api.find_calls, [])


if __name__ == "__main__":
    unittest.main()
