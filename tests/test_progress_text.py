import unittest

from oh_streamer.runtime_env import progress_text as pt

MB = 2 ** 20
GB = 2 ** 30


class FormatTests(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(pt.fmt_bytes(512), "1 KB")
        self.assertEqual(pt.fmt_bytes(260 * MB), "260 MB")
        self.assertEqual(pt.fmt_bytes(3.2 * GB), "3.2 GB")

    def test_pair_uses_the_total_unit(self):
        self.assertEqual(pt.fmt_pair(120 * MB, 260 * MB), "120 / 260 MB")
        self.assertEqual(pt.fmt_pair(900 * MB, 3.2 * GB), "0.9 / 3.2 GB")

    def test_speed(self):
        self.assertEqual(pt.fmt_speed(5.24 * MB), "5.2 MB/s")
        self.assertEqual(pt.fmt_speed(300 * 1024), "300 KB/s")

    def test_eta_is_coarse(self):
        self.assertEqual(pt.fmt_eta(4), "剩不到 10 秒")
        self.assertEqual(pt.fmt_eta(32), "剩約 30 秒")
        self.assertEqual(pt.fmt_eta(157), "剩約 2 分 40 秒")
        self.assertEqual(pt.fmt_eta(120), "剩約 2 分")
        self.assertEqual(pt.fmt_eta(3 * 3600 + 5 * 60), "剩約 3 小時 5 分")


class DescribeTests(unittest.TestCase):
    def test_unmeasured_is_a_moving_bar(self):
        self.assertEqual(pt.describe(None), (None, pt.UNMEASURED))
        self.assertEqual(pt.describe({"downloaded_bytes": 0, "total_bytes": 0}), (None, pt.UNMEASURED))

    def test_downloading(self):
        snap = {"downloaded_bytes": 120 * MB, "total_bytes": 260 * MB, "speed_bps": 5.24 * MB, "eta_s": 27}
        fraction, text = pt.describe(snap)
        self.assertAlmostEqual(fraction, 120 / 260)
        self.assertEqual(text, "已下載 120 / 260 MB（46%） · 5.2 MB/s · 剩約 25 秒")

    def test_no_speed_yet(self):
        _, text = pt.describe({"downloaded_bytes": 1 * MB, "total_bytes": 260 * MB, "speed_bps": None, "eta_s": None})
        self.assertTrue(text.endswith("正在估計剩餘時間…"))
        self.assertNotIn("MB/s", text)

    def test_download_finished_means_installing(self):
        fraction, text = pt.describe({"downloaded_bytes": 270 * MB, "total_bytes": 260 * MB, "speed_bps": None, "eta_s": 0})
        self.assertIsNone(fraction)
        self.assertEqual(text, f"已下載 260 MB，{pt.INSTALLING}")


if __name__ == "__main__":
    unittest.main()
