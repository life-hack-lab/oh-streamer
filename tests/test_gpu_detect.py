"""gpu_detect: WMI / nvidia-smi parsing with fake command output, merging, and every branch of recommend()."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env import gpu_detect
from oh_streamer.runtime_env.gpu_detect import (Detection, GpuInfo, detect, query_nvidia_smi, query_wmi_gpus, recommend,
                                                vendor_of)

RTX_5060 = "NVIDIA GeForce RTX 5060 Laptop GPU"
INTEL_ARC = "Intel(R) Arc(TM) Graphics"


def nv(name=RTX_5060, driver="576.02", cc=(12, 0)):
    return GpuInfo(name, "nvidia", driver, cc)


def other(vendor, name="x"):
    return GpuInfo(name, vendor, None, None)


def wmi_json(*items):
    rows = [{"Name": n, "DriverVersion": d} for n, d in items]
    return json.dumps(rows[0] if len(rows) == 1 else rows, separators=(",", ":"))


def fake_run(wmi=None, smi=None, calls=None):
    """Routes powershell / nvidia-smi calls; an outcome of None means 'command failed', an Exception is raised."""
    def run(args):
        if calls is not None:
            calls.append(list(args))
        outcome = wmi if args[0].lower().startswith("powershell") else smi
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return run


class WmiTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(gpu_detect, "_is_windows", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_object(self):
        gpus = query_wmi_gpus(fake_run(wmi=wmi_json((INTEL_ARC, "32.0.101.6874"))))
        self.assertEqual(gpus, [GpuInfo(INTEL_ARC, "intel", "32.0.101.6874", None)])

    def test_array_and_nvidia_driver_is_converted_to_marketing_version(self):
        gpus = query_wmi_gpus(fake_run(wmi=wmi_json((RTX_5060, "32.0.15.7602"), (INTEL_ARC, "32.0.101.6874"))))
        self.assertEqual([g.vendor for g in gpus], ["nvidia", "intel"])
        self.assertEqual(gpus[0].driver_version, "576.02")
        self.assertEqual(gpus[1].driver_version, "32.0.101.6874")

    def test_unparsable_nvidia_driver_becomes_none(self):
        gpus = query_wmi_gpus(fake_run(wmi=wmi_json((RTX_5060, "weird"))))
        self.assertIsNone(gpus[0].driver_version)

    def test_empty_and_blank_output(self):
        self.assertEqual(query_wmi_gpus(fake_run(wmi="")), [])
        self.assertEqual(query_wmi_gpus(fake_run(wmi="  \r\n")), [])

    def test_invalid_json(self):
        self.assertEqual(query_wmi_gpus(fake_run(wmi="{not json")), [])
        self.assertEqual(query_wmi_gpus(fake_run(wmi="Get-CimInstance : access denied")), [])

    def test_json_of_the_wrong_shape(self):
        for text in ("5", '"text"', "null", "[1, 2]", '[{"DriverVersion": "1.0"}]', '{"Name": null}'):
            with self.subTest(text=text):
                self.assertEqual(query_wmi_gpus(fake_run(wmi=text)), [])

    def test_bom_prefix_is_tolerated(self):
        gpus = query_wmi_gpus(fake_run(wmi="﻿" + wmi_json((INTEL_ARC, None))))
        self.assertEqual(gpus, [GpuInfo(INTEL_ARC, "intel", None, None)])

    def test_run_failure_returns_empty(self):
        self.assertEqual(query_wmi_gpus(fake_run(wmi=None)), [])
        self.assertEqual(query_wmi_gpus(fake_run(wmi=FileNotFoundError("powershell"))), [])

    def test_command_asks_for_name_and_driver_as_compact_json(self):
        calls = []
        query_wmi_gpus(fake_run(wmi="", calls=calls))
        command = calls[0][-1]
        self.assertIn("Get-CimInstance Win32_VideoController", command)
        self.assertIn("Select-Object Name,DriverVersion", command)
        self.assertIn("ConvertTo-Json -Compress", command)

    def test_non_windows_returns_empty_without_running_anything(self):
        calls = []
        with patch.object(gpu_detect, "_is_windows", return_value=False):
            self.assertEqual(query_wmi_gpus(fake_run(wmi=wmi_json((RTX_5060, "32.0.15.7602")), calls=calls)), [])
        self.assertEqual(calls, [])


class NvidiaSmiTests(unittest.TestCase):
    def test_normal_rows(self):
        text = f"{RTX_5060}, 576.02, 12.0\nNVIDIA GeForce GTX 1060, 576.02, 6.1\n"
        self.assertEqual(query_nvidia_smi(fake_run(smi=text)), [
            GpuInfo(RTX_5060, "nvidia", "576.02", (12, 0)),
            GpuInfo("NVIDIA GeForce GTX 1060", "nvidia", "576.02", (6, 1)),
        ])

    def test_windows_line_endings(self):
        self.assertEqual(query_nvidia_smi(fake_run(smi=f"{RTX_5060}, 576.02, 12.0\r\n")),
                         [GpuInfo(RTX_5060, "nvidia", "576.02", (12, 0))])

    def test_compute_cap_column_missing(self):
        self.assertEqual(query_nvidia_smi(fake_run(smi=f"{RTX_5060}, 576.02\n")),
                         [GpuInfo(RTX_5060, "nvidia", "576.02", None)])

    def test_compute_cap_not_available(self):
        for cell in ("[N/A]", "N/A", "", "[Not Supported]", "garbage"):
            with self.subTest(cell=cell):
                gpus = query_nvidia_smi(fake_run(smi=f"{RTX_5060}, 576.02, {cell}\n"))
                self.assertEqual(gpus, [GpuInfo(RTX_5060, "nvidia", "576.02", None)])

    def test_driver_not_available(self):
        self.assertEqual(query_nvidia_smi(fake_run(smi=f"{RTX_5060}, [N/A], 12.0\n")),
                         [GpuInfo(RTX_5060, "nvidia", None, (12, 0))])

    def test_command_missing(self):
        self.assertEqual(query_nvidia_smi(fake_run(smi=FileNotFoundError("nvidia-smi"))), [])
        self.assertEqual(query_nvidia_smi(fake_run(smi=OSError("denied"))), [])
        self.assertEqual(query_nvidia_smi(fake_run(smi=None)), [])

    def test_no_device_or_error_text(self):
        for text in ("", "\n", "No devices were found", "NVIDIA-SMI has failed because it couldn't communicate"):
            with self.subTest(text=text):
                self.assertEqual(query_nvidia_smi(fake_run(smi=text)), [])

    def test_command_line(self):
        calls = []
        query_nvidia_smi(fake_run(smi="", calls=calls))
        self.assertEqual(calls[0], ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                                    "--format=csv,noheader,nounits"])


class DetectTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(gpu_detect, "_is_windows", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_smi_fills_driver_and_compute_capability_of_the_wmi_entry(self):
        run = fake_run(wmi=wmi_json((RTX_5060, "32.0.15.7602"), (INTEL_ARC, "32.0.101.6874")),
                       smi=f"{RTX_5060}, 576.02, 12.0\n")
        found = detect(run)
        self.assertTrue(found.nvidia_smi_ok)
        self.assertEqual(found.gpus, [GpuInfo(RTX_5060, "nvidia", "576.02", (12, 0)),
                                      GpuInfo(INTEL_ARC, "intel", "32.0.101.6874", None)])

    def test_smi_unavailable_keeps_wmi_data(self):
        found = detect(fake_run(wmi=wmi_json((RTX_5060, "32.0.15.7602")), smi=FileNotFoundError("nvidia-smi")))
        self.assertFalse(found.nvidia_smi_ok)
        self.assertEqual(found.gpus, [GpuInfo(RTX_5060, "nvidia", "576.02", None)])

    def test_wmi_unavailable_still_reports_the_smi_gpu(self):
        found = detect(fake_run(wmi=None, smi=f"{RTX_5060}, 576.02, 12.0\n"))
        self.assertTrue(found.nvidia_smi_ok)
        self.assertEqual(found.gpus, [GpuInfo(RTX_5060, "nvidia", "576.02", (12, 0))])

    def test_everything_unavailable(self):
        self.assertEqual(detect(fake_run(wmi=None, smi=None)), Detection([], False))

    def test_two_nvidia_cards_pair_by_name(self):
        run = fake_run(wmi=wmi_json(("NVIDIA GeForce GTX 1060", "32.0.15.7602"), (RTX_5060, "32.0.15.7602")),
                       smi=f"{RTX_5060}, 576.02, 12.0\nNVIDIA GeForce GTX 1060, 576.02, 6.1\n")
        by_name = {g.name: g for g in detect(run).gpus}
        self.assertEqual(by_name[RTX_5060].compute_capability, (12, 0))
        self.assertEqual(by_name["NVIDIA GeForce GTX 1060"].compute_capability, (6, 1))

    def test_single_card_pairs_even_when_names_differ(self):
        run = fake_run(wmi=wmi_json(("NVIDIA GeForce RTX 5060 Laptop GPU", "32.0.15.7602")),
                       smi="NVIDIA GeForce RTX 5060 Laptop, 576.02, 12.0\n")
        found = detect(run)
        self.assertEqual(len(found.gpus), 1)
        self.assertEqual(found.gpus[0].name, "NVIDIA GeForce RTX 5060 Laptop GPU")
        self.assertEqual(found.gpus[0].compute_capability, (12, 0))

    def test_smi_card_missing_from_wmi_is_added(self):
        found = detect(fake_run(wmi=wmi_json((INTEL_ARC, "1.0")), smi=f"{RTX_5060}, 576.02, 12.0\n"))
        self.assertEqual({g.vendor for g in found.gpus}, {"nvidia", "intel"})

    def test_detect_then_recommend(self):
        run = fake_run(wmi=wmi_json((RTX_5060, "32.0.15.7602"), (INTEL_ARC, "32.0.101.6874")),
                       smi=f"{RTX_5060}, 576.02, 12.0\n")
        rec = recommend(detect(run).gpus)
        self.assertEqual((rec.flavor, rec.status, rec.warnings), ("nvidia", "ok", []))


class VendorTests(unittest.TestCase):
    def test_vendor_of(self):
        cases = {
            "NVIDIA GeForce RTX 3060 Laptop GPU": "nvidia", "NVIDIA RTX A2000": "nvidia", "Quadro P1000": "nvidia",
            "Intel(R) UHD Graphics 770": "intel", "Intel(R) Arc(TM) A770 Graphics": "intel",
            "AMD Radeon(TM) Graphics": "amd", "Radeon RX 7800 XT": "amd", "ATI Radeon HD 4850": "amd",
            "Microsoft Basic Display Adapter": "other", "Parsec Virtual Display Adapter": "other",
        }
        for name, vendor in cases.items():
            with self.subTest(name=name):
                self.assertEqual(vendor_of(name), vendor)


class RecommendTests(unittest.TestCase):
    def check(self, gpus, flavor, status, warnings=None, **kwargs):
        rec = recommend(gpus, **kwargs)
        self.assertEqual((rec.flavor, rec.status), (flavor, status))
        if warnings is not None:
            self.assertEqual(len(rec.warnings), warnings)

    # a. modern NVIDIA
    def test_blackwell_with_new_driver_is_ok(self):
        self.check([nv()], "nvidia", "ok", 0)

    def test_turing_minimum_capability_is_ok(self):
        self.check([nv("NVIDIA GeForce GTX 1650", "571.10", (7, 5))], "nvidia", "ok", 0)

    def test_driver_570_exactly_is_ok(self):
        self.check([nv(driver="570.00")], "nvidia", "ok", 0)

    def test_driver_just_below_570_is_too_old_but_keeps_flavor(self):
        self.check([nv(driver="569.99")], "nvidia", "driver_too_old", 1)

    def test_old_driver_is_too_old_with_one_warning(self):
        self.check([nv(driver="472.12", cc=(8, 6))], "nvidia", "driver_too_old", 1)

    def test_driver_unknown_is_not_blocked(self):
        self.check([nv(driver=None)], "nvidia", "ok", 0)

    # b. Maxwell / Pascal and the other capabilities below Turing
    def test_pascal_is_legacy_deferred(self):
        self.check([nv("NVIDIA GeForce GTX 1060", "576.02", (6, 1))], None, "legacy_nvidia_deferred", 0)

    def test_maxwell_is_legacy_deferred(self):
        self.check([nv("NVIDIA GeForce GTX 750 Ti", "576.02", (5, 0))], None, "legacy_nvidia_deferred", 0)

    def test_volta_is_legacy_deferred(self):
        self.check([nv("NVIDIA TITAN V", "576.02", (7, 0))], None, "legacy_nvidia_deferred", 0)

    def test_kepler_has_no_usable_flavor(self):
        self.check([nv("NVIDIA GeForce GTX 780", "470.00", (3, 5))], None, "cpu_only", 0)

    # c. NVIDIA without compute capability: judged by name
    def test_name_only_rtx_and_gtx16_are_modern(self):
        for name in ("NVIDIA GeForce RTX 3060 Laptop GPU", "NVIDIA GeForce RTX 2080 Ti", "NVIDIA GeForce RTX 4090",
                     "NVIDIA GeForce RTX 5070", "NVIDIA GeForce GTX 1650", "NVIDIA GeForce GTX 1660 Ti",
                     "Quadro RTX 4000", "NVIDIA RTX A2000"):
            with self.subTest(name=name):
                self.check([nv(name, "576.02", None)], "nvidia", "ok", 0)

    def test_name_only_gtx9_and_gtx10_are_legacy(self):
        for name in ("NVIDIA GeForce GTX 960", "NVIDIA GeForce GTX 960M", "NVIDIA GeForce GTX 1050 Ti",
                     "NVIDIA GeForce GTX 1060 with Max-Q Design", "NVIDIA GeForce GTX 1080 Ti"):
            with self.subTest(name=name):
                self.check([nv(name, "576.02", None)], None, "legacy_nvidia_deferred", 0)

    def test_name_only_unrecognised_nvidia_is_unknown(self):
        for name in ("NVIDIA Tesla K80", "NVIDIA GeForce MX450", "NVIDIA Something New"):
            with self.subTest(name=name):
                self.check([nv(name, "576.02", None)], None, "unknown_nvidia", 0)

    def test_name_only_modern_still_checks_driver(self):
        self.check([nv("NVIDIA GeForce RTX 3060", "511.65", None)], "nvidia", "driver_too_old", 1)

    def test_compute_capability_beats_the_name(self):
        self.check([nv("NVIDIA GeForce GTX 1060", "576.02", (7, 5))], "nvidia", "ok", 0)

    # d. no NVIDIA
    def test_intel_only_is_not_offered(self):
        self.check([other("intel", INTEL_ARC)], None, "intel_not_offered", 0)

    def test_amd_only_is_not_offered(self):
        self.check([other("amd", "AMD Radeon RX 7800 XT")], None, "amd_not_offered", 0)

    def test_intel_and_amd_report_intel(self):
        self.check([other("amd"), other("intel")], None, "intel_not_offered", 0)

    # e. nothing usable
    def test_no_gpu_is_cpu_only(self):
        self.check([], None, "cpu_only", 0)

    def test_other_vendor_is_cpu_only(self):
        self.check([other("other", "Microsoft Basic Display Adapter")], None, "cpu_only", 0)

    # f. supported filter
    def test_supported_intel_recommends_intel(self):
        self.check([other("intel")], "intel", "ok", 0, supported=("nvidia", "intel"))

    def test_supported_amd_recommends_amd(self):
        self.check([other("amd")], "amd", "ok", 0, supported=("nvidia", "amd"))

    def test_amd_still_not_offered_when_only_intel_is_supported(self):
        self.check([other("amd")], None, "amd_not_offered", 0, supported=("nvidia", "intel"))

    def test_nvidia_outside_supported_is_not_offered(self):
        self.check([nv()], None, "cpu_only", 0, supported=())

    # g. NVIDIA beats the iGPU
    def test_nvidia_plus_intel_igpu_picks_nvidia(self):
        self.check([other("intel", INTEL_ARC), nv()], "nvidia", "ok", 0)

    def test_nvidia_plus_intel_even_when_intel_is_supported(self):
        self.check([other("intel"), nv()], "nvidia", "ok", 0, supported=("nvidia", "intel"))

    def test_nvidia_plus_amd_igpu_picks_nvidia(self):
        self.check([other("amd"), nv()], "nvidia", "ok", 0)

    def test_legacy_nvidia_plus_intel_stays_legacy(self):
        self.check([other("intel"), nv("NVIDIA GeForce GTX 1060", "576.02", (6, 1))],
                   None, "legacy_nvidia_deferred", 0, supported=("nvidia", "intel"))

    def test_unknown_nvidia_plus_intel_stays_unknown(self):
        self.check([other("intel"), nv("NVIDIA Something New", "576.02", None)], None, "unknown_nvidia", 0)

    def test_two_nvidia_cards_modern_one_wins(self):
        self.check([nv("NVIDIA GeForce GTX 1060", "576.02", (6, 1)), nv()], "nvidia", "ok", 0)

    def test_driver_is_read_from_any_modern_card(self):
        self.check([nv(driver=None), nv("NVIDIA GeForce RTX 3060", "500.00", (8, 6))], "nvidia", "driver_too_old", 1)


if __name__ == "__main__":
    unittest.main()
