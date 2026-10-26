"""首次開啟前的 WebView2 runtime 檢查。只用標準函式庫；非 Windows 一律視為已安裝。"""

from __future__ import annotations

import sys

_HKLM_CLIENT = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_HKCU_CLIENT = r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_NOT_INSTALLED_VERSIONS = ("", "0.0.0.0")


def _pv_valid(winreg, hive, subkey: str) -> bool:
    try:
        with winreg.OpenKey(hive, subkey) as key:
            pv, _ = winreg.QueryValueEx(key, "pv")
    except OSError:
        return False
    return isinstance(pv, str) and pv.strip() not in _NOT_INSTALLED_VERSIONS


def webview2_installed() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import winreg
    except ImportError:
        return False
    return _pv_valid(winreg, winreg.HKEY_LOCAL_MACHINE, _HKLM_CLIENT) or _pv_valid(
        winreg, winreg.HKEY_CURRENT_USER, _HKCU_CLIENT
    )
