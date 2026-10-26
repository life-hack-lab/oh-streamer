"""把支援用的診斷資料打包成 zip。只收 logs 最近幾個檔、current.json、偵測結果與前置檢查；不收設定、佇列或其他資料。"""

from __future__ import annotations

import json
import platform
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from .preflight import webview2_installed

LOG_LIMIT = 5


def _recent_logs(logs_dir: Path, limit: int) -> list[Path]:
    try:
        files = [p for p in logs_dir.iterdir() if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def export_diagnostics(install_root, dest_dir, *, detection: dict | None = None, now: datetime | None = None) -> Path:
    install_root = Path(install_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    zip_path = dest_dir / f"oh_streamer-diagnostics-{stamp}.zip"

    preflight = {
        "webview2_installed": webview2_installed(),
        "sys_platform": sys.platform,
        "python_version": platform.python_version(),
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for log in _recent_logs(install_root / "logs", LOG_LIMIT):
            zf.write(log, arcname=f"logs/{log.name}")
        current = install_root / "current.json"
        if current.is_file():
            zf.write(current, arcname="current.json")
        zf.writestr("detection.json", json.dumps(detection, ensure_ascii=False, indent=2, default=str))
        zf.writestr("preflight.json", json.dumps(preflight, ensure_ascii=False, indent=2))
    return zip_path
