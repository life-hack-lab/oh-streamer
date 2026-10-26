r"""Post-install torch self-check: runs a short snippet in the new env and reads its last stdout line as JSON.

Read-only: nothing is installed, deleted or changed. Stdlib only: it runs before torch exists in the base env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

_CREATE_NO_WINDOW = 0x08000000                  # setup mode runs under pythonw: the console python child must not pop up a window

SNIPPET = """
import json
import torch
import torchvision
from torchvision.ops import deform_conv2d

result = {"cuda_available": bool(torch.cuda.is_available()), "device_name": None, "deform_conv2d_ok": False, "message": ""}
if not result["cuda_available"]:
    result["message"] = "CUDA 不可用"
else:
    try:
        result["device_name"] = torch.cuda.get_device_name(0)
        x = torch.ones(1, 8, 16, 16, device="cuda")
        w = torch.ones(4, 8, 3, 3, device="cuda")
        offset = torch.zeros(1, 18, 14, 14, device="cuda")
        y = deform_conv2d(x, offset, w)
        result["deform_conv2d_ok"] = tuple(y.shape) == (1, 4, 14, 14) and bool(torch.isfinite(y).all())
        if not result["deform_conv2d_ok"]:
            result["message"] = "deform_conv2d 輸出形狀或數值不正確"
    except Exception as exc:
        result["message"] = "deform_conv2d 失敗：" + repr(exc)[:300]
print(json.dumps(result))
"""


@dataclass
class SelfCheckResult:
    ok: bool
    cuda_available: bool
    device_name: str | None
    deform_conv2d_ok: bool
    message: str


def _fail(message: str) -> SelfCheckResult:
    return SelfCheckResult(False, False, None, False, message)


def run_self_check(python_exe, timeout_s: float = 300, *, snippet: str = SNIPPET) -> SelfCheckResult:
    try:
        # run() kills and waits for the child itself when it times out, so nothing is left behind
        proc = subprocess.run(
            [str(python_exe), "-c", snippet],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s,
            env={**os.environ, "PYTHONUTF8": "1"},                      # child writes UTF-8 too (zh-TW Windows defaults to cp950)
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        return _fail(f"自我檢查逾時（超過 {timeout_s:g} 秒），已結束檢查程序")
    except OSError as exc:
        return _fail(f"無法執行 {python_exe}：{exc}")

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-20:]).strip()
        return _fail(f"自我檢查程式結束碼 {proc.returncode}" + (f"：\n{tail}" if tail else ""))

    lines = proc.stdout.strip().splitlines()
    if not lines:
        return _fail("自我檢查沒有輸出")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return _fail(f"最後一行不是合法 JSON：{lines[-1][:200]}")

    cuda = bool(data.get("cuda_available"))
    deform = bool(data.get("deform_conv2d_ok"))
    device = data.get("device_name")
    device_name = device if isinstance(device, str) else None
    ok = cuda and deform
    if ok:
        message = f"自我檢查通過：{device_name}"
    else:
        message = str(data.get("message") or ("CUDA 不可用" if not cuda else "deform_conv2d 檢查未通過"))
    return SelfCheckResult(ok, cuda, device_name, deform, message)
