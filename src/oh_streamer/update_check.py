"""Manual "check for updates": asks the GitHub releases API, only when the user presses the button."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from oh_streamer.version import __version__

DEFAULT_URL = "https://api.github.com/repos/life-hack-lab/oh-streamer/releases/latest"
TIMEOUT = 10
NOTES_MAX = 2000
_VER = re.compile(r"^v?(\d+(?:\.\d+)*)(?:[-.]?([0-9A-Za-z.\-]+))?$")


def parse_version(text):
    """-> (numbers tuple, is_release, pre-release text) or None. '0.2.0-rc1' is a pre-release of 0.2.0."""
    m = _VER.match(str(text or "").strip())
    if not m:
        return None
    nums = tuple(int(x) for x in m.group(1).split("."))
    while len(nums) > 1 and nums[-1] == 0:
        nums = nums[:-1]
    return nums, not m.group(2), m.group(2) or ""


def is_newer(latest, current):
    a, b = parse_version(latest), parse_version(current)
    if a is None or b is None:
        return False
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[1] != b[1]:
        return a[1]            # final release beats its pre-release
    return not a[1] and a[2] > b[2]


def safe_release_url(url):
    return isinstance(url, str) and url.startswith("https://github.com/")


def _result(status, current, message, latest="", notes="", url=""):
    return {"status": status, "current": current, "latest": latest, "notes": notes, "url": url, "message": message}


def check(opener=None):
    current = __version__
    url = os.environ.get("OH_STREAMER_UPDATE_URL") or DEFAULT_URL
    req = urllib.request.Request(url, headers={
        "User-Agent": f"oh-streamer/{current}", "Accept": "application/vnd.github+json"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return _result("no_release", current, "目前還沒有公開的版本可以下載。")
        return _result("error", current, f"伺服器回應異常（{e.code}），請稍後再試。")
    except (urllib.error.URLError, TimeoutError, OSError):
        return _result("error", current, "無法連線，請確認網路後再試。")
    except ValueError:
        return _result("error", current, "收到的回應無法解讀，請稍後再試。")
    if not isinstance(data, dict):
        return _result("error", current, "收到的回應無法解讀，請稍後再試。")
    tag = str(data.get("tag_name") or "")
    if data.get("draft") or data.get("prerelease") or parse_version(tag) is None:
        return _result("no_release", current, "目前還沒有公開的版本可以下載。")
    latest = tag[1:] if tag[:1] in "vV" else tag
    if not is_newer(tag, current):
        return _result("latest", current, f"目前已是最新版本（{current}）。", latest=latest)
    notes = str(data.get("body") or "")
    if len(notes) > NOTES_MAX:
        notes = notes[:NOTES_MAX].rstrip() + "…"
    page = data.get("html_url") or ""
    return _result("update", current, f"有新版本 {latest} 可以下載。", latest=latest, notes=notes,
                   url=page if safe_release_url(page) else "")
