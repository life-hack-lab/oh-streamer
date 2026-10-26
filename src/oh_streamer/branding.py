"""Product display name, read from the single source ``resources/queue_ui/branding.js``.

The file holds ``window.BRAND = {"name": ..., "logo": ...};`` (strict JSON object). Technical identifiers
(exe name, data dirs, package names) are not branding and never come from here.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

from oh_streamer.paths import resource_root

FALLBACK_NAME = "Streamer"
BRANDING_RELATIVE = Path("queue_ui") / "branding.js"
_OBJECT = re.compile(r"window\.BRAND\s*=\s*(\{.*?\})\s*;", re.DOTALL)


def parse_branding(text: str) -> dict:
    """Return {"name", "logo"} extracted from branding.js text; fallbacks for anything missing or invalid."""
    name = FALLBACK_NAME
    logo = None
    match = _OBJECT.search(text or "")
    if match:
        try:
            data = json.loads(match.group(1))
        except ValueError:
            data = None
        if isinstance(data, dict):
            raw_name = data.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
            raw_logo = data.get("logo")
            if isinstance(raw_logo, str) and raw_logo.strip():
                logo = raw_logo.strip()
    return {"name": name, "logo": logo or name[:2].upper()}


@lru_cache(maxsize=1)
def load_branding() -> dict:
    try:
        text = (resource_root() / BRANDING_RELATIVE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = ""
    return parse_branding(text)


APP_NAME: str = load_branding()["name"]
LOGO_TEXT: str = load_branding()["logo"]
