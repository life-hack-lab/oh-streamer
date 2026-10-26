"""Source resolution helpers shared by the player (oh_player.py) and the queue (queue_download.py).

No torch / lada imports here: the queue window process must stay light. yt-dlp is imported lazily.
`log` is a hook: oh_player sets `source_resolve.log` to its own logger (stderr / log file).
"""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

from oh_streamer import user_data
from oh_streamer.paths import checkout_root, resource_root
from oh_streamer.platform import platform_support
from oh_streamer.sources import qualities
from oh_streamer.sources.http_headers import merge_for_media



def log(msg):
    print(f"[source_resolve] {msg}", file=sys.stderr, flush=True)


def parse_headers(header_args: list[str]) -> dict:
    headers = {}
    for h in header_args or []:
        if ":" not in h:
            raise SystemExit(f"Invalid --header '{h}', expected 'Name: value'")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def looks_like_media_url(url: str) -> bool:
    path = url.split("?", 1)[0].lower()
    # .m3u8 is deliberately excluded: a master playlist must go through yt-dlp to pick a single variant
    return path.endswith((".mp4", ".mkv", ".webm", ".ts", ".mov", ".flv"))


def check_url_support(url: str) -> dict:
    """Classify a URL using local resolver/extractor rules only; never contact the site."""
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname or any(c.isspace() for c in url):
            return {"status": "invalid", "message": "網址格式不正確"}
    except ValueError:
        return {"status": "invalid", "message": "網址格式不正確"}
    resolver = find_resolver(url, migrate=False)
    if resolver is not None and resolver.get("command") and resolver.get("builtin") != "browser":
        return {"status": "recognized", "message": "已辨識支援的網站規則；尚未驗證影片"}
    if looks_like_media_url(url) or parts.path.lower().endswith(".m3u8"):
        return {"status": "recognized", "message": "已辨識媒體網址；尚未驗證影片"}
    try:
        from yt_dlp.extractor import gen_extractor_classes
        for extractor in gen_extractor_classes():
            if extractor.IE_NAME != "generic" and extractor.suitable(url):
                return {"status": "recognized", "message": "已辨識支援的網站規則；尚未驗證影片"}
    except ImportError:
        pass
    return {"status": "unknown", "message": "無法預先判定此網址；仍可嘗試加入佇列"}


def pick_thumbnail_format(info: dict, min_height: int = 144) -> tuple[str, dict] | None:
    """Lowest-resolution video format with height >= min_height (the source for hover thumbnails).

    Returns (url, http_headers) or None. Only formats that carry video are considered (storyboards and audio-only
    formats are skipped); ties on height go to the lower bitrate.
    """
    best = None
    for f in info.get("formats") or []:
        url, height = f.get("url"), f.get("height")
        vcodec = f.get("vcodec")
        if not url or not height or height < min_height or vcodec in (None, "none"):
            continue
        if (f.get("ext") or "") == "mhtml" or str(f.get("format_note") or "").lower() == "storyboard":
            continue
        key = (height, f.get("tbr") or math.inf)
        if best is None or key < best[0]:
            best = (key, f)
    if best is None:
        return None
    f = best[1]
    return f["url"], dict(f.get("http_headers") or info.get("http_headers") or {})


def resolve_with_ytdlp(page_url: str, headers: dict, fake_formats: str | None = None, refresh: bool = False,
                       with_info: bool = False, *, _allow_browser: bool = True) -> tuple:
    """Returns (qualities, title, thumbnail source (url, headers) or None); with_info=True appends the yt-dlp info
    dict (duration, formats with their codecs / sizes) as a fourth element.

    qualities: playable video+audio formats, one per height, high -> low (see qualities.build_quality_list).
    fake_formats: path of an info-dict-like JSON (offline testing); refresh=True reads its "refresh_formats" key,
    which stands for what a second yt-dlp run returns (fresh signed URLs).
    """
    if fake_formats:
        info = qualities.load_fake_info(fake_formats, "refresh_formats" if refresh else "formats")
    else:
        import yt_dlp

        # Prefers one URL carrying both video and audio (Lada and ffmpeg read the same source); sites that only serve
        # separate streams (YouTube) get a video URL + an audio URL from qualities. The selector only has to succeed (it also fills in the top-level fields); the full info["formats"] list is
        # what the quality menu is built from.
        fmt = "best[vcodec!=none][acodec!=none]/bestvideo+bestaudio/best"
        opts = {"format": fmt, "quiet": True, "no_warnings": True, "noplaylist": True, "http_headers": headers}
        media_request = (not _allow_browser or urlsplit(page_url).path.lower().endswith(".m3u8")
                         or (any(k.lower() == "referer" for k in headers) and not _has_page_extractor(page_url)))
        if shutil.which("node"):   # YouTube needs a JS runtime (yt-dlp enables only deno by default; yt-dlp-ejs is in the venv)
            opts["js_runtimes"] = {"node": {}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                from oh_streamer.sources.http_transport import configure
                if media_request:
                    configure(ydl, page_url, headers)
                info = ydl.extract_info(page_url, download=False)
        except yt_dlp.utils.DownloadError:
            if (media_request or looks_like_media_url(page_url) or _browser_opted_out(page_url)):
                raise
            # A dedicated extractor may recognize a page but fail on its current
            # player. Capability discovery is still available without host rules.
            rule = {"builtin": "browser", "command": ["{python}", "-m",
                    "oh_streamer.sources.resolvers.resolve_browser"]}
            media, stream_headers, title, metadata = resolve_external(page_url, headers, rule, with_info=True)
            result = resolve_with_ytdlp(media, stream_headers, with_info=True, _allow_browser=False)
            quals, _title, thumb, info = result
            info = {**info, "title": title, "browser_resolved": True, "browser_media_url": media,
                    "cover": metadata.get("cover")}
            if with_info:
                return quals, title, thumb, info
            return quals, title, thumb
    quals = qualities.build_quality_list(info) or qualities.single_quality_from_info(info)
    if not quals:
        raise RuntimeError("yt-dlp did not return a playable URL with both video and audio")
    title = info.get("title") or page_url
    log(f"yt-dlp: {info.get('extractor') or 'fake formats'} | {len(quals)} qualit{'y' if len(quals) == 1 else 'ies'}: "
        f"{', '.join(q['label'] for q in quals)} | {title}")
    if with_info:
        return quals, title, pick_thumbnail_format(info), info
    return quals, title, pick_thumbnail_format(info)


RESOLVERS_FILE = platform_support.data_dir() / "resolvers.json"
LEGACY_RESOLVERS_FILE = checkout_root() / "resolvers.json"
DEFAULT_RESOLVERS_FILE = resource_root() / "resolvers.default.json"
_MIGRATION_LOCK = threading.Lock()
_MIGRATION_ATTEMPTED: set[str] = set()


def _valid_resolver_config(payload: bytes) -> bool:
    try:
        return isinstance(json.loads(payload.decode("utf-8")), list)
    except (UnicodeError, ValueError):
        return False


def _migrate_default_resolvers(path) -> None:
    if not user_data.same_path(path, RESOLVERS_FILE):
        return
    key = str(Path(path).resolve())
    with _MIGRATION_LOCK:
        if key in _MIGRATION_ATTEMPTED:
            return
        _MIGRATION_ATTEMPTED.add(key)
    _copied, warning = user_data.migrate_legacy_file(LEGACY_RESOLVERS_FILE, path, _valid_resolver_config)
    if warning:
        log(warning)


def _host_matches(cfg_host, hostname: str) -> bool:
    if not cfg_host:
        return True
    if isinstance(cfg_host, str):
        h = cfg_host.lower()
        return hostname == h or hostname.endswith("." + h)
    if isinstance(cfg_host, list):
        return any(isinstance(item, str) and (hostname == item.lower() or hostname.endswith("." + item.lower()))
                   for item in cfg_host)
    return False


def _has_page_extractor(url: str) -> bool:
    try:
        from yt_dlp.extractor import gen_extractor_classes
        return any(e.IE_NAME != "generic" and e.suitable(url) for e in gen_extractor_classes())
    except ImportError:
        return False


def _browser_opted_out(url: str) -> bool:
    """Honor external-rule opt-outs even after a native extractor fails."""
    try:
        entries = json.loads(RESOLVERS_FILE.read_text(encoding="utf-8"))
        hostname = (urlsplit(url).hostname or "").lower()
        return isinstance(entries, list) and any(
            isinstance(e, dict) and e.get("command") == []
            and isinstance(e.get("match"), str) and bool(e["match"])
            and (e["match"] == "*" or e["match"].lower() in url.lower())
            and _host_matches(e.get("host"), hostname) for e in entries)
    except (OSError, ValueError):
        return False


def find_resolver(s: str, resolvers_file: Path | None = None, default_file: Path | None = None,
                  *, migrate: bool = True) -> dict | None:
    """Optional external page resolvers from user-data/resolvers.json (or the bundled default file):
    [{"match": "<substring of page URL>", "command": ["python", "script.py"]}]. The command gets the page URL as its
    last argument and prints JSON {"url": ..., "headers": {...}, "title": ...} to stdout.
    resolvers_file / default_file override the module defaults (oh_player passes its own, test-patchable ones)."""
    if not is_url(s):
        return None
    primary = resolvers_file if resolvers_file is not None else RESOLVERS_FILE
    fallback = default_file if default_file is not None else DEFAULT_RESOLVERS_FILE
    if migrate:
        _migrate_default_resolvers(primary)
    cfg = primary if primary.exists() else fallback
    try:
        entries = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    low = s.lower()
    entries = entries if isinstance(entries, list) else []
    try:
        hostname = (urlsplit(s).hostname or "").lower()
    except ValueError:
        return None
    if not hostname or any(c.isspace() for c in s):
        return None
    for e in entries:
        if not (isinstance(e, dict) and isinstance(e.get("match"), str) and e["match"]
                and (e["match"] == "*" or e["match"].lower() in low)
                and _host_matches(e.get("host"), hostname)):
            continue
        # An explicit empty command disables the browser fallback for this rule.
        if e.get("command") == []:
            return None
        if e.get("match") == "*":
            continue
        if isinstance(e.get("command"), list) and e["command"]:
            command = e["command"]
            python_command = str(command[0]).lower() in ("python", "python3", "python.exe", "{python}")
            bundled_module = (len(command) >= 3 and command[1] == "-m" and isinstance(command[2], str)
                              and command[2].startswith("oh_streamer.sources.resolvers.resolve_"))
            missing_shim = (len(command) >= 2 and isinstance(command[1], str)
                            and command[1].replace("\\", "/").startswith("resolvers/resolve_")
                            and command[1].endswith(".py") and user_data.same_path(cfg, RESOLVERS_FILE)
                            and not (cfg.parent / command[1]).exists())
            if python_command and (bundled_module or missing_shim):
                return {**e, "builtin": "browser"}
            return e
    # The bundled wildcard is a capability fallback, never proof of support.
    # Keep native extractors and direct media on their existing paths.
    try:
        defaults = json.loads(fallback.read_text(encoding="utf-8")) if cfg != fallback else entries
    except (OSError, ValueError):
        defaults = []
    if isinstance(defaults, list) and not looks_like_media_url(s) and not urlsplit(s).path.lower().endswith(".m3u8"):
        for e in defaults:
            if (isinstance(e, dict) and e.get("match") == "*" and e.get("builtin") == "browser"
                    and isinstance(e.get("command"), list) and e["command"] and not _has_page_extractor(s)):
                return e
    return None


def resolve_external(page_url: str, headers: dict, resolver: dict, resolvers_file: Path | None = None,
                     *, with_info: bool = False) -> tuple:
    """Resolve a stream; with_info appends resolver metadata without changing existing callers."""
    log(f"resolving page URL via external resolver: {page_url}")
    config_path = Path(resolvers_file if resolvers_file is not None else RESOLVERS_FILE)
    base_dir = config_path.parent
    migrated_config = user_data.same_path(config_path, RESOLVERS_FILE)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    raw_cmd = [str(c) for c in resolver["command"]]
    cmd = []
    python_command = bool(raw_cmd and raw_cmd[0].lower() in ("python", "python.exe", "python3", "{python}"))
    module_prefix = "oh_streamer.sources.resolvers.resolve_"
    bundled = (python_command and len(raw_cmd) >= 3 and raw_cmd[1] == "-m"
               and raw_cmd[2].startswith(module_prefix))
    legacy_script = (python_command and migrated_config and len(raw_cmd) >= 2
                     and raw_cmd[1].replace("\\", "/").startswith("resolvers/resolve_")
                     and raw_cmd[1].endswith(".py") and not (base_dir / raw_cmd[1]).exists())
    if legacy_script:
        raw_cmd = [raw_cmd[0], "-m", module_prefix + "browser", *raw_cmd[2:]]
        bundled = True
    if bundled:
        # Migrate bundled commands by their reserved module namespace. Custom
        # scripts remain untouched; old built-ins now share the capability resolver.
        raw_cmd[2] = module_prefix + "browser"
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--resolver", "browser", *raw_cmd[3:]]
        else:
            cmd = [sys.executable, *raw_cmd[1:]]
            package_source = Path(__file__).resolve().parents[2]
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(package_source), existing_pythonpath) if value)
    else:
        for i, c in enumerate(raw_cmd):
            if i == 0 and c.lower() in ("python", "python.exe", "python3", "{python}"):
                cmd.append(sys.executable)
                continue
            rel = base_dir / c
            if not Path(c).is_absolute() and rel.exists():
                cmd.append(str(rel.resolve()))
                continue
            if migrated_config and not Path(c).is_absolute():
                legacy = checkout_root() / c
                if legacy.exists():
                    cmd.append(str(legacy.resolve()))
                    continue
            cmd.append(c)
    cmd.append(page_url)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    except OSError as e:
        raise RuntimeError(f"External resolver could not start ({cmd[0]}): {e}") from e
    if p.returncode != 0:
        raise RuntimeError(f"External resolver failed: {p.stderr.strip() or p.stdout.strip()}")
    try:
        data = json.loads(p.stdout.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from external resolver: {p.stdout}") from e
    if not isinstance(data, dict) or "url" not in data:
        err = data.get("error") if isinstance(data, dict) else p.stdout.strip()
        raise RuntimeError(f"External resolver did not return a valid stream URL: {err}")

    m3u8_url = data["url"]
    merged_headers = (merge_for_media(page_url, m3u8_url, data.get("headers", {}), headers) if bundled else
                      {**data.get("headers", {}), **headers})
    title = data.get("title") or page_url
    log(f"resolved: {title} | {m3u8_url}")
    if with_info:
        return m3u8_url, merged_headers, title, data
    return m3u8_url, merged_headers, title


class FFmpegHttpOptions:
    """HTTP options shared by ffprobe, PyAV (Lada's readers) and the ffmpeg muxer."""

    def __init__(self, headers: dict):
        headers = dict(headers)
        self.user_agent = None
        for k in list(headers):
            if k.lower() == "user-agent":
                self.user_agent = headers.pop(k)
        self.header_blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())

    def cli_args(self) -> list[str]:
        args = []
        if self.user_agent:
            args += ["-user_agent", self.user_agent]
        if self.header_blob:
            args += ["-headers", self.header_blob]
        return args

    def av_options(self) -> dict:
        opts = {"rw_timeout": "20000000"}  # 20s, microseconds
        if self.user_agent:
            opts["user_agent"] = self.user_agent
        if self.header_blob:
            opts["headers"] = self.header_blob
        return opts
