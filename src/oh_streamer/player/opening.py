"""Resolve, probe, and asynchronously open player inputs.

The facade supplies player-specific behavior through :class:`OpeningDependencies`. This keeps source opening
independent of the player module and makes the resolver and opener deterministic to exercise in isolation.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from oh_streamer.player import info_panel, open_input
from oh_streamer.sources import qualities
from oh_streamer.sources.http_headers import merge_for_media, scope_for_url


@dataclass(frozen=True)
class OpeningDependencies:
    """Functions owned by the player facade that the opening flow needs."""

    source_factory: Callable[..., Any]
    find_resolver: Callable[[str], dict | None]
    resolve_external: Callable[[str, dict, dict], tuple[str, dict, str]]
    resolve_with_ytdlp: Callable[..., tuple[list[dict], str, Any]]
    log: Callable[[str], None]
    is_url: Callable[[str], bool]
    looks_like_media_url: Callable[[str], bool]
    pick_quality: Callable[..., dict] = qualities.pick_quality
    quality_label: Callable[..., str] = qualities.quality_label
    source_label: Callable[[str], str] = info_panel.source_label


class Opened:
    """A resolved and probed input, ready to be played (what ``open_source`` returns)."""

    def __init__(self, **kw):
        self.input = self.quals = self.chosen = self.source = self.title = self.thumb_src = self.refresh = None
        self.user_headers, self.source_text, self.start = {}, "", 0.0
        self.overrides: dict = {}   # settings for this source only (player_settings.OVERRIDE_KEYS); set by the Opener
        self.stream_item: str | None = None   # the queue's subtitle-only stream item for this live source (open message)
        self.__dict__.update(kw)


def resolve_input(inp: str | None, user_headers: dict, args, quality_spec, fake_any: bool = True, *,
                  deps: OpeningDependencies) -> dict:
    """Resolve page URLs and select a starting quality without probing.

    ``fake_any`` lets the CLI replace any input with ``--fake-formats``. The player passes false so local files still
    open directly while URLs can use the fake formats for offline testing.
    """
    src, title = inp or "fake formats", inp or "fake formats"
    headers = dict(user_headers)
    thumb_src = None  # (url, headers) of a lighter rendition for hover thumbnails; None means the main source
    quals = chosen = refresh = None
    use_fake = bool(args.fake_formats) and (fake_any or deps.is_url(src))
    if use_fake or (deps.is_url(src) and deps.find_resolver(src) is None
                    and (args.ytdl or not deps.looks_like_media_url(src))):
        page_url = inp
        fake = args.fake_formats if use_fake else None
        quals, title, thumb_pick = deps.resolve_with_ytdlp(page_url, user_headers, fake)
        refresh = lambda: deps.resolve_with_ytdlp(page_url, user_headers, fake, refresh=True)[0]
        chosen = deps.pick_quality(quals, quality_spec, args.max_height)
        if thumb_pick:
            thumb_src = (thumb_pick[0], {**thumb_pick[1], **user_headers})
            deps.log(f"thumbnail source: lowest-resolution video format ({thumb_pick[0][:80]}...)")
        src, headers = chosen["url"], {**chosen["http_headers"], **user_headers}
    elif (resolver := deps.find_resolver(src)) is not None:
        page_url = src
        result = deps.resolve_external(page_url, headers, resolver)
        src, resolved_headers, title = result[:3]
        if resolver.get("builtin") == "browser":
            headers = merge_for_media(page_url, src, resolved_headers, user_headers)
            def expand(media_url, media_headers):
                formats, _, thumb_pick = deps.resolve_with_ytdlp(media_url, media_headers, None)
                formats = [{**fmt, "http_headers": merge_for_media(
                    page_url, fmt["url"], {**scope_for_url(media_url, fmt["url"], media_headers),
                                           **scope_for_url(media_url, fmt["url"], fmt.get("http_headers", {}))}, user_headers)}
                           for fmt in formats]
                return formats, thumb_pick

            quals, thumb_pick = expand(src, headers)
            chosen = deps.pick_quality(quals, quality_spec, args.max_height)
            if thumb_pick:
                thumb_src = (thumb_pick[0], merge_for_media(page_url, thumb_pick[0],
                    {**scope_for_url(src, thumb_pick[0], headers),
                     **scope_for_url(src, thumb_pick[0], thumb_pick[1])}, user_headers))
            src, headers = chosen["url"], chosen["http_headers"]

            def refresh():
                fresh = deps.resolve_external(page_url, dict(user_headers), resolver)
                media_url, fresh_headers = fresh[:2]
                return expand(media_url, merge_for_media(page_url, media_url, fresh_headers, user_headers))[0]
        else:
            headers = {**resolved_headers, **user_headers}
    elif not deps.is_url(src):
        src = str(Path(src).resolve())
        title = Path(src).name
    return {"input": inp, "src": src, "headers": headers, "title": title, "quals": quals, "chosen": chosen,
            "refresh": refresh, "thumb_src": thumb_src, "user_headers": dict(user_headers)}


def probe_input(r: dict, *, deps: OpeningDependencies) -> Opened:
    """Probe the selected rendition once and package its metadata for the player."""
    quals, chosen, src, headers = r["quals"], r["chosen"], r["src"], r["headers"]
    audio_src = chosen.get("audio_url") if chosen is not None else None
    if chosen is not None:
        deps.log(f"quality: {chosen['label']} (id {chosen['id']}) of {', '.join(q['label'] for q in quals)}")
    source = deps.source_factory(src, headers, audio_src)
    meta = source.meta
    deps.log(source.describe())
    if quals is None:
        quals = [{"id": "source", "height": meta.video_height, "width": meta.video_width,
                  "fps": float(meta.video_fps_exact), "tbr": None, "url": src, "http_headers": dict(headers),
                  "label": deps.quality_label(meta.video_height, float(meta.video_fps_exact))}]
        chosen = quals[0]
    elif not chosen.get("height") or not chosen.get("fps"):
        if not chosen.get("height") and meta.video_height:
            chosen["height"] = meta.video_height
            chosen["width"] = meta.video_width
        if not chosen.get("fps") and meta.video_fps_exact:
            chosen["fps"] = float(meta.video_fps_exact)
        if chosen.get("height"):
            chosen["label"] = deps.quality_label(chosen["height"], chosen.get("fps"))
    elif chosen["height"] and chosen["height"] != meta.video_height:
        deps.log(f"note: format reports {chosen['height']}p but the stream is {meta.video_height}p")
    inp = r["input"]
    return Opened(input=inp, quals=quals, chosen=chosen, source=source, title=r["title"],
                  thumb_src=r["thumb_src"] or (src, headers), refresh=r["refresh"], user_headers=r["user_headers"],
                  source_text=deps.source_label(inp if inp and deps.is_url(inp) else src))


def open_source(inp: str, user_headers: dict, args, quality_spec, fake_any: bool = True, *,
                deps: OpeningDependencies) -> Opened:
    """Resolve, choose a rendition, and probe everything needed before a Session can play ``inp``."""
    return probe_input(resolve_input(inp, user_headers, args, quality_spec, fake_any, deps=deps), deps=deps)


def short_error(e: Exception) -> str:
    """One short line for the OSD (details are in the log)."""
    if isinstance(e, open_input.OpenInputError):
        return str(e)
    first = (str(e).strip().splitlines() or [repr(e)])[0]
    first = first.removeprefix("ERROR: ").strip()
    if first.startswith("ffprobe failed"):
        return "無法開啟：無法讀取影片（檔案損毀或格式不支援）"
    return "無法開啟：" + (first[:70] + "…" if len(first) > 70 else first)


class Opener:
    """Resolve inputs on worker threads while the player loop continues serving mpv.

    A newer submit supersedes an in-flight resolution. yt-dlp cannot be interrupted, so an old result is dropped
    when it returns.
    """

    def __init__(self, args, quality_spec, *, open_source: Callable[..., Opened], log: Callable[[str], None]):
        self.args, self.quality_spec = args, quality_spec
        self._open_source, self._log = open_source, log
        self.lock = threading.Lock()
        self.gen = 0
        self.pending: int | None = None
        self.done = None  # (Opened | ProcessedOpen | None, Exception | None, seconds, label) for the latest request
        self.lookup = None  # oi -> ProcessedOpen | None: the queue already has it (run_player sets it)
        # stream_item -> None: called (on whatever thread) with the id of a stream item that no Opened will carry to
        # begin_source: the queue had a processed output, the open failed, or a newer request superseded it
        self.on_unused_stream_item: Callable[[str], None] | None = None

    @property
    def busy(self) -> bool:
        with self.lock:
            return self.pending is not None

    def submit(self, oi: open_input.OpenInput, label: str, start: float = 0.0, *, quality=None, overrides=None,
               stream_item=None):
        """quality: a quality spec (a height, 'best' / 'worst') for this request only; None = the opener's default.
        overrides: settings that apply to this source only; they come back on the Opened (``opened.overrides``).
        stream_item: id of the queue's stream item for this source; comes back as ``opened.stream_item`` (only a live
        source has it: a processed file the queue already has plays as is, no subtitle scheduling by the stream)."""
        with self.lock:
            self.gen += 1
            gen = self.pending = self.gen
            dropped, self.done = self.done, None
        self._unused(self._stream_item_of(dropped))
        threading.Thread(target=self._run, args=(gen, oi, label, start, quality, dict(overrides or {}), stream_item),
                         name="resolver", daemon=True).start()

    @staticmethod
    def _stream_item_of(done) -> str | None:
        """The stream item id a finished-but-not-picked-up result (see ``done``) would have carried."""
        if done is None or done[0] is None:
            return None
        return getattr(done[0], "stream_item", None) or None

    def _unused(self, stream_item) -> None:
        callback = self.on_unused_stream_item
        if not stream_item or callback is None:
            return
        try:
            callback(stream_item)
        except Exception as e:  # noqa: BLE001 - telling the queue is best effort
            self._log(f"unused stream item {stream_item}: {e!r}")

    def _run(self, gen, oi, label, start, quality=None, overrides=None, stream_item=None):
        spec = self.quality_spec if quality is None else quality
        t = time.perf_counter()
        opened, err = None, None
        try:
            if oi.kind == "file" and not os.path.isfile(oi.target):
                raise open_input.OpenInputError(f"找不到檔案：{Path(oi.target).name}")
            opened = None
            if self.lookup is not None:
                try:
                    opened = self.lookup(oi) if quality is None else self.lookup(oi, spec)
                except Exception as e:  # noqa: BLE001 - a broken lookup just means "play it live"
                    self._log(f"queue lookup failed ({e!r})")
            if opened is None:
                opened = self._open_source(oi.target, oi.headers, self.args, spec, fake_any=False)
                opened.overrides = dict(overrides or {})   # (a queue's processed file plays as is: nothing to override)
                opened.stream_item = stream_item or None
            opened.start = start
        except Exception as e:  # noqa: BLE001
            err = e
        with self.lock:
            superseded = gen != self.gen
            if not superseded:
                self.pending = None
                self.done = (opened, err, time.perf_counter() - t, label)
        if superseded or opened is None or getattr(opened, "stream_item", None) != stream_item:
            self._unused(stream_item)            # superseded / failed / the queue's own output plays instead: nobody uses it
        if superseded:
            self._log(f"resolution of {label} superseded by a newer request (dropped)")

    def cancel(self):
        """Drop a resolution in flight and a finished one not picked up yet."""
        with self.lock:
            self.gen += 1
            dropped, self.pending, self.done = self.done, None, None
        self._unused(self._stream_item_of(dropped))

    def poll(self):
        with self.lock:
            done, self.done = self.done, None
        return done
