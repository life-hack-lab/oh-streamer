"""Resolve a video page's HLS playlist from browser player state and requests."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import re
import sys
import time
from urllib.parse import urljoin, urlsplit

from .browser import launch_browser
from .covers import cover_url, fetch_cover


HLS_TYPES = {"application/vnd.apple.mpegurl", "application/x-mpegurl",
             "audio/mpegurl", "audio/x-mpegurl"}
PLAYER_SOURCES = r"""() => {
  const found = [];
  const add = (url) => { if (typeof url === 'string' && url) found.push(url); };
  try { add(window.hls && window.hls.url); } catch (_) {}
  try {
    const p = window.player;
    if (p) {
      if (typeof p.src === 'function') add(p.src()); else add(p.src);
      add(p.url);
    }
  } catch (_) {}
  try { add(window.art && window.art.url); } catch (_) {}
  for (const video of document.querySelectorAll('video')) {
    try { add(video._hls && video._hls.url); } catch (_) {}
    add(video.currentSrc); add(video.src);
    for (const source of video.querySelectorAll('source[src]')) add(source.src);
  }
  return found;
}"""
HTML_HLS = re.compile(r"(?P<url>(?:https?:)?//[^\s\"'<>\\]+?\.m3u8(?:\?[^\s\"'<>\\]*)?|/[^\s\"'<>\\]+?\.m3u8(?:\?[^\s\"'<>\\]*)?)", re.I)
HEADER_NAMES = {"user-agent": "User-Agent", "referer": "Referer", "origin": "Origin",
                "authorization": "Authorization", "cookie": "Cookie"}


def _http_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return parts.scheme in ("http", "https") and bool(parts.hostname)
    except (TypeError, ValueError):
        return False


def _is_hls(url: str, content_type: str = "") -> bool:
    return _http_url(url) and (urlsplit(url).path.lower().endswith(".m3u8") or
                               content_type.split(";", 1)[0].strip().lower() in HLS_TYPES)


def _is_playlist(body: str) -> bool:
    return isinstance(body, str) and body.lstrip("\ufeff \t\r\n").startswith("#EXTM3U")


def _headers(request_headers: dict, frame_url: str, user_agent: str) -> dict:
    result = {HEADER_NAMES[k.lower()]: v for k, v in request_headers.items()
              if k.lower() in HEADER_NAMES and isinstance(v, str)}
    result.setdefault("User-Agent", user_agent)
    if _http_url(frame_url):
        result.setdefault("Referer", frame_url)
    return result


@dataclass
class Candidate:
    url: str
    frame_url: str
    frame_id: int
    rank: int
    headers: dict = field(default_factory=dict)
    verified: bool = False
    observed: bool = False
    source_url: str | None = None


class Collector:
    def __init__(self, page, context, user_agent: str, deadline: float):
        self.page, self.context = page, context
        self.user_agent, self.deadline = user_agent, deadline
        self.candidates: list[Candidate] = []
        self.checked: set[tuple] = set()
        self.player_frames: set[int] = set()
        self.active_players: set[tuple[str, int]] = set()
        self.redirect_sources: dict[tuple[str, int], set[str]] = {}

    def add(self, url: str, frame_url: str, frame_id: int, rank: int,
            headers: dict | None = None, verified: bool = False, observed: bool = False):
        url = urljoin(frame_url, url) if _http_url(frame_url) else url
        if not _is_hls(url) and rank not in (2, 3):
            return
        if not _http_url(url):
            return
        if rank == 3 and not headers and any(
            c.rank == 3 and c.observed and (c.source_url or c.url) == url and c.frame_id == frame_id
            for c in self.candidates
        ):
            return
        headers = _headers(headers or {}, frame_url, self.user_agent)
        key = (url, frame_id, tuple(sorted(headers.items())))
        for candidate in self.candidates:
            if (candidate.url, candidate.frame_id, tuple(sorted(candidate.headers.items()))) == key:
                candidate.rank = max(candidate.rank, rank)
                candidate.verified |= verified
                candidate.observed |= observed
                return
        self.candidates.append(Candidate(url, frame_url, frame_id, rank, headers, verified, observed,
                                         url if rank == 3 else None))

    def observe(self, response):
        try:
            if not 200 <= response.status < 300:
                return
            request = response.request
            frame = request.frame
            if frame.page != self.page:
                return
            frame_url = frame.url
            if not _is_hls(response.url, response.headers.get("content-type", "")):
                return
            if not _is_playlist(response.text()):
                return
            frame_id = id(frame)
            prior = request.redirected_from
            for _ in range(8):
                if prior is None:
                    break
                if isinstance(prior.url, str) and _http_url(prior.url):
                    self.redirect_sources.setdefault((response.url, frame_id), set()).add(prior.url)
                prior = prior.redirected_from
            self.add(response.url, frame_url, frame_id, 2, request.all_headers(), True, True)
        except Exception:
            pass  # A frame may detach while a response is being read.

    def scan(self):
        self.player_frames = set()
        self.active_players = set()
        for frame in list(self.page.frames):
            try:
                frame_url = frame.url
            except Exception:
                continue
            try:
                for source in frame.evaluate(PLAYER_SOURCES) or []:
                    if isinstance(source, str):
                        if source.startswith("blob:"):
                            self.player_frames.add(id(frame))
                        else:
                            resolved = urljoin(frame_url, source) if _http_url(frame_url) else source
                            if _http_url(resolved):
                                self.active_players.add((resolved, id(frame)))
                            self.add(source, frame_url, id(frame), 3)
            except Exception:
                pass
            try:
                content = frame.content().replace("\\/", "/").replace("&amp;", "&")
                for match in HTML_HLS.finditer(content):
                    self.add(match.group("url"), frame_url, id(frame), 1)
            except Exception:
                pass
        self._associate_requests()

    def _associate_requests(self):
        observed = [c for c in self.candidates if c.observed]
        for player in [c for c in self.candidates if c.rank == 3 and not c.observed]:
            matches = [c for c in observed if c.frame_id == player.frame_id and
                       (c.url == player.url or player.url in self.redirect_sources.get(
                           (c.url, c.frame_id), set()))]
            if len(matches) == 1:
                player.url = matches[0].url
                player.headers = dict(matches[0].headers)
                player.verified = True
                player.observed = True

    def verify(self):
        for candidate in list(self.candidates):
            if candidate.verified or time.monotonic() >= self.deadline:
                continue
            key = (candidate.url, candidate.frame_id, tuple(sorted(candidate.headers.items())))
            if key in self.checked:
                continue
            self.checked.add(key)
            response = None
            try:
                remaining = max(1, int(min(4000, (self.deadline - time.monotonic()) * 1000)))
                # Do not carry player authorization or cookies across an HTTP
                # redirect to a different host. Browser traffic observes the
                # redirected request with its own real headers instead.
                response = self.context.request.get(candidate.url, headers=candidate.headers,
                                                    timeout=remaining, max_redirects=0)
                if response.ok and _is_playlist(response.text()):
                    candidate.url = response.url
                    candidate.verified = True
            except Exception:
                pass
            finally:
                if response is not None:
                    try:
                        response.dispose()
                    except Exception:
                        pass

    def choose(self) -> Candidate | None:
        # A bare network playlist can be an advertisement. A blob-backed
        # player supplies the missing relation when its HLS URL is hidden.
        valid = [c for c in self.candidates if c.verified and
                 (c.rank != 2 or c.frame_id in self.player_frames) and
                 (c.rank != 3 or (c.source_url or c.url, c.frame_id) in self.active_players)]
        if not valid:
            return None
        best_rank = max(c.rank for c in valid)
        best = [c for c in valid if c.rank == best_rank]
        # An explicit player in the top document is stronger evidence than a
        # separate embedded player, which may be an advertisement.
        if best_rank == 3:
            top_id = id(self.page.main_frame)
            top = [c for c in best if c.frame_id == top_id]
            if top:
                best = top
        identities = {(c.url, tuple(sorted(c.headers.items()))) for c in best}
        if len(identities) != 1:
            raise RuntimeError("Ambiguous HLS streams: multiple equally supported candidates")
        return best[0]


def _result(page, context, candidate: Candidate, fallback: str) -> dict:
    headers = dict(candidate.headers)
    try:
        cookies = context.cookies([candidate.url])
        if cookies:
            headers["Cookie"] = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
    except Exception:
        pass
    title, cover = fallback, None
    try:
        title = page.locator('meta[property="og:title"]').first.get_attribute("content", timeout=500) or page.title() or fallback
        if any(x in title.lower() for x in ("just a moment", "please wait", "请稍候")):
            title = fallback
    except Exception:
        pass
    try:
        cover = fetch_cover(context, cover_url(page.content(), page.url), page.url)
    except Exception:
        pass
    return {"url": candidate.url, "headers": headers, "title": title,
            "cover": cover, "media_type": "hls"}


def _resolve_page(page, context, page_url: str, user_agent: str, deadline: float) -> dict:
    collector = Collector(page, context, user_agent, deadline)
    page.on("response", collector.observe)
    try:
        try:
            page.goto(page_url, wait_until="domcontentloaded",
                      timeout=max(1, int(min(30000, (deadline - time.monotonic()) * 1000))))
        except Exception as exc:
            if "timeout" not in str(exc).lower():
                raise RuntimeError(f"Could not open video page: {exc}") from exc
        stable_signature = None
        stable_since = None
        while time.monotonic() < deadline:
            collector.scan()
            collector.verify()
            try:
                chosen = collector.choose()
            except RuntimeError:
                chosen = None  # A later player can resolve ambiguity.
            if chosen and chosen.rank == 3:
                signature = (chosen.url, chosen.frame_id, tuple(sorted(chosen.headers.items())))
                if signature != stable_signature:
                    stable_signature, stable_since = signature, time.monotonic()
                if time.monotonic() - stable_since >= 12:
                    return _result(page, context, chosen, page_url)
            else:
                stable_signature = stable_since = None
            page.wait_for_timeout(min(250, max(1, int((deadline - time.monotonic()) * 1000))))
        chosen = collector.choose()
        if chosen:
            return _result(page, context, chosen, page_url)
        raise RuntimeError("Failed to resolve HLS stream: timeout, unavailable video, or player changed")
    finally:
        page.remove_listener("response", collector.observe)


def resolve_browser(page_url: str, timeout: float = 45, *, profile_dir: str | None = None) -> dict:
    if not _http_url(page_url):
        raise ValueError("Expected an http(s) video page URL")
    if not 0 < timeout <= 180:
        raise ValueError("Timeout must be between 0 and 180 seconds")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed. Run: uv pip install playwright") from exc
    deadline = time.monotonic() + timeout
    with sync_playwright() as playwright:
        browser = launch_browser(playwright, profile_dir=profile_dir)
        try:
            context = browser if profile_dir else browser.new_context()
            page = context.pages[0] if profile_dir and context.pages else context.new_page()
            user_agent = page.evaluate("navigator.userAgent")
            return _resolve_page(page, context, page_url, user_agent, deadline)
        finally:
            browser.close()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Resolve a video page to an HLS playlist")
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--profile-dir", help="Explicit persistent browser profile for challenge compatibility")
    args = parser.parse_args()
    try:
        print(json.dumps(resolve_browser(args.url, args.timeout, profile_dir=args.profile_dir), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
