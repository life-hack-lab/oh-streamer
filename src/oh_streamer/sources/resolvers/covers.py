"""Shared website cover metadata and browser-session image fetching."""
import base64
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


def image_data_uri(data: bytes) -> str | None:
    if not data or len(data) > 5 * 1024 * 1024:
        return None
    if data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        return None
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def fetch_http_cover(url: str, headers: dict) -> str | None:
    """Download extractor-provided covers with bounded size/time and supplied HTTP headers."""
    try:
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        with urlopen(Request(url, headers=headers), timeout=8) as response:
            return image_data_uri(response.read(5 * 1024 * 1024 + 1))
    except Exception:
        return None


def cover_url(content: str, page_url: str) -> str | None:
    """Read the page's own cover metadata; tolerate relative URLs and HTML entities."""
    candidates = []

    class Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "meta":
                value = attrs.get("content") or ""
                name = (attrs.get("property") or attrs.get("name") or "").lower()
                if name in ("og:image", "og:image:url", "twitter:image", "twitter:image:src"):
                    candidates.append((0 if name.startswith("og:") else 1, value))
                elif "preview.jpg" in value.lower():
                    candidates.append((3, value))
            elif tag == "video" and attrs.get("poster"):
                candidates.append((2, attrs["poster"]))

    Parser().feed(content)
    for _, value in sorted(candidates, key=lambda item: item[0]):
        if value.strip():
            try:
                url = urljoin(page_url, value.strip())
                if urlsplit(url).scheme in ("http", "https"):
                    return url
            except ValueError:
                continue
    return None


def fetch_cover(context, url: str | None, page_url: str) -> str | None:
    """Use the browser's cookies to fetch a cover; a failed image must not fail video resolution."""
    if not url:
        return None
    response = None
    try:
        response = context.request.get(url, headers={"Referer": page_url}, timeout=8000)
        if not response.ok:
            return None
        return image_data_uri(response.body())
    except Exception:
        return None
    finally:
        if response is not None:
            try:
                response.dispose()
            except Exception:
                pass


