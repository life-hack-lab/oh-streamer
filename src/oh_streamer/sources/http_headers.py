"""Case-insensitive HTTP header merging with origin-bound credentials."""

from urllib.parse import urlsplit


_SENSITIVE = {"cookie", "authorization", "proxy-authorization"}


def _origin(url: str) -> tuple | None:
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            return None
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
        return parts.scheme.lower(), parts.hostname.lower(), port
    except ValueError:
        return None


def _put(result: dict, key: str, value: str) -> None:
    for old in list(result):
        if old.lower() == key.lower():
            del result[old]
    result[key] = value


def merge_for_media(page_url: str, media_url: str, media_headers: dict, user_headers: dict) -> dict:
    """Merge browser media headers and user headers without moving page credentials.

    Browser-observed media credentials belong to the media URL. User credentials
    belong to the page origin and only override when the origins match exactly.
    """
    result = {}
    for key, value in media_headers.items():
        _put(result, key, value)
    same_origin = _origin(page_url) is not None and _origin(page_url) == _origin(media_url)
    for key, value in user_headers.items():
        if key.lower() not in _SENSITIVE or same_origin:
            _put(result, key, value)
    return result


def scope_for_url(source_url: str, target_url: str, headers: dict) -> dict:
    """Keep non-sensitive headers, removing credentials on cross-origin requests."""
    same_origin = _origin(source_url) is not None and _origin(source_url) == _origin(target_url)
    return {key: value for key, value in headers.items() if same_origin or key.lower() not in _SENSITIVE}
