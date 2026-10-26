"""A hostname-scoped standard HTTP adapter for yt-dlp media requests.

Some media servers accept requests' ordinary TLS transport but reject yt-dlp's
custom adapter.  yt-dlp still owns request construction and playlist parsing.
"""

from urllib.parse import urljoin, urlsplit


def configure(ydl, url: str, headers: dict) -> None:
    """Prefer the ordinary requests adapter for one media URL's exact host.

    Call once per YoutubeDL instance with the media URL. Repeated calls for the
    same host are harmless; a different host needs a separate YoutubeDL instance.
    ``headers`` remains owned by yt-dlp and is accepted for call-site symmetry.
    """
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return

    director = ydl._request_director
    name = "ScopedHTTPCompatibility"
    if name in director.handlers:
        return

    import requests
    import urllib3
    from yt_dlp.networking._requests import RequestsRH, RequestsSession
    from yt_dlp.networking.exceptions import UnsupportedRequest

    class ScopedSession(RequestsSession):
        def get_redirect_target(self, response):
            target = super().get_redirect_target(response)
            if target and (urlsplit(urljoin(response.url, target)).hostname or "").lower() != host:
                return None
            return target

    class ScopedHTTPCompatibilityRH(RequestsRH):
        RH_NAME = name

        def _validate(self, request):
            if (urlsplit(request.url).hostname or "").lower() != host:
                raise UnsupportedRequest("HTTP compatibility is limited to its configured hostname")
            super()._validate(request)

        def _create_instance(self, cookiejar, legacy_ssl_support=None):
            session = ScopedSession()
            adapter = requests.adapters.HTTPAdapter(max_retries=urllib3.util.retry.Retry(False))
            session.adapters.clear()
            session.headers = requests.structures.CaseInsensitiveDict()
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.cookies = cookiejar
            session.verify = True
            session.trust_env = False  # yt-dlp passes configured proxies explicitly.
            return session

    handler = next(iter(ydl.build_request_director([ScopedHTTPCompatibilityRH]).handlers.values()))
    director.add_handler(handler)
    director.preferences.add(lambda rh, request: 1000 if isinstance(rh, ScopedHTTPCompatibilityRH) else 0)
