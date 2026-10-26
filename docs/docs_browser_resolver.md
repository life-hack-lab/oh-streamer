# Browser resolver

The browser resolver is the shared fallback for page URLs that do not have a matching custom rule or a usable native `yt-dlp` page extractor. It discovers an HLS playlist from a browser session, then returns the playlist URL and the request headers needed to read it.

## Resolution flow

`source_resolve.find_resolver()` reads app-data `resolvers.json` when present, otherwise the bundled defaults. Explicit matching rules take priority. The bundled wildcard rule enables a browser attempt only when the URL is not direct media and no native page extractor is available; it does not promise that the page is supported.

`resolve_browser` opens a browser context and inspects player state in each frame, HLS references in the page HTML, and successful playlist responses observed during navigation. A candidate must be an HTTP(S) URL and its response must contain an HLS playlist (`#EXTM3U`). A playlist seen only in network traffic is considered only when the same frame also exposes a blob-backed player, which helps reject unrelated streams such as ads. The resolver ranks direct player references above observed responses and HTML references. If the best evidence identifies multiple distinct streams, it reports an ambiguity instead of guessing.

The JSON result includes `url`, `headers`, `title`, `cover`, and `media_type: "hls"`. The title falls back to the page URL; cover metadata is best-effort and may be `null`. Returned request headers are limited to `User-Agent`, `Referer`, `Origin`, `Authorization`, and `Cookie`. Browser cookies are queried for the selected playlist URL only. The player and queue pass the playlist to `yt-dlp` to select a rendition; `http_transport` limits its compatibility handler to that media host.

## Command line and browser profile

From an installed environment, run:

```powershell
oh-streamer-resolve-browser "<video-page-url>" --timeout 45
```

From a source checkout, use:

```powershell
python -m oh_streamer.sources.resolvers.resolve_browser "<video-page-url>" --timeout 45
```

The bundled dispatcher also accepts `oh-streamer --resolver browser "<video-page-url>"`. On success, JSON is written to stdout; an error object is written to stderr. The timeout is measured in seconds and may be fractional. The default is 45 seconds; valid values satisfy `0 < timeout <= 180`.

The default browser context is temporary. The resolver tries the platform's installed Edge or Chrome before Playwright Chromium, and launches a visible browser window positioned off-screen. For a site that needs persistent browser state, pass `--profile-dir`:

```powershell
python -m oh_streamer.sources.resolvers.resolve_browser "<video-page-url>" --timeout 45 --profile-dir "<dedicated-profile-directory>"
```

Use a dedicated directory when persistence is needed. The profile can retain cookies and other browser state between runs. The resolver has no persistent result cache: it opens the page and resolves the playlist again on each invocation.

## Resolver overrides and opt-out

The app-data `resolvers.json` file is a JSON list. Rules are checked in order. `match` is a substring of the full URL; optional `host` limits the rule to that host and its subdomains. A matching custom resolver runs before the bundled browser fallback.

```json
[
  {
    "match": "pages.example.test/watch/",
    "host": "pages.example.test",
    "command": ["{python}", "resolvers/my_resolver.py"]
  }
]
```

Use an empty `command` to opt out of resolver fallback for a host or URL section:

```json
[
  {
    "match": "pages.example.test",
    "host": "pages.example.test",
    "command": []
  }
]
```

Remove the empty-command rule to restore the bundled fallback. Custom resolver commands receive the page URL as their final argument and must write a JSON result containing a playable `url` to stdout. `{python}` selects the app's current Python interpreter.

## Limits

- Browser discovery targets HLS playlists. It does not turn a page into a direct file URL or bypass a site's own access rules.
- The resolver does not click page controls. Pages that require a user action before exposing the player may fail.
- Browser challenges, unavailable videos, and player changes can prevent resolution. A page accepted by the wildcard rule is only a candidate for an attempt.
- Multiple equally strong stream candidates produce an ambiguity error. The resolver does not choose one based on URL order.
- Browser resolution requires Playwright and an available Edge, Chrome, or Playwright Chromium installation.
- Direct media URLs and URLs with a working native `yt-dlp` extractor do not use the wildcard browser fallback.
- The local metadata sample supports MPEG-TS playlists, including AES-128 segments through the existing HLS decryption helper. Unsupported playlist features such as fMP4 initialization maps remain explicit errors in this sample path.
- Only the listed playback headers are returned. Cookies are collected for the chosen playlist URL, and the compatibility HTTP handler is scoped to that URL's exact hostname. A persistent profile can retain authentication state on disk; do not point it at a profile containing unrelated browsing sessions.

## Targeted checks

Run from the repository root with the development Python environment. Automated resolver and routing tests use synthetic `example.test` hosts, mocked browser/page responses, and controlled local fixtures. They verify selection, ambiguity handling, header filtering, cookie scoping, routing, and transport behavior; they do not establish support for every source or browser platform.

```powershell
python -m unittest tests.test_resolve_browser tests.test_browser_platform tests.test_browser_routing tests.test_http_transport tests.test_http_headers tests.test_hls_http_scope tests.test_resolvers
python -m unittest tests.test_source_info tests.test_player_opening tests.test_queue_download
python -m unittest tests.test_source_cover tests.test_user_data_migration tests.test_packaging_entry tests.test_packaged_resources tests.test_package_imports tests.test_borrow_mac_bundle
```

The separate live pipeline smoke is manual and depends on an accessible page and a locally installed browser:

```powershell
oh-streamer-resolve-browser "<video-page-url>" --timeout 45
```

Inspect the returned fields locally. The result can contain short-lived media URLs, authorization headers, or cookies; avoid sharing its raw output.
