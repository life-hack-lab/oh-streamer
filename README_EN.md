# Oh Streamer

[**繁體中文**](README.md) | [**English**](README_EN.md)

---

Oh Streamer is a video player built on [Lada](https://codeberg.org/ladaapp/lada)'s mosaic restoration models. It plays a local file or a stream URL (m3u8 / mp4, optional HTTP headers) through restoration in near real time, with audio, in mpv, and can also process videos completely in a background queue. It includes seeking, a quality menu, an advanced-settings menu, an info panel, a persistent player window (opened from the main app), local AI subtitles, and casting to Apple TV (HLS + AirPlay).

> **Preview release**: this is the first public preview (0.1.0). It is usable but evolving quickly, and rough edges or breaking changes are likely. Tested on Windows 11 with an NVIDIA RTX GPU (NVENC); this is an independent project, not an official Lada component.

**Download**: get the Windows installer `OhStreamer-Setup-<version>.exe` from [GitHub Releases](https://github.com/life-hack-lab/oh-streamer/releases). An NVIDIA GPU is required (GTX 16 series or RTX 20-50 series, driver 570 or newer). The installer is not code-signed, so Windows SmartScreen may warn; see [Installation](#installation-windows).

Pair AirPlay directly in the player: open the cast button (or press `c`), select the device's pairing row, enter the PIN shown on Apple TV using the menu's digits, and confirm. Then select the device's cast row. The menu supports deleting a digit, cancelling, and pairing again. PIN entry times out after two minutes and can be retried; no terminal is required.

## Installation (Windows)

**Requirements**: 64-bit Windows 10/11 (currently tested only on Windows 11); an NVIDIA GTX 16 series or RTX 20-50 series GPU with driver 570 or newer (AMD and Intel GPUs come later); about 9 GB of free disk space and a stable connection for the first install.

1. Download `OhStreamer-Setup-<version>.exe` from [GitHub Releases](https://github.com/life-hack-lab/oh-streamer/releases) and run it. It installs for the current user only, without admin rights, to `%LOCALAPPDATA%\Programs\oh_streamer` by default (changeable in the installer). The installer is not code-signed; if Windows SmartScreen blocks it, click "More info -> Run anyway".
2. The first launch opens the setup screen: it detects the GPU, suggests a version, and downloads the runtime (about 4 GB, of which torch is about 3.2 GB). torch is a single large file and restarts from zero if the download is interrupted; other downloaded packages are kept.
3. The main app opens when setup is done. For local AI subtitles, download the local subtitle models under Settings -> Subtitles -> Local (about 12.8 GB by default; cancel and resume are supported). Cloud subtitles need no download. For model sources and licenses see [Subtitle models and licenses](#subtitle-models-and-licenses).

- **Repair and diagnostics**: the Start-menu shortcut "Oh Streamer 修復與診斷" reopens the setup screen to reinstall, switch versions, or export a diagnostics bundle (attach it to bug reports).
- **Updates**: "Check for updates" in Settings queries GitHub Releases and opens the download page when a newer version exists. Run the new installer over the old one; settings and the queue are kept, and torch is not downloaded again if the runtime did not change.
- **Uninstall**: from Settings -> Apps -> Installed apps. It asks separately whether to delete the runtime and download cache (about 5 GB), downloaded subtitle models, and user data (queue and settings); all are kept by default so a reinstall does not download them again.
- User data lives in `%LOCALAPPDATA%\oh_streamer`, separate from the program folder.

## macOS (experimental)

The macOS version is **experimental and unverified**: MPS/VideoToolbox paths and platform services have been added and are covered by Windows regression tests, but no Mac installer is published and it has not been fully validated on real Apple Silicon hardware. Help from Mac users with testing and reports is very welcome. See [Mac source setup](docs/docs_macos_setup.md) and the [macOS packaging notes](packaging/README_macos.md).

## Usage

The installed app opens from the Start menu as "Oh Streamer". When developing from source, the main entry point is `Oh Streamer.bat` in the repository root on Windows and `Oh Streamer.command` on macOS.

The main window is unified: paste a URL, cURL command, or local file path, then choose **串流播放** (stream playback) to open it in the player or **加入佇列** (add to queue; Enter also works) for background processing. Shift+Enter inserts a newline. Stream playback does not first download and save a complete video, though startup may wait for buffering and AI restoration to initialize. Player playback settings and background queue processing settings are separate.

### Subtitles (subtitle icon or `u`, applied live)

The subtitle button sits to the left of the settings gear. Choose a subtitle track, show or hide subtitles, set their scale (50%–200%), or shift timing by 0.1 seconds at a time and reset the offset. Selecting a track also restores subtitle visibility. The menu explains when no subtitle tracks are available. Use arrow keys to navigate, Enter to select, and Esc to go back or close. These controls apply to the current local player session and are not saved as global preferences. Use the receiving device's subtitle controls when casting. AI subtitle generation is configured in the processing queue.

Before casting, select a subtitle track and make it visible. AirPlay/HLS includes the selected external SRT, ASS, WebVTT, or embedded text track with the current timing offset. Dual subtitle mode casts the primary track. Select “字幕” in the receiver's subtitle menu if needed. Stop and cast again to change the track or timing offset. The receiver controls font and size; ASS effects and image subtitles are unsupported. Conversion failures display a message while video casting continues.

### Display options (settings menu `s`, applied live)

- **RTX Super Resolution** (default on; `--vsr on|off`): NVIDIA RTX Video Super Resolution upscales the video to the
  display resolution (mpv `d3d11vpp` filter, up to 4x). Needs an RTX GPU, mpv decoding with d3d11va, and
  "RTX Video Enhancement → Super Resolution" enabled in the NVIDIA App. The info panel (`i`) shows its state.
- **Smooth playback** (default on; `--smooth-playback on|off`): paces playback to the display refresh rate (mpv
  `video-sync=display-resample` + `interpolation`, `tscale=oversample`) to reduce judder; this is not frame interpolation.

Both only affect local mpv playback, not what is cast to Apple TV.

Casting has its own separate **Cast RTX Super Resolution** switch (press `c` to open the cast menu; default off). It runs NVIDIA VSR on the computer and lets you choose "2x (up to 4K)" or "Up to 4K" (e.g. 720p to 4K) before sending to Apple TV or an HLS receiver. It needs a Windows RTX GPU and the optional NVEncC, and can only be changed while casting is stopped. For installation and limits see the [cast Super Resolution notes](docs/docs_cast_vsr.md).

### Restoration frame rate (settings menu `s`)

**Restoration frame rate** (default original; `--restoration-fps original|30`): "Up to 30 fps" samples frames at even intervals before Lada detection and restoration, which reduces the compute needed. 60 becomes 30 and 59.94 becomes 29.97 fps; sources at 30 fps or below are left as they are, and higher rates are sampled at an integer interval to at most 30 fps. Combined with "Interpolation: auto", a 60 fps source can be restored at 30 fps and brought back to 60 fps; when the GPU has no headroom the repeated-frame fallback is used. Audio and video timing do not change, a change restarts from the current position, and casting must be stopped first. The info panel shows the source and restoration frame rates. This setting applies to live restoration playback; background queue processing keeps its own settings.

### Frame interpolation (RIFE; settings menu `s`, restarts at the current position)

**插幀（RIFE）** (default off; `--interpolation off|auto`): "auto" outputs at 2x the source frame rate (e.g. 30 → 60
fps). Between two restored frames it inserts a RIFE v4.6 middle frame (`third_party/rife`, fp16 + CUDA graph; flow at
half resolution above 720p) while processing is comfortably ahead of playback, and repeats the previous frame
otherwise: when the GPU cannot keep up you just get the normal frame rate for a while, never a stall caused by it.

- Signal: seconds of output ahead of mpv's playback position (capped by what mpv has actually cached), minus the
  restoration time of the mosaic clips already detected ahead; interpolation starts at >= 6 s and stops below 3 s.
  Nothing is interpolated before a start / seek is playing, so seeks are not slower.
- Paused while one RIFE frame (median of the last 15) takes more than 90 % of a source frame interval (e.g. a 4K
  source, or another app on the GPU); retried after 15 s.
- Scene cuts (SSIM of 32x32 thumbnails < 0.2) repeat the previous frame instead of blending two shots.
- Repeated frames are dropped before encoding (variable frame rate), so while it is not interpolating the encoder and
  mpv (decoding, RTX VSR) do the same work as with "off"; interpolated stretches cost them twice the frames.
- Restoration frame rates above 40 fps (e.g. restoring a 60 fps video at its original rate) or an unknown frame rate keep the original frame rate; casting to Apple TV does not interpolate and stays at the selected restoration frame rate.
- The info panel (`i`) shows e.g. 「插幀：自動 62%（近 10 秒補幀比例）」 (share of interpolated pairs over the last
  10 s) and the current RIFE time per frame.
- The RIFE model takes about 0.3–0.7 GB of VRAM and is loaded in the background the first time a resolution plays.
  With `--output <file>` a simulated viewer decides when to interpolate.

### Compare with the original (`compare`, default off; settings menu, restarts at the current position)

When on, the stream carries an extra track with the unrestored original frames. Keys: `o` hold to show the original,
`Shift+o` step back one stage, `[` `]` change the stage shown, `w` split-screen wipe (original on the left, processed on the right; drag with the left mouse button or use `Ctrl+Left` / `Ctrl+Right` to move the divider, `w` or `Esc` leaves). The wipe needs software-copy decoding, so entering it temporarily turns RTX VSR off (with an on-screen note) and stalls about 0.25 s; leaving restores VSR, smooth playback and the track you had. It needs at least two pipeline tracks (`compare` on) and exits by itself when the stream is reloaded or restarted. It costs one more NVENC
encode and VRAM.

### Background processing queue (`a`)

The background queue lives in the unified window opened by `Oh Streamer.bat` (needs `pywebview` and `pystray`; closing the window minimizes it to the system tray where processing continues, and "Exit" in the tray menu really quits). Add a URL / cURL command / local file with "加入佇列" or Enter, or use "加入檔案" or drag and drop. Supported HLS VOD sources are processed while they download; other URLs download the original first. One video at a time is fully restored in the background (a segment is saved about every 4 seconds, and work resumes from the interrupted segment after a pause or shutdown); when done the segments are joined into `output.mkv` and the original is deleted. Queue settings: cache limit (when full, watched, unkept, least recently used outputs are deleted first, and if that is not enough the queue pauses and says so), encoder (default HEVC), interpolation (default "每格補幀", which outputs 2x the frame rate for sources of 40 fps or less), keep comparison picture (default off), prevent sleep while processing, and shut down when all are done. Each video can be kept (lock icon), exported as mp4, or have its login information cleared. Cache location: `%LOCALAPPDATA%\oh_streamer\queue`.

Pressing `a` during playback (or the last row of the settings menu `s`, "加入處理佇列") adds the current video with its current quality and restoration settings to the queue. The queue app must be open first, otherwise "處理佇列未開啟" is shown; on success "已加入處理佇列" is shown.

- While the player restores live it tells the queue to pause (reported every 10 seconds; the queue resumes after restoration stops or the player closes), so they do not compete for the GPU.
- When you open a URL or file the player asks the queue first: an already processed video plays the finished output ("播放已處理的版本") instead of restoring live.
- **Processed mode** (the queue "播放" button, or the developer-only command line `oh_player.py --player <file> --processed`): the output is an ordinary video file that does not go through Lada, so seeking, caching, RTX VSR and smooth playback behave as in normal playback; the quality menu and the settings that would need re-restoration (model, restoration window, interpolation and so on) only show "已處理的影片"; the output can be cast from the cast menu. When the output has `oh:*` comparison tracks, `o` `[` `]` `w` work as usual, otherwise "對照未啟用" is shown. Reaching 90% or the end tells the queue it has been watched.
- **Watch while processing** (press "播放" on a queue item that is not finished): processed segments play directly (no re-encoding, same quality as the output) while background processing continues; the green range on the seekbar is the processed part. Playing or seeking to an unprocessed spot makes the background job jump there first, with "處理中" shown for a few seconds; skipped parts are filled in later and the output is only merged once everything is done. With interpolation set to "每格補幀", if background processing is about to be caught by playback (less than 10 seconds ahead) it temporarily switches to automatic interpolation to keep up, and returns when the lead is back to 20 seconds; those segments are redone with full interpolation later, so the output is still full quality. Pressing "播放" on an item that is still downloading remembers the request and attaches to the same background result once the source is readable, without opening the original URL again. Supported HLS sources do not have to finish downloading, and download and processing progress are shown separately; for conditions and recovery see [shared queue sources](docs/docs_queue_overlap.md).

### Casting finished and in-progress queue videos

Finished queue videos can also be cast from the player menu. Casting uses the processed track without running AI restoration again. Stopping casting resumes locally at the receiver position.

Progressive queue playback can also be cast from the same menu. Casting reads the already processed queue segments and converts them to H.264/AAC without rerunning AI restoration. Missing segments are requested from the queue and waited for; Apple TV position and pause state are reported back to the worker. Segment files and source audio remain available while a viewer is active, including after processing completes. Stopping casting resumes local playback at the receiver position.

During AirPlay casting, use the local seekbar or arrow keys to seek directly, and the play button or Space to pause/resume. Commands use the MRP tunnel over AirPlay and keep the existing casting connection. Position and pause state follow receiver updates. Seeking to an unprocessed segment may require buffering. Unsupported or failed remote commands display an error without restarting casting. For HLS-only streaming, control playback on the receiver.

## Optional external resolvers

For page URLs that yt-dlp cannot handle you can plug in your own script. The player first loads `resolvers.json`
from the platform user data directory, then falls back to the packaged `resolvers.default.json`. Use
`resolvers.example.json` as a starting point for custom settings. The player matches page URL substrings against
`match` and runs the corresponding `command`. Setting `"python"` as the first command argument uses the current
Python environment. Relative script paths resolve beside the active resolver settings file; absolute paths are kept
as configured. The command must print JSON `{"url": ..., "headers": {...}, "title": ...}` to stdout.

Video page URLs can be pasted directly into the player or queue. Existing custom resolver matches take priority; the bundled browser resolver is used when none matches. See the [browser resolver notes](docs/docs_browser_resolver.md) for behavior and limitations.

A generic browser resolver (package module `oh_streamer.sources.resolvers.resolve_browser`, usable from source, wheel and frozen launches; matching all page URLs via `resolvers.default.json`) is pre-configured. It opens the page in a system browser and finds the HLS playlist from player state and network responses. To use it, simply install playwright:
```bash
uv pip install playwright     # or: pip install playwright
```
*(No need to run `playwright install chromium` — it automatically connects to your Windows system's Microsoft Edge / Chrome).*

## Short clips for speed testing

Double-click `Make Clip.bat` and enter the video link, start time and clip length in turn, or run this from the repository root:

```powershell
..\lada\.venv\Scripts\python.exe make_clip.py "video link" --start 00:05:00 --duration 30 --quality 720p -o clips/test_720p.mp4
```

It supports the built-in browser resolver, other video pages yt-dlp can resolve, direct video / m3u8 URLs and local files. Times accept seconds, `MM:SS` or `HH:MM:SS`, with decimals. Omitted arguments are asked interactively; the default is the highest quality, and output goes to `clips/`. The clip is cut exactly with H.264 CRF 18 / AAC, keeping resolution and frame rate, and Lada restoration is not run. ffmpeg / ffprobe must be on PATH. Existing output is not overwritten unless you add `--overwrite`.

A `.json` file of the same name records the source, cut time, actual duration, size / frame rate and preparation time, for comparing speed; the preparation time includes page resolution and cutting, and is not Lada processing speed. If the clip runs past the end the actual duration is shown; if it is entirely past the end it fails and leaves no empty MP4.

## User data

Settings, queue data, logs, AirPlay pairing credentials, and the `resolvers.json` override live under the platform
user data directory: `%LOCALAPPDATA%\oh_streamer` on Windows, `~/Library/Application Support/oh_streamer` on
macOS, and `${XDG_DATA_HOME:-~/.local/share}/oh_streamer` on Linux. Logs are in its `logs/` subdirectory. An explicit `--settings-file` path is used as given.

The background queue can process supported HLS VOD sources while they download. The initial implementation
supports complete, unencrypted MPEG-TS media playlists with one combined video/audio rendition. Downloading,
processing, progressive playback, and final audio remux share the same local source cache. Other source types
finish downloading before processing starts. Pressing Play on an unready queue item remembers the request and
opens its processed segments when the source becomes readable. See [shared queue sources](docs/docs_queue_overlap.md)
for lifecycle, recovery, and current source limitations.

## Subtitle models and licenses

The models used for local AI subtitles (Japanese to Traditional Chinese) are **downloaded on demand by the user inside the app and are not distributed by this project**. Please note:

- The Sakura translation models are licensed CC BY-NC-SA 4.0 (non-commercial use only).
- The default speech recognition (ASR) model's model card declares no license.
- Check that the licenses of the models you download fit your use.

Sources and licenses of components and models are collected in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development

> This section is maintainer-oriented. The current development setup expects sibling checkouts: an upstream Lada checkout with its `.venv` and model weights at `../lada`, and RIFE and other third-party resources at `../third_party`, next to this repository. This is not yet a one-command reproducible setup, so external contributors may not be able to fully reproduce it; a reproducible uv-based dev environment is planned. Until then, anyone can run the offline CI tests with only the standard library; see [CONTRIBUTING.md](CONTRIBUTING.md). Naming rules are in [AGENTS.md](AGENTS.md).

The application package lives in `src/oh_streamer/`, grouped into player, queue, processing, sources, and platform modules. Root scripts such as `oh_player.py` and `queue_app.py` are compatibility launch shims that add `src/` automatically, so no `pip install -e .` is needed for development or launch. Documentation, tests, and benchmarks live in `docs/`, `tests/`, and `benchmarks/`. Lua, the queue Web UI, and resolver default/example JSON files live in `src/oh_streamer/resources/`.

External dependencies for development: ffmpeg/ffprobe on PATH (NVENC-capable build), mpv (`tools\mpv\mpv.exe` or on PATH), yt-dlp; `pyatv` for casting; `nvidia-ml-py` for the info panel.

```powershell
# Direct command-line playback (development/diagnostics only):
..\lada\.venv\Scripts\python.exe oh_player.py <file-or-url> [options]
```

`Oh Player.bat` is a developer-only direct player launcher; end users open videos from Oh Streamer.

### Running tests

Run from the repository root (on macOS use `../lada/.venv/bin/python run_tests.py`):

```powershell
..\lada\.venv\Scripts\python.exe run_tests.py
```

The runner locates the repository and discovers `tests/` automatically. CI runs a set of offline tests with only the standard library (see `.github/workflows/` for the list), for example:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_update_check tests.test_build_provenance
```

Subtitle tests need a separate subtitle environment with OpenCC (`opencc-python-reimplemented`, declared in the `subtitles` extra of `pyproject.toml`):

```powershell
$env:PYTHONPATH = "$PWD;$PWD/src"
python -m unittest discover -s tests -p "test_subtitle*.py"
```

The existing `lada/.venv` lacks OpenCC, so running the full suite there produces two subtitle CLI errors with `OpenCC is required for s2twp conversion`. These indicate a missing test dependency, not a subtitle logic regression.

Build the installer with `python packaging/build_installer.py installer --out build-installer/out` (Inno Setup, about 6-8 minutes); for the release flow see the [release checklist](docs/docs_release_checklist.md).

## Disclaimers

- This project is not affiliated with, endorsed by, or sponsored by Apple, NVIDIA, Groq, or Microsoft. Trademarks belong to their respective owners.
- It builds on the upstream [Lada](https://codeberg.org/ladaapp/lada) project but is not an official Lada component and is not affiliated with it.
- DRM-protected content is not supported and will not be processed.
- Users are responsible for the content they process and for complying with the laws that apply to them.

## License

AGPL-3.0-only, Copyright (C) 2026 life-hack-lab. See [LICENSE](LICENSE); licenses of third-party components and models are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project builds on Lada, which is also AGPL-3.0.

## Links

- Reporting issues and security reports: [SECURITY.md](SECURITY.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Maintainer release flow: [docs/docs_release_checklist.md](docs/docs_release_checklist.md)
