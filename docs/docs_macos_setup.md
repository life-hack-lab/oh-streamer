# macOS source setup (experimental)

This is an experimental source-launch path for the Oh Streamer player and queue on macOS. It has not yet been verified on a Mac. The upstream Lada macOS instructions install Lada's CLI and optional GTK application; this app uses its own mpv player and pywebview queue window, so the GTK/GStreamer GUI setup is not required here.

## Prepare Lada

The pinned model environment currently targets Apple Silicon. Intel macOS official PyTorch binaries stopped at 2.2.x; this upstream requires torch 2.9.1, so switching to CPU does not solve installation. See [the Intel testing limits](docs_portability_tools.md) before attempting setup.

For a separate borrowed-Mac checkout with pinned dependencies and a source bundle, follow [the borrowed-Mac guide](docs_borrow_mac.md). After setup, use [the preflight commands](docs_portability_tools.md) to record diagnostic and acceptance results.

Follow `docs/macOS_install.md` in the Lada repository through the CLI setup, including its Python environment, patches, and model weights. The launchers expect that environment at `../lada/.venv`, so check out Lada next to this repository (`lada/` and `oh-streamer/` side by side). Prefer native arm64 Python on Apple Silicon and follow Lada's own instructions for its PyTorch build. Do not replace or upgrade PyTorch independently to install Oh Oh Streamer packages.

## Install system tools and Oh Streamer packages

Install Homebrew, then the external tools used by the player:

```sh
brew install ffmpeg mpv
```

From the root of this repository, install the Oh Streamer dependencies into the Lada environment:

```sh
uv pip install --python ../lada/.venv/bin/python -r requirements-streamer.txt
# Optional features:
uv pip install --python ../lada/.venv/bin/python pyatv playwright
```

`psutil`, `yt-dlp`, `pywebview`, `pystray`, and Pillow support the player and queue. Cocoa bindings are installed on macOS; NVIDIA monitoring is installed only on Windows. `pyatv` is only needed for Apple TV casting. Playwright is only needed for the optional browser-based resolver. On macOS, the resolver tries installed Chrome, then Edge, then Playwright Chromium. If neither installed browser is available, install Playwright Chromium with `uv run --no-project --python ../lada/.venv/bin/python playwright install chromium`.

This requirements file omits Lada's model stack, including PyTorch; the upstream Lada environment manages those versions.

## Launch

Double-click `Oh Streamer.command` in Finder to open the main Oh Streamer window. `Oh Player.command` is a developer-only direct player launcher for diagnostics. You can pass queue options from Terminal:

```sh
chmod +x "Oh Streamer.command" "Oh Player.command"
# Developer-only direct player diagnostic:
"$PWD/Oh Player.command" "/path/to/video file.mp4"
"$PWD/Oh Streamer.command" --hidden
```

Application code lives in `src/oh_streamer/`. The root Python files are source-compatible command shims that add `src/` before loading the package, so setup does not require an editable install. The launch scripts call those shims by absolute path, locate the project relative to themselves, use `../lada/.venv/bin/python`, and add standard Homebrew paths for Finder launches. Open videos from the main Oh Streamer window; use the direct player launcher only for development diagnostics. Set `OH_STREAMER_PYTHON` to use a different interpreter. If macOS blocks a downloaded script, inspect it and use Finder's Open action or run it from Terminal according to your local security settings.

Lua, queue UI and resolver default/example files live in the package resource directory and are included by source, editable and wheel installs. A wheel still uses the separate upstream Lada model code and weights, and external FFmpeg/mpv tools. For a wheel install, use the main queue entry point `oh-streamer-queue` or `python -m oh_streamer queue`; the source launch scripts use the checkout's compatibility shims.

## Status and limits

These scripts provide source launch only; they do not create a standalone `.app`. A successful launch is not evidence that every Oh Streamer feature works on macOS. Cocoa window behavior, MPS inference, VideoToolbox encoding, queue lifecycle, casting, and performance still need real Mac testing. Treat failures as expected during this porting phase and report the Mac model, macOS/Python/torch/mpv/FFmpeg versions, and the relevant log when diagnosing them.

The initial Mac version disables RTX VSR, RIFE and automatic shutdown. `--device auto` selects CUDA, then MPS, then CPU; explicit `--device mps` or `--device cpu` is available per launch. An unavailable explicit device reports an error rather than silently falling back. Player settings, queue data, logs, AirPlay credentials and the resolver override live under `~/Library/Application Support/oh_streamer`; logs are in its `logs/` folder. An old checkout-level settings file, credentials file or `resolvers.json` is copied there on first use if no destination exists, and the original remains. Old checkout logs are kept where they are; new logs go under Application Support. `--settings-file` continues to use any explicitly selected path.
