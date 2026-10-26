# Notes for coding agents (Codex, Claude Code, ...)

## Product name and branding

The official product display name is **Oh Streamer**. Its source of truth is:

`src/oh_streamer/resources/queue_ui/branding.js`

```js
window.BRAND = {"name": "Oh Streamer", "logo": "Oh!"};
```

- `name`: shown in window titles, tray menu, notifications, installer dialogs, and platform launchers.
- `logo`: text drawn by the pixel logo in the queue UI header (canvas width auto-fits the text).
- Keep the braces strict JSON (double quotes, no trailing commas): Python parses it too.

Rules:

- **Never hard-code the product name in a user-visible string.** Use `BRAND_NAME` / `BRAND_LOGO` in `queue_ui/app.js`, and `from oh_streamer.branding import APP_NAME` (also `LOGO_TEXT`) in Python. Tests should assert against `APP_NAME`, not a literal.
- **Technical identifier is `oh_streamer`; keep it stable.** This covers the Python package `oh_streamer`, the data folder `%LOCALAPPDATA%\oh_streamer` / `Application Support/oh_streamer` / `$XDG_DATA_HOME/oh_streamer`, the persisted files `oh_settings.json` and `oh_player*.log`, the `window.ohStreamer` JS bridge, the `ohStreamer.*` localStorage keys, the `oh-streamer*` commands, the player module `oh_streamer.player.oh_player` (root `oh_player.py`), the Lua scripts `oh_*.lua`, mpv `user-data/oh/*` and `oh-*` script messages, key bindings and script-opts, temp-file prefixes `oh_*`, the player mutex `Local\oh-player-<name>`, the mpv pipe `oh-mpv-<pid>`, the macOS names `oh-streamer` / `oh-player.app` / `local.oh.streamer[.player]`, and the subtitle combos manifest format `oh_streamer.subtitle_combos`. If the display name changes, edit only `branding.js` (and user-visible text sourced from it); never rename these identifiers, because that would alter existing installs and user data.
- **On-disk formats are final.** MKV tags `OH_*` (`OH_ORIGINAL`, `OH_COMPARE`, `OH_SOURCE_FPS`, `OH_RATE_MULT`, `OH_VIDEO_START`, `OH_ENGINE`, `OH_ENGINE_VERSION`) and track titles `oh:processed|original|restored` are written into output files and compared with `==` on read; never rename them, and keep the manifest `"format": "oh_streamer.subtitle_combos"` as is.
- **Technical environment variables use the `OH_STREAMER_` prefix** (for example `OH_STREAMER_SUBTITLE_*`, `OH_STREAMER_PACKAGING_*`, `OH_STREAMER_ENGINE`, `OH_STREAMER_PROJECT_ROOT`). Do not use `LADA_*` names for streamer's own variables. The only `LADA_*` names are `LADA_MODEL_WEIGHTS_DIR` (read by the upstream `lada` package) and `LADA_DIR`; neither may be renamed.
- **Do not use `lada` names for streamer's own identifiers** (modules, scripts, messages, files, env vars, tags, executables).
- "Lada" only names the upstream restoration engine: the `lada` package, `engines/lada/`, engine id `lada`, model weight names, `LADA_MODEL_WEIGHTS_DIR`, `LADA_DIR`, and the `../lada/.venv` checkout. Those are upstream identifiers; leave them unchanged.
- The sole user-facing app entry point is `Oh Streamer.bat` on Windows and `Oh Streamer.command` on macOS. `Oh Player.bat` and `Oh Player.command` are developer-only direct-player launchers; end users open videos from Oh Streamer.
- Windows installs use the Inno Setup installer built by `packaging/build_installer.py installer --out <dir>` (architecture and release flow: `docs/docs_multi_gpu_release_plan.md`). Developers still launch from a development environment with `Oh Streamer.bat`. Keep the `oh_streamer` technical identifiers (package, data folder, install folders) stable.
- Docs and logs use the current names.

For future branding changes, update `branding.js`, active user-facing documentation and launcher names together. Keep display strings sourced through the branding API and rebuild packaged applications after changing the branding source.
