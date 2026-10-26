# Experimental macOS streamer packaging

This is a source-first build skeleton, not a verified distributable. Use the existing source setup,
doctor and acceptance tools before packaging. It does not use upstream Lada's GTK GUI spec.

On a Mac, prepare the compatible upstream checkout and its Python environment as described in
`../docs/docs_macos_setup.md`, install PyInstaller separately into that environment, and run from the repository root:

```sh
../lada/.venv/bin/python packaging/build_macos.py \
  --lada-source ../lada \
  --mpv /absolute/path/to/prepared/mpv \
  --mpv-notices /absolute/path/to/prepared/ThirdPartyNotices
```

For an mpv executable whose linked dependencies include Vulkan, add the matching MoltenVK ICD
manifest:

```sh
../lada/.venv/bin/python packaging/build_macos.py \
  --lada-source ../lada \
  --mpv /absolute/path/to/prepared/mpv \
  --mpv-notices /absolute/path/to/prepared/ThirdPartyNotices \
  --mpv-vulkan-icd /absolute/path/to/MoltenVK_icd.json
```

The required `--mpv` path names a prepared macOS mpv executable; it is copied into
`Oh Streamer.app/Contents/Helpers/oh-player.app/Contents/MacOS/mpv` with its supported linked
Mach-O library dependencies. `--mpv-notices` must name a nonempty directory of notices and source
offers for that exact mpv build; its files are copied into
`Oh Streamer.app/Contents/Helpers/oh-player.app/Contents/Resources/ThirdPartyNotices/`. The build
does not produce an installer, install packages,
download files, copy model weights, or modify a system mpv installation. It rejects missing
libraries, incompatible architectures, and external framework dependencies. `--mpv-vulkan-icd` is
optional and required only when the mpv dependency closure includes `libvulkan`; it must point to the
matching `MoltenVK_icd.json`. The build copies the manifest-referenced MoltenVK driver and its linked
dependencies, then rewrites the manifest to
`Oh Streamer.app/Contents/Helpers/oh-player.app/Contents/Resources/vulkan/icd.d/MoltenVK_icd.json`.
Native arm64 and x86_64 builds need matching Python and dependency
architectures; this spec does not create a universal binary. The build ad hoc-signs the player and
outer app, but does not use Developer ID signing, hardened runtime, or notarization. Both the build
entry and spec reject non-macOS hosts. Build output defaults to `dist-macos/Oh Streamer.app` (bundle
name follows `branding.js`) and scratch files to `build-macos/`.

The frozen executable opens the main queue when launched without arguments, including when the user
double-clicks `Oh Streamer.app`. Explicit player CLI arguments are for development and internal use.
The executable also accepts `--queue`, `--worker --job <job.json>`, and `--resolver browser <url>` for
packaged application routes. It routes existing child-process arguments ending in `oh_player.py`,
`queue_app.py` or `oh_streamer.queue.queue_worker` (`src/oh_streamer/queue/queue_worker.py`), plus `resolvers/resolve_browser.py`, to the canonical
`oh_streamer.*` modules. This is necessary because frozen `sys.executable` is the app, not a
general Python interpreter. Other `.py` scripts fail clearly. For custom resolver scripts, configure
an **absolute external Python interpreter** in the resolver command; `python`, `python3`, and
`{python}` are replaced by the current executable by `source_resolve`. The bundled generic browser resolver's
browser/dependency availability is still a Mac validation item; no browser is included or downloaded
by this build.

Application code is packaged under canonical `oh_streamer.*` module names. Lua files,
`queue_ui/`, `resolvers.default.json`, and `resolvers.example.json` are package resources in
`src/oh_streamer/resources/`; `resource_root()` resolves them in source, editable, wheel and frozen
builds. The wheel includes these application assets. Upstream Lada model weights, FFmpeg and ffprobe
remain external dependencies. The macOS app runs the bundled mpv helper above as a separate process;
it does not use libmpv. An explicit `--mpv <path>` override is accepted; source runs can discover
mpv on PATH, while packaged playback selects the nested helper by default. The nested bundle's
display name is Oh Streamer Player. The Windows player and both macOS bundles
use the shared Oh Streamer icon artwork (`app.ico` on Windows and `app.icns` on macOS). The spec
preserves console streams because worker progress and resolver JSON use stdout. Source launchers
continue to use root compatibility shims; installed-wheel queue children use `python -m oh_streamer`
because the source shims are absent.

Settings, queue data, logs, AirPlay credentials and the resolver override are stored under
`~/Library/Application Support/oh_streamer`. An existing checkout-level settings file,
credentials file or `resolvers.json` is copied there on first use only if the destination does not
exist; the original is kept. Existing checkout logs are left in place, and new logs go under the
user data directory's `logs/`. An explicit `--settings-file` path remains in effect. Resolver
settings resolve relative script paths from the active configuration file, preserving absolute
custom paths.

Set `LADA_MODEL_WEIGHTS_DIR` to the absolute external model directory before launch. The frozen
entry defaults it to `~/Library/Application Support/oh_streamer/model_weights`. Models are not
relocated or downloaded. `OH_STREAMER_PROJECT_ROOT` remains available for source development;
compiled Lada imports are resolved from the bundle rather than a runtime checkout.

FFmpeg and ffprobe remain system tools on PATH. The frozen entry prepends `/opt/homebrew/bin` and
`/usr/local/bin` for Finder launches so it can find them. The app does not require or alter a system
mpv installation; packaged playback uses the nested helper and fails clearly if that helper is
missing. Check external tool selection and versions with the doctor before testing.

Required on a real Mac before distribution: build and inspect PyInstaller warnings; verify torch,
PyAV, Cocoa/pywebview and every nested mpv dylib, install name and architecture. The build's
`mpv --no-config --load-scripts=no --version` smoke check only confirms startup; it does not validate
rendering. Move the `.app` outside the build directories (including to a path with spaces) and test
it on a Mac with no mpv installed or available on PATH. Confirm the nested player and any MoltenVK
ICD resolve from the bundle; test Finder launch, player IPC, playback, pause, seek, stop and shutdown;
and check queue/worker/resolver subprocesses, MPS and VideoToolbox, settings/log/queue writes outside
resources, and external FFmpeg/ffprobe discovery. Check that the separate Oh Streamer Player process
keeps normal Dock behavior and shows the complete branded icon and display name in the Dock and
Command-Tab switcher. Repeat on each supported native architecture.
Determine Developer ID signing, hardened-runtime entitlements, nested signing and notarization from
the actual bundle, then test Gatekeeper. No working `.app`, dylib compatibility, Developer ID signing
or notarization result is claimed by the Windows tests.
