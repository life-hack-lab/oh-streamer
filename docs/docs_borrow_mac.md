# Mac source test kit

This kit contains Oh Streamer source, packaged runtime resources and tooling. Public
Lua, queue UI, and resolver default/example files are under
`src/oh_streamer/resources/`. It excludes user settings, logs, credentials,
private `resolvers.json`, video footage, virtual environments, model weights and
Git history. Tests are opt-in. It is an
experimental source setup; MPS, VideoToolbox, Cocoa, seek and audio synchronisation
still require testing on a real Mac.

## Build and verify before copying

Run from the repository root with the prepared Lada environment (on Windows use `..\lada\.venv\Scripts\python.exe`):

```sh
python borrow_mac_bundle.py --output borrow-mac.zip
python borrow_mac_bundle.py --verify borrow-mac.zip
# Optional public tests:
python borrow_mac_bundle.py --output borrow-mac-tests.zip --include-tests
```

`--project-root` defaults to the repository root. Inventory is taken
at execution, so new public root Python tools and packaging scripts are included.
The allowlist admits root compatibility shims, `src/oh_streamer/` Python modules
and package resources, `docs/*.md`, the named launchers and packaging source. Legacy benchmarks
are excluded; `benchmarks/bench_portability.py` is included. Source symlinks are rejected.
Review the manifest member names before sharing a bundle if you add public modules.

The archive has `manifest.json` and a `streamer/` folder holding the repository files (the macOS scripts expect this name). Every payload member has its size
and SHA256; the manifest excludes itself to avoid a circular digest. Verification
rejects extra/missing members, duplicates, symlinks, unsafe paths and mismatched
hashes. Hash verification detects accidental alteration; it does not authenticate
the sender. Verify before extracting. No archive extraction is performed by the tool.

## Prepare a separate upstream checkout

The full pinned model setup below targets Apple Silicon. Official PyTorch macOS
x86_64 binaries ended with 2.2.x ([official announcement](https://docs.pytorch.org/blog/pytorch2-2/));
the pinned torch 2.9.1 cannot be installed on an Intel Mac by following this normal
binary installation. Do not downgrade the model environment casually. Intel Macs
remain useful for independent media, IPC and desktop integration checks; see
`docs_portability_tools.md`. Full Intel model support and model-bearing app builds
are a separate compatibility task.

Extract into a new test directory. Its layout must become:

```text
borrowed-mac/
  manifest.json
  streamer/
  lada/                 # obtained separately, never included in the bundle
    .venv/
    model_weights/
```

The expected upstream baseline is Lada `0.11.1-dev`, revision `20cb34a20a83c72c87a991d2c949032c70085b16`, public
origin `https://github.com/ladaapp/lada.git`. This is the compatibility
baseline, not a record of where a tester's existing checkout came from. Manifest upstream
metadata lists dependency pins, patch names/hashes and upstream file hashes.
Local newline conventions can affect raw file hashes; compare the revision and
review local modifications too. No source clone or environment is overwritten.

In a **new** test directory, obtain the upstream revision explicitly:

```sh
brew install git uv ffmpeg mpv
git clone https://github.com/ladaapp/lada.git lada
git -C lada checkout --detach 20cb34a20a83c72c87a991d2c949032c70085b16
cd lada
uv sync --locked --extra cpu
```

Read `lada/docs/macOS_install.md` from that checkout and complete its **CLI**
instructions. The upstream CPU extra includes MPS support and pins torch 2.9.1 /
torchvision 0.24.1. It also pins ultralytics 8.4.4 and mmengine 0.10.7 because their
installed packages need patches. Do not install the upstream GTK GUI extra for Oh
Streamer. Follow the upstream patch instructions, using the actual site-packages
directory for your Python version (the guide's `python3.13` path is an example):

```sh
SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
patch -u -p1 -d "$SITE" < patches/increase_mms_time_limit.patch
patch -u -p1 -d "$SITE" < patches/remove_ultralytics_telemetry.patch
patch -u -p1 -d "$SITE" < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
```

Download weights using the upstream instructions and check their hashes against
upstream `model_weights/checksums_sha256.txt`. The manifest gives the v2/v4 fast/v4
accurate detection and generic v1.2 restoration filenames/hashes expected by this
baseline. Choose the model pair required by your test; weights are never bundled.
RIFE and third-party optional weights require separate upstream/source guidance.

From the new test directory, run:

```sh
chmod +x streamer/*.command
streamer/setup_macos.command
streamer/setup_macos.command --install-streamer
```

The first command prechecks tools, the sibling checkout/revision, Python architecture
and upstream package pins. The second explicitly installs Oh Streamer requirements
into that existing environment, constraining all installed package versions so the
model stack cannot be silently upgraded. It does not install or apply upstream
patches and does not download weights. Resolve conflicts by reviewing requirements;
do not bypass constraints by upgrading torch. On Apple Silicon check that the printed
architecture is `arm64`; Intel Macs use `x86_64`. Package-patch correctness still needs
your upstream checklist and a model smoke test.

## Rebuild public fixtures and run preflight

```sh
lada/.venv/bin/python streamer/portability_fixtures.py --output preflight/synthetic.mp4
lada/.venv/bin/python streamer/portability_doctor.py --help
lada/.venv/bin/python streamer/portability_acceptance.py --help
lada/.venv/bin/python streamer/benchmarks/bench_portability.py --help
```

The fixture defaults to 4 seconds, 320x180, 30 fps, H.264 video and a 440 Hz AAC
tone. Moving colour bars, frame numbers and `HH:MM:SS:FF` timecode make seek,
segmentation and audio checks visible. It uses Pillow's bundled font, avoiding
FFmpeg drawtext/system fonts. It is a **synthetic transport/audio test**, not a real
mosaic detection benchmark. For model quality and dense-mosaic performance use a
separate, legally obtained clip selected by the tester.

Use each tool's help for current report/output options. Concrete
acceptance commands and required-hardware profiles are documented in
`docs_portability_tools.md`. In particular, run with `--require-mps` and
`--require-videotoolbox` on Apple Silicon so CPU/software fallback cannot pass a
hardware acceptance test. Use a short local clip for the model queue check.

Keep diagnostic and
acceptance reports locally, and review them before sharing. Launch `streamer/Oh Streamer.command` to check the main entry and queue startup. For direct player diagnostics only, launch
`streamer/Oh Player.command` with the synthetic file.
Test MPS repair, VideoToolbox segmentation, playback/seek and queue recovery in that
order. Keep RIFE off until its separate probe succeeds. See `docs_macos_setup.md`
for optional browser/casting setup. Finder launch, menu bar behaviour and clean exit cannot be
validated by this source bundle alone.
