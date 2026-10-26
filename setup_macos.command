#!/bin/bash
# Precheck only by default. Explicit --install-streamer mutates an existing env.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
STREAMER_DIR="$(cd "$(dirname "$0")" && pwd)"
LADA_DIR="$(dirname "$STREAMER_DIR")/lada"
EXPECTED_REV="20cb34a20a83c72c87a991d2c949032c70085b16"
INSTALL=0
case "${1:-}" in
  "") ;;
  --install-streamer) INSTALL=1 ;;
  *) echo "Usage: $0 [--install-streamer]" >&2; exit 2 ;;
esac
if [ "$(uname -s)" != Darwin ]; then
  echo "This setup precheck is for macOS." >&2; exit 1
fi
for tool in git uv ffmpeg ffprobe mpv; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; see docs/docs_borrow_mac.md." >&2; exit 1; }
done
PYTHON="$LADA_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ] || [ ! -f "$LADA_DIR/pyproject.toml" ]; then
  echo "Prepare sibling lada/ and its .venv using upstream CLI instructions first." >&2
  echo "This script does not clone a repository or create/replace an environment." >&2
  exit 1
fi
ACTUAL_REV="$(git -C "$LADA_DIR" rev-parse HEAD)"
if [ "$ACTUAL_REV" != "$EXPECTED_REV" ]; then
  echo "Upstream revision differs: $ACTUAL_REV (expected $EXPECTED_REV)." >&2
  echo "Review compatibility and use manual setup; this script will not checkout or overwrite it." >&2
  exit 1
fi
# Import from the caller-selected sibling checkout, not a guessed Windows source.
cd "$LADA_DIR"
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import platform
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Lada requires Python >=3.12")
if platform.machine() not in {"arm64", "x86_64"}:
    raise SystemExit("Unexpected Python architecture: " + platform.machine())
for name, expected in {"torch": "2.9.1", "torchvision": "0.24.1", "ultralytics": "8.4.4", "mmengine": "0.10.7"}.items():
    actual = metadata.version(name).split("+")[0]
    if actual != expected:
        raise SystemExit(f"{name}={actual}; expected upstream CPU/MPS {expected}. Follow upstream setup first.")
import torch
print("Python:", sys.version.split()[0], platform.machine())
print("torch:", torch.__version__, "MPS available:", torch.backends.mps.is_available())
print("Check upstream's three required package patches and model checksums before model tests.")
PY
if [ "$INSTALL" = 1 ]; then
  # Freeze every installed distribution, protecting the prepared model stack.
  CONSTRAINTS="$(mktemp -t oh-streamer-constraints)"
  trap 'rm -f "$CONSTRAINTS"' EXIT
  "$PYTHON" - > "$CONSTRAINTS" <<'PY'
import importlib.metadata as metadata
import re
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name", "")
    version = distribution.version
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
        print(f"{name}=={version}")
PY
  uv pip install --python "$PYTHON" --constraint "$CONSTRAINTS" -r "$STREAMER_DIR/requirements-streamer.txt"
fi
cd "$STREAMER_DIR"
"$PYTHON" "$STREAMER_DIR/portability_doctor.py" --help
echo "Precheck completed. See docs/docs_borrow_mac.md for diagnosis and acceptance commands."
