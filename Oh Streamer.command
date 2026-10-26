#!/bin/bash
# Main macOS launcher for Oh Streamer. Finder passes arguments through unchanged.
# The root Python entry is a no-install shim for the src/oh_streamer package.
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="${OH_STREAMER_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
OH_PY="${OH_STREAMER_PYTHON:-$PROJECT_DIR/lada/.venv/bin/python}"

export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/homebrew/sbin:$PATH"

if [[ ! -x "$OH_PY" ]]; then
  cat >&2 <<EOF
Lada's Python environment was not found:
  $OH_PY

Complete the source setup in streamer/docs/docs_macos_setup.md first.
EOF
  read -r -p 'Press Return to close this window...'
  exit 1
fi

exec "$OH_PY" "$SCRIPT_DIR/queue_app.py" "$@"
