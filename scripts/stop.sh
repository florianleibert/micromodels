#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PORT="${1:-${MICROMODEL_SHIP_PORT:-8051}}"
PIDS="$(lsof -t -nP -iTCP:${PORT} -sTCP:LISTEN || true)"
if [[ -z "$PIDS" ]]; then
  echo "no listener on port ${PORT}"
  exit 0
fi
echo "$PIDS" | xargs kill
echo "stopped listener(s) on port ${PORT}: $PIDS"
