#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PORT="${1:-${MICROMODEL_SHIP_PORT:-8051}}"
PIDS="$(lsof -t -nP -iTCP:${PORT} -sTCP:LISTEN || true)"

if [[ -z "$PIDS" ]]; then
  echo "micromodel-ship is not listening on port ${PORT}"
  exit 1
fi

echo "micromodel-ship listener(s) on port ${PORT}: $PIDS"
echo
echo "healthz:"
curl -fsS "http://127.0.0.1:${PORT}/healthz"
echo
echo
echo "metrics:"
curl -fsS "http://127.0.0.1:${PORT}/metrics"
echo
