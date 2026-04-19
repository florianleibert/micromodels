#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p dist
tar -czf dist/micromodel-ship-offline.tar.gz \
  --exclude='.venv' \
  --exclude='dist' \
  -C "$(dirname "$ROOT_DIR")" \
  "$(basename "$ROOT_DIR")"
echo "wrote dist/micromodel-ship-offline.tar.gz"
