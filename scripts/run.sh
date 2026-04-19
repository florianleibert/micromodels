#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ $# -eq 0 ]]; then
  echo "usage: ./scripts/run.sh \"prompt\" [extra micromodel-ship run flags]" >&2
  exit 2
fi
if [[ "$1" != -* ]]; then
  PROMPT="$1"
  shift
  exec ./shipmodel run --prompt "$PROMPT" "$@"
fi
exec ./shipmodel run "$@"
