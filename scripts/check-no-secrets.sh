#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PATTERN='(BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY|ssh-ed25519 AAAA|ssh-rsa AAAA|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|(?i:(api[_-]?key|secret|token))\s*[:=]\s*["'\'']?[A-Za-z0-9_./+=-]{24,})'
if rg -n -I -S \
  -g '!dist/**' \
  -g '!.git/**' \
  -g '!.venv/**' \
  -g '!scripts/check-no-secrets.sh' \
  "$PATTERN" "$ROOT_DIR"; then
  echo "secret-like material found" >&2
  exit 1
fi
echo "no secret-like material found in scanned text files"
