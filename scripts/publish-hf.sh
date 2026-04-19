#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HF_REPO="${HF_REPO:-florianleibert/micromodel-ship}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.accounts/hugginface}"
TARBALL="dist/micromodel-ship-offline.tar.gz"

if [ ! -f "$HF_TOKEN_FILE" ]; then
  echo "error: HF token file not found at $HF_TOKEN_FILE" >&2
  exit 1
fi
export HF_TOKEN
HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"

if [ ! -f "$TARBALL" ]; then
  echo "building $TARBALL"
  ./scripts/package.sh
fi

echo "uploading README.md to $HF_REPO"
hf upload "$HF_REPO" HUGGINGFACE.md README.md --repo-type model --commit-message "update model card"

echo "uploading $TARBALL to $HF_REPO"
hf upload "$HF_REPO" "$TARBALL" "$(basename "$TARBALL")" --repo-type model --commit-message "update offline bundle"

echo "published to https://huggingface.co/$HF_REPO"
