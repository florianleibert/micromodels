#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROMPT="${1:-Write a Go HTTP server with a /health endpoint.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
METRICS_DIR="$ROOT_DIR/metrics"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="$METRICS_DIR/bench-$STAMP.txt"
LATEST_FILE="$METRICS_DIR/latest-bench.txt"

mkdir -p "$METRICS_DIR"

{
  echo "# micromodel-ship benchmark"
  echo
  echo "timestamp_utc: $STAMP"
  echo "prompt: $PROMPT"
  echo "max_new_tokens: $MAX_NEW_TOKENS"
  echo
  echo "## Plain MLX-LM BF16 baseline"
  uv run --directory /Users/floleibert/dev/dflash-mlx dflash-mlx-bench \
    --model mlx-community/Qwen3-4B-bf16 \
    --prompt "$PROMPT" \
    --warmup-prompts 1 \
    --num-prompts 1 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --no-history
  echo
  echo "## DFlash BF16"
  uv run --directory /Users/floleibert/dev/dflash-mlx dflash-mlx \
    --target-model "$ROOT_DIR/models/target" \
    --draft-model "$ROOT_DIR/models/draft" \
    --prompt "$PROMPT" \
    --warmup-runs 1 \
    --warmup-max-new-tokens 64 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --no-history \
    --profile
} 2>&1 | tee "$OUT_FILE"

cp "$OUT_FILE" "$LATEST_FILE"
echo
echo "wrote $OUT_FILE"
echo "updated $LATEST_FILE"
