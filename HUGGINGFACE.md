---
license: apache-2.0
library_name: mlx
tags:
  - mlx
  - apple-silicon
  - speculative-decoding
  - qwen3
  - dflash
  - offline-bundle
base_model:
  - mlx-community/Qwen3-4B-bf16
  - z-lab/Qwen3-4B-DFlash-b16
---

# micromodel-ship

Offline Apple Silicon inference bundle for Qwen3-4B with DFlash exact speculative decoding.

**Source code:** [github.com/florianleibert/micromodels](https://github.com/florianleibert/micromodels)

This repo hosts the shippable offline tarball (`micromodel-ship-offline.tar.gz`) that is too large to live in the GitHub repo. The tarball contains the full runnable runtime, both model payloads (target + draft), and the helper scripts needed to serve a local OpenAI-compatible API.

## Contents

`micromodel-ship-offline.tar.gz` bundles:

- target model: `mlx-community/Qwen3-4B-bf16`
- DFlash draft: `z-lab/Qwen3-4B-DFlash-b16`
- MLX-based runtime with exact speculative decoding
- minimal OpenAI-compatible API server (`POST /v1/chat/completions`)
- run, chat, serve, and benchmark scripts

## Quick start

```bash
curl -L -o micromodel-ship-offline.tar.gz \
  https://huggingface.co/florianleibert/micromodel-ship/resolve/main/micromodel-ship-offline.tar.gz
tar -xzf micromodel-ship-offline.tar.gz
cd micromodel-ship
uv sync
./scripts/serve.sh
```

Health check:

```bash
curl http://127.0.0.1:8051/healthz
```

## Performance

Measured on Apple M5 Max, macOS 26.4, parallel-replay verifier, 101-token prompt:

| Max new tokens | Runtime | Generation tok/s | End-to-end tok/s |
|---:|---|---:|---:|
| 512 | Plain MLX-LM BF16 | 55.13 | 48.67 |
| 512 | DFlash BF16 | 190.73 | 186.89 |
| 1024 | Plain MLX-LM BF16 | 48.18 | 44.05 |
| 1024 | DFlash BF16 | 159.35 | 157.98 |

Observed: `3.46x` decode / `3.84x` end-to-end speedup at 512 tokens. Full numbers in the GitHub repo's `PERFORMANCE.md`.

## Requirements

- Apple Silicon (M-series)
- macOS
- Python with `uv`

## Links

- GitHub repo (source, issues, releases): [florianleibert/micromodels](https://github.com/florianleibert/micromodels)
