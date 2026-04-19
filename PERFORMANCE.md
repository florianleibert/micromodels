# Performance Metrics

Environment:

- Hardware: `Apple M5 Max`
- OS: `macOS 26.4`
- Target model: `mlx-community/Qwen3-4B-bf16`
- Draft model: `z-lab/Qwen3-4B-DFlash-b16`
- Draft block size: `16`
- Verify mode: `parallel-replay`
- Sampling: greedy, `temperature=0.0`
- Prompt tokens: `101`

## Headline Results

| Max new tokens | Runtime | Prompt tok/s | Generation tok/s | End-to-end tok/s | Peak memory |
|---:|---|---:|---:|---:|---:|
| 512 | Plain MLX-LM BF16 | 87.00 | 55.13 | 48.67 | 8.18 GB |
| 512 | DFlash BF16 | 1831.04 | 190.73 | 186.89 | 9.33 GB |
| 1024 | Plain MLX-LM BF16 | 53.73 | 48.18 | 44.05 | 8.28 GB |
| 1024 | DFlash BF16 | 1809.52 | 159.35 | 157.98 | 9.43 GB |

## Speedup Summary

| Max new tokens | Decode speedup | End-to-end speedup |
|---:|---:|---:|
| 512 | 3.46x | 3.84x |
| 1024 | 3.31x | 3.59x |

## DFlash Acceptance

| Max new tokens | Average acceptance |
|---:|---:|
| 512 | 9.12 |
| 1024 | 9.83 |

## 1024-token DFlash profile

- prefill time: `0.06s`
- decode time: `6.43s`
- total time: `6.48s`
- draft time: `1.11s`
- verify forward/logits time: `5.25s`
- verify total time: `5.31s`
- steps: `105`

## Interpretation

- DFlash is materially faster than plain MLX on this machine.
- The observed win is around `3.3x` to `3.8x`, depending on whether you compare decode-only or end-to-end throughput.
- Memory cost rises by about `~1.1 GB` relative to the plain BF16 target path in these measured runs.
