"""Model registry — maps short model IDs to the backend and weights to load.

Keeping this centralized means the CLI's ``--model-id`` flag, the
``capabilities`` subcommand, and the server's /v1/models endpoint all agree
on what's supported without string-matching each model name in three places.

Adding a new entry:
  1. Append to MODELS below with backend in {"dflash", "plain_mlx"}.
  2. DFlash entries need (target, draft); plain-MLX entries need (target, "").
  3. If the model is not yet on disk, set target_repo so prefetch can pull
     it from Hugging Face on demand.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str
    backend: str  # "dflash" | "plain_mlx"
    target: str
    draft: str = ""
    context_tokens: int = 32768
    max_output_tokens: int = 4096
    approx_ram_gb: float = 0.0
    approx_disk_gb: float = 0.0
    licenses: tuple[str, ...] = ()
    notes: str = ""


MODELS: dict[str, ModelEntry] = {
    # Qwen3-4B DFlash — the flagship local model. Speculative decoding
    # produces 3.3–3.8× over plain MLX on M-series; paired draft.
    "qwen3-4b-dflash": ModelEntry(
        id="qwen3-4b-dflash",
        display_name="Qwen3-4B DFlash",
        backend="dflash",
        target="mlx-community/Qwen3-4B-bf16",
        draft="z-lab/Qwen3-4B-DFlash-b16",
        context_tokens=32768,
        max_output_tokens=4096,
        approx_ram_gb=10.0,
        approx_disk_gb=18.0,
        licenses=("apache-2.0",),
        notes="Exact speculative decoding. No tool calls. No prefix cache.",
    ),
    # Gemma 3n E2B — plain MLX backend. No paired DFlash draft exists, so
    # throughput matches stock mlx-lm; the reason to use this profile is
    # model coverage (Gemma's chat template, multilingual strength, ~2 GB
    # runtime footprint). Repo ID targets the Hugging Face MLX community
    # build; swap via --target-model if a better MLX conversion lands.
    "gemma-3n-e2b": ModelEntry(
        id="gemma-3n-e2b",
        display_name="Gemma 3n E2B (plain MLX)",
        backend="plain_mlx",
        target="mlx-community/gemma-3n-E2B-it-bf16",
        draft="",
        context_tokens=32768,
        max_output_tokens=2048,
        approx_ram_gb=4.0,
        approx_disk_gb=10.0,
        licenses=("gemma",),
        notes=(
            "Plain MLX — no DFlash speedup. Use --target-model to substitute "
            "a different Gemma MLX conversion (e.g. a 4-bit quant)."
        ),
    ),
    # Qwen3-30B-A3B Instruct (2507) at 4-bit — MoE with 3B active params.
    # Plain MLX for now; no DFlash draft exists for this size on Apple
    # Silicon yet. Sized to fit a 32 GB+ M-series Mac (peak memory around
    # 18–22 GB under generation). The "2507" tag is Qwen's July 2025
    # instruct refresh and is the recommended chat checkpoint.
    "qwen3-30b-a3b-instruct": ModelEntry(
        id="qwen3-30b-a3b-instruct",
        display_name="Qwen3-30B-A3B Instruct 4-bit (MoE, plain MLX)",
        backend="plain_mlx",
        target="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        draft="",
        context_tokens=131072,
        max_output_tokens=8192,
        approx_ram_gb=20.0,
        approx_disk_gb=18.0,
        licenses=("apache-2.0",),
        notes=(
            "MoE with 3B active params. Plain MLX — no DFlash speedup. "
            "Pick this over qwen3-4b-dflash when model quality matters more "
            "than decode speed. Use --target-model to substitute a higher "
            "bit-width variant (6-bit or 8-bit) if your Mac has the RAM."
        ),
    ),
}

DEFAULT_MODEL_ID = "qwen3-4b-dflash"


def get(model_id: str) -> ModelEntry:
    """Return the registry entry for model_id or raise a clear KeyError."""
    if model_id not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        raise KeyError(f"unknown model id {model_id!r} (known: {known})")
    return MODELS[model_id]


def ids() -> list[str]:
    return sorted(MODELS.keys())
