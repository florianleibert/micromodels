"""Plain MLX backend — stock mlx_lm generation for models without a DFlash draft.

This is a thin wrapper around `mlx_lm.load` + `mlx_lm.generate` / `mlx_lm.stream_generate`
so models like Gemma 3n E2B can run through the same micromodel-ship server
the DFlash runtime exposes. Output shape matches ModelRuntime exactly so
server.py can treat both runtimes as duck-typed equivalents.

There is no speculative decoding here — throughput tracks stock mlx-lm. The
reason to use this backend is not speed; it is model coverage. Any MLX-
compatible bf16/4-bit model works.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .runtime import GenerationRequest  # reuse the same request dataclass


@dataclass(frozen=True)
class PlainMLXSpec:
    """Shape-compatible with runtime.ModelSpec so server.py stays backend-agnostic.

    DFlash has a (target, draft) pair; plain MLX has only a single model. We
    keep the same attribute names and set `draft` to an empty string plus a
    `draft_is_local=False` marker so existing log lines stay readable.
    """

    target: str
    draft: str = ""
    target_is_local: bool = False
    draft_is_local: bool = False


class PlainMLXRuntime:
    """Runtime adapter that exposes the ModelRuntime interface on top of mlx_lm.

    Public surface (must stay in sync with runtime.ModelRuntime):
      - spec: PlainMLXSpec with {target, draft, target_is_local, draft_is_local}
      - warm(): eagerly load the model so /healthz flips to ready
      - generate(request) -> dict with text/metrics/generated_token_count
      - stream_generate(request) -> iterator of delta / done events

    The lock around the model mirrors DFlash: MLX state is not safe across
    concurrent inference calls, so we serialize in-process.
    """

    def __init__(self, target_model: str | None, seed: int = 0) -> None:
        if target_model is None:
            raise ValueError("plain MLX runtime requires an explicit target model")
        self.spec = PlainMLXSpec(
            target=target_model,
            target_is_local=Path(target_model).exists(),
        )
        self.seed = seed
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Import lazily so unit tests that don't exercise generation don't
        # pay the MLX import cost (this module itself imports mx.core at
        # module top, which is already loaded by runtime.py).
        from mlx_lm.utils import load

        self._model, self._tokenizer = load(self.spec.target)

    def warm(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        from mlx_lm.generate import generate as mlx_generate

        with self._lock:
            self._ensure_loaded()
            t0 = time.perf_counter()
            peak_before = mx.get_peak_memory() / (1024**3)
            mx.reset_peak_memory()
            text = mlx_generate(
                self._model,
                self._tokenizer,
                prompt=request.prompt,
                max_tokens=request.max_new_tokens,
                verbose=False,
            )
            elapsed = max(time.perf_counter() - t0, 1e-6)
            peak = max(mx.get_peak_memory() / (1024**3), peak_before)
            # Estimate token counts from the tokenizer. Plain MLX's `generate`
            # does not return usage, so we round-trip through the tokenizer
            # to approximate — fine for usage reporting; not for billing.
            input_tokens = len(self._tokenizer.encode(request.prompt))
            output_tokens = len(self._tokenizer.encode(text)) - input_tokens
            if output_tokens < 0:
                output_tokens = 0
        gen_tps = output_tokens / elapsed if output_tokens > 0 else 0.0
        return {
            "text": text,
            "metrics": {
                "num_input_tokens": input_tokens,
                "num_generated_tokens": output_tokens,
                "finish_reason": "stop",
                "generation_tps": gen_tps,
                "end_to_end_tps": gen_tps,
                "avg_acceptance_length": 1.0,  # no speculation; trivially 1
                "peak_memory_gb": peak,
            },
            "generated_token_count": output_tokens,
            "output_token_count": output_tokens,
            "target_model": self.spec.target,
            "draft_model": "",
            "target_path": self.spec.target,
            "draft_path": "",
        }

    def stream_generate(self, request: GenerationRequest) -> Iterator[dict[str, Any]]:
        from mlx_lm.generate import stream_generate as mlx_stream_generate

        with self._lock:
            self._ensure_loaded()
            t0 = time.perf_counter()
            peak_before = mx.get_peak_memory() / (1024**3)
            mx.reset_peak_memory()
            input_tokens = len(self._tokenizer.encode(request.prompt))
            output_text = ""
            output_tokens = 0
            for chunk in mlx_stream_generate(
                self._model,
                self._tokenizer,
                prompt=request.prompt,
                max_tokens=request.max_new_tokens,
            ):
                # mlx_lm yields GenerationResponse objects. Grab the .text
                # field only — never fall back to str(chunk), which returns
                # the dataclass repr (``GenerationResponse(text='', token=...)``)
                # and would leak as a "delta" on the EOS chunk where text is
                # the empty string.
                text = getattr(chunk, "text", "") or ""
                if text:
                    output_text += text
                    output_tokens += 1
                    yield {"type": "delta", "text": text}
            elapsed = max(time.perf_counter() - t0, 1e-6)
            peak = max(mx.get_peak_memory() / (1024**3), peak_before)
        yield {
            "type": "done",
            "metrics": {
                "num_input_tokens": input_tokens,
                "num_generated_tokens": output_tokens,
                "finish_reason": "stop",
                "generation_tps": output_tokens / elapsed if output_tokens else 0.0,
                "end_to_end_tps": output_tokens / elapsed if output_tokens else 0.0,
                "avg_acceptance_length": 1.0,
                "peak_memory_gb": peak,
            },
            "generated_token_count": output_tokens,
            "text": output_text,
            "target_path": self.spec.target,
            "draft_path": "",
            "target_model": self.spec.target,
            "draft_model": "",
        }
