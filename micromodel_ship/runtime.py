from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.generate import wired_limit

from dflash_mlx.api import DFlashGenerator
from dflash_mlx.chat_cli import build_prompt as build_history_prompt
from dflash_mlx.runtime import (
    add_profile_elapsed,
    generated_token_count,
    peak_memory_gb,
    profile_start,
    sample_tokens,
    stop_position,
    trim_draft_cache,
    verify_block_chunked,
    verify_block_parallel_greedy_argmax,
    verify_block_parallel_lazy_logits,
    verify_block_parallel_replay,
    verify_block_stream,
)

from .config import (
    DEFAULT_DRAFT_REPO,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    DEFAULT_SPECULATIVE_TOKENS,
    DEFAULT_TARGET_REPO,
    DEFAULT_TEMPERATURE,
    DEFAULT_VERIFY_CHUNK_SIZE,
    DEFAULT_VERIFY_MODE,
    LOCAL_DRAFT_DIR,
    LOCAL_TARGET_DIR,
    ModelSpec,
    bundled_model_spec,
)


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    speculative_tokens: int | None = DEFAULT_SPECULATIVE_TOKENS
    verify_mode: str = DEFAULT_VERIFY_MODE
    verify_chunk_size: int = DEFAULT_VERIFY_CHUNK_SIZE
    profile: bool = False
    skip_special_tokens: bool = True


class ModelRuntime:
    def __init__(
        self,
        target_model: str | None = None,
        draft_model: str | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.spec: ModelSpec = bundled_model_spec(target_model, draft_model)
        self.seed = seed
        self._lock = threading.Lock()
        self._generator: DFlashGenerator | None = None

    @property
    def generator(self) -> DFlashGenerator:
        if self._generator is None:
            self._generator = DFlashGenerator(
                target_model=self.spec.target,
                draft_model=self.spec.draft,
                seed=self.seed,
            )
        return self._generator

    def warm(self) -> None:
        _ = self.generator

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        with self._lock:
            result = self.generator.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                speculative_tokens=request.speculative_tokens,
                verify_mode=request.verify_mode,
                verify_chunk_size=request.verify_chunk_size,
                skip_special_tokens=request.skip_special_tokens,
                profile=request.profile,
            )
        return {
            "text": result.text,
            "metrics": result.metrics,
            "generated_token_count": len(result.generated_tokens),
            "output_token_count": len(result.output_tokens),
            "target_model": self.spec.target,
            "draft_model": self.spec.draft,
            "target_path": str(self.generator.target_model_path),
            "draft_path": str(self.generator.draft_path),
        }

    def stream_generate(self, request: GenerationRequest) -> Iterator[dict[str, Any]]:
        with self._lock:
            generator = self.generator
            target = generator.target
            draft = generator.draft
            prompt_tokens = generator.encode_prompt(request.prompt)
            with wired_limit(target.model):
                mx.reset_peak_memory()
                yield from _stream_dflash_generate(
                    target=target,
                    draft=draft,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    stop_token_ids=target.stop_token_ids(),
                    layer_ids=draft.target_layer_ids,
                    speculative_tokens=request.speculative_tokens,
                    verify_mode=request.verify_mode,
                    verify_chunk_size=request.verify_chunk_size,
                    skip_special_tokens=request.skip_special_tokens,
                    profile=request.profile,
                    target_path=str(generator.target_model_path),
                    draft_path=str(generator.draft_path),
                    target_model=self.spec.target,
                    draft_model=self.spec.draft,
                )


def _verify_block(
    *,
    verify_mode: str,
    target: Any,
    target_cache: list[Any],
    block_tokens: list[int],
    block_size: int,
    temperature: float,
    layer_ids: list[int],
    verify_chunk_size: int,
    profile_times: dict[str, float] | None,
) -> tuple[int, int, mx.array]:
    if verify_mode == "stream":
        return verify_block_stream(
            target=target,
            target_cache=target_cache,
            block_tokens=block_tokens,
            temperature=temperature,
            layer_ids=layer_ids,
        )
    if verify_mode == "chunked":
        return verify_block_chunked(
            target=target,
            target_cache=target_cache,
            block_tokens=block_tokens,
            draft_block_size=block_size,
            temperature=temperature,
            layer_ids=layer_ids,
            verify_chunk_size=verify_chunk_size,
        )
    if verify_mode == "parallel-lazy-logits":
        return verify_block_parallel_lazy_logits(
            target=target,
            target_cache=target_cache,
            block_tokens=block_tokens,
            draft_block_size=block_size,
            temperature=temperature,
            layer_ids=layer_ids,
            logit_chunk_size=verify_chunk_size,
            profile=profile_times,
        )
    if verify_mode == "parallel-greedy-argmax":
        return verify_block_parallel_greedy_argmax(
            target=target,
            target_cache=target_cache,
            block_tokens=block_tokens,
            draft_block_size=block_size,
            temperature=temperature,
            layer_ids=layer_ids,
            profile=profile_times,
        )
    return verify_block_parallel_replay(
        target=target,
        target_cache=target_cache,
        block_tokens=block_tokens,
        draft_block_size=block_size,
        temperature=temperature,
        layer_ids=layer_ids,
        profile=profile_times,
    )


def _stream_dflash_generate(
    *,
    target: Any,
    draft: Any,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    speculative_tokens: int | None,
    verify_mode: str,
    verify_chunk_size: int,
    skip_special_tokens: bool,
    profile: bool,
    target_path: str,
    draft_path: str,
    target_model: str,
    draft_model: str,
) -> Iterator[dict[str, Any]]:
    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    profile_times: dict[str, float] | None = {} if profile else None
    prompt_len = int(prompt_tokens.shape[0])
    total_max_tokens = prompt_len + max_new_tokens
    if speculative_tokens is None:
        block_size = draft.block_size
    else:
        block_size = max(1, min(speculative_tokens, draft.block_size))

    tokenizer = target.tokenizer
    emitted_text = ""

    def emit_delta(tokens: list[int]) -> Iterator[dict[str, Any]]:
        nonlocal emitted_text
        decoded = tokenizer.decode(
            tokens[prompt_len:], skip_special_tokens=skip_special_tokens
        )
        if len(decoded) > len(emitted_text) and decoded.startswith(emitted_text):
            delta = decoded[len(emitted_text):]
            emitted_text = decoded
            if delta:
                yield {"type": "delta", "text": delta}
        elif decoded != emitted_text:
            emitted_text = decoded

    sync_start = time.perf_counter()
    logits, target_hidden = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits, target_hidden)
    prefill_time = time.perf_counter() - sync_start

    output_tokens = prompt_tokens.tolist() + [first_token]
    start = prompt_len
    acceptance_lengths: list[int] = []

    yield from emit_delta(output_tokens)

    decode_start = time.perf_counter()
    while start < total_max_tokens:
        draft_start = profile_start(profile_times)
        block_tokens = [output_tokens[start]] + [draft.mask_token_id] * (block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.embed_tokens(block_input)

        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = target.lm_head_logits(draft_hidden[:, 1:, :])
        drafted_tokens = sample_tokens(draft_logits, temperature)
        mx.eval(drafted_tokens)
        trim_draft_cache(draft_cache, block_size)
        drafted_suffix = drafted_tokens[0].tolist()
        block_tokens[1:] = drafted_suffix[: block_size - 1]
        add_profile_elapsed(profile_times, "draft_time_s", draft_start)

        verify_start = profile_start(profile_times)
        accepted_inputs, posterior_token, verifier_hidden = _verify_block(
            verify_mode=verify_mode,
            target=target,
            target_cache=target_cache,
            block_tokens=block_tokens,
            block_size=block_size,
            temperature=temperature,
            layer_ids=layer_ids,
            verify_chunk_size=verify_chunk_size,
            profile_times=profile_times,
        )
        acceptance_lengths.append(accepted_inputs)

        target_hidden = verifier_hidden[:, :accepted_inputs, :]
        hidden_start = profile_start(profile_times)
        if profile_times is not None:
            mx.eval(target_hidden)
        add_profile_elapsed(profile_times, "verify_target_hidden_time_s", hidden_start)
        add_profile_elapsed(profile_times, "verify_time_s", verify_start)

        bookkeeping_start = profile_start(profile_times)
        output_tokens = output_tokens[:start]
        output_tokens.extend(block_tokens[:accepted_inputs])
        output_tokens.append(posterior_token)
        start += accepted_inputs
        add_profile_elapsed(profile_times, "bookkeeping_time_s", bookkeeping_start)

        stop_idx = stop_position(output_tokens, prompt_len, stop_token_ids)
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            yield from emit_delta(output_tokens)
            break

        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            yield from emit_delta(output_tokens)
            break

        yield from emit_delta(output_tokens)

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    generated_tokens = generated_token_count(output_tokens, prompt_len)
    total_time = prefill_time + decode_time

    metrics: dict[str, Any] = {
        "num_input_tokens": prompt_len,
        "num_output_tokens": generated_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "prompt_tps": prompt_len / max(prefill_time, 1e-9),
        "generation_tps": generated_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": generated_tokens / max(total_time, 1e-9),
        "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
        "acceptance_lengths": acceptance_lengths,
        "peak_memory_gb": peak_memory_gb(),
        "target_cache_summary": target.cache_summary(target_cache),
        "speculative_tokens": block_size,
    }
    if profile_times is not None:
        profiled_time = sum(
            profile_times.get(key, 0.0)
            for key in ("draft_time_s", "verify_time_s", "bookkeeping_time_s")
        )
        metrics["profile"] = {
            **profile_times,
            "unattributed_decode_time_s": decode_time - profiled_time,
            "steps": len(acceptance_lengths),
        }

    yield {
        "type": "done",
        "metrics": metrics,
        "generated_token_count": generated_tokens,
        "text": emitted_text,
        "target_path": target_path,
        "draft_path": draft_path,
        "target_model": target_model,
        "draft_model": draft_model,
    }


def prefetch_models(
    target_repo: str = DEFAULT_TARGET_REPO,
    draft_repo: str = DEFAULT_DRAFT_REPO,
) -> dict[str, str]:
    LOCAL_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    target_snapshot = Path(snapshot_download(repo_id=target_repo))
    draft_snapshot = Path(snapshot_download(repo_id=draft_repo))
    _replace_dir_with_real_files(target_snapshot, LOCAL_TARGET_DIR)
    _replace_dir_with_real_files(draft_snapshot, LOCAL_DRAFT_DIR)
    return {"target": str(LOCAL_TARGET_DIR), "draft": str(LOCAL_DRAFT_DIR)}


def build_chat_prompt_from_messages(messages: list[dict[str, Any]], max_turns: int = 6) -> str:
    if not messages:
        raise ValueError("messages are required")

    system_lines: list[str] = []
    history: list[tuple[str, str]] = []
    pending_user: str | None = None

    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        content = flatten_content(msg.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_lines.append(content)
        elif role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is not None:
                history.append((pending_user, content))
                pending_user = None
            else:
                history.append(("", content))
        else:
            if pending_user is not None:
                history.append((pending_user, content))
                pending_user = None
            else:
                history.append((role, content))

    if pending_user is None:
        raise ValueError("the final message must be a user message")

    prompt = build_history_prompt(history, pending_user, max_turns=max_turns)
    if system_lines:
        return "System: " + "\n".join(system_lines) + "\n\n" + prompt
    return prompt


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text = item["text"].strip()
                    if text:
                        parts.append(text)
                        continue
                if isinstance(item.get("content"), str):
                    text = item["content"].strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        if isinstance(content.get("content"), str):
            return content["content"].strip()
    return str(content).strip()


def _replace_dir_with_real_files(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=False)
