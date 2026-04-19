from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from dflash_mlx.api import DFlashGenerator
from dflash_mlx.chat_cli import build_prompt as build_history_prompt

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
