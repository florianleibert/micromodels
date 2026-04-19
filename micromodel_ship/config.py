from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_TARGET_DIR = MODELS_DIR / "target"
LOCAL_DRAFT_DIR = MODELS_DIR / "draft"
DEFAULT_TARGET_REPO = "mlx-community/Qwen3-4B-bf16"
DEFAULT_DRAFT_REPO = "z-lab/Qwen3-4B-DFlash-b16"
DEFAULT_MODEL_NAME = "micromodel-qwen3-4b-dflash"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8051
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_VERIFY_MODE = "parallel-replay"
DEFAULT_VERIFY_CHUNK_SIZE = 4
DEFAULT_SPECULATIVE_TOKENS = None
DEFAULT_SEED = 0


@dataclass(frozen=True)
class ModelSpec:
    target: str
    draft: str
    target_is_local: bool
    draft_is_local: bool


def bundled_model_spec(
    target_override: str | None = None,
    draft_override: str | None = None,
) -> ModelSpec:
    if target_override:
        target = target_override
        target_is_local = Path(target_override).exists()
    elif LOCAL_TARGET_DIR.exists() and any(LOCAL_TARGET_DIR.iterdir()):
        target = str(LOCAL_TARGET_DIR)
        target_is_local = True
    else:
        target = DEFAULT_TARGET_REPO
        target_is_local = False

    if draft_override:
        draft = draft_override
        draft_is_local = Path(draft_override).exists()
    elif LOCAL_DRAFT_DIR.exists() and any(LOCAL_DRAFT_DIR.iterdir()):
        draft = str(LOCAL_DRAFT_DIR)
        draft_is_local = True
    else:
        draft = DEFAULT_DRAFT_REPO
        draft_is_local = False

    return ModelSpec(
        target=target,
        draft=draft,
        target_is_local=target_is_local,
        draft_is_local=draft_is_local,
    )
