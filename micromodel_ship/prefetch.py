from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import DEFAULT_DRAFT_REPO, DEFAULT_TARGET_REPO, LOCAL_DRAFT_DIR, LOCAL_TARGET_DIR


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


def _replace_dir_with_real_files(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=False)
