"""Registry tests. No MLX weights required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from micromodel_ship import registry

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = REPO_ROOT / "capabilities.json"


def test_default_model_exists() -> None:
    entry = registry.get(registry.DEFAULT_MODEL_ID)
    assert entry.id == registry.DEFAULT_MODEL_ID
    assert entry.backend in {"dflash", "plain_mlx"}


def test_dflash_entry_has_draft() -> None:
    entry = registry.get("qwen3-4b-dflash")
    assert entry.backend == "dflash"
    assert entry.draft, "DFlash backend must declare a draft repo"


def test_plain_mlx_entry_has_no_draft() -> None:
    entry = registry.get("gemma-3n-e2b")
    assert entry.backend == "plain_mlx"
    assert entry.draft == "", "plain-MLX entries must leave draft empty"


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError, match="unknown model id"):
        registry.get("does-not-exist")


def test_ids_sorted() -> None:
    ids = registry.ids()
    assert ids == sorted(ids)
    assert registry.DEFAULT_MODEL_ID in ids


def test_capabilities_json_in_sync_with_registry() -> None:
    """capabilities.json is load-bearing for flocode's config preset;
    every registry entry must appear in it so consumers see the same list."""
    data = json.loads(CAPABILITIES.read_text())
    shipped_ids = {m["id"] for m in data["models"]}
    for entry in registry.MODELS.values():
        expected_shipped_id = f"micromodel-{entry.id}"
        assert expected_shipped_id in shipped_ids, (
            f"capabilities.json missing entry for {entry.id}; "
            f"expected id={expected_shipped_id!r}"
        )
