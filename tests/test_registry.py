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


def test_qwen3_30b_a3b_instruct_entry() -> None:
    """Validate the Qwen3-30B MoE entry. Load-bearing for flocode's preset:
    if the HF repo ID changes, the corresponding `local-qwen3-30b-a3b`
    preset in openflo will silently point at nothing useful."""
    entry = registry.get("qwen3-30b-a3b-instruct")
    assert entry.backend == "plain_mlx"
    assert entry.target == "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
    assert entry.draft == ""
    assert entry.context_tokens == 131072
    assert "apache-2.0" in entry.licenses


def test_qwen3_4b_abliterated_entry() -> None:
    """Validate the abliterated Qwen3-4B entry. The HF target repo ID is
    load-bearing: the companion flocode preset resolves against it, and the
    abliteration pipeline in flo/abliteration publishes to this exact name.
    If you change it, update both in the same PR."""
    entry = registry.get("qwen3-4b-abliterated")
    assert entry.backend == "plain_mlx"
    assert entry.target == "fabianbaier/Qwen3-4B-abliterated-4bit"
    assert entry.draft == "", "abliterated entry should have no DFlash draft"
    assert entry.context_tokens == 32768
    assert "apache-2.0" in entry.licenses
    # Safety-relevant metadata must be present so operators immediately see
    # this is not a drop-in safe replacement for qwen3-4b-dflash.
    assert "abliterat" in entry.notes.lower() or "refusal" in entry.notes.lower()


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
