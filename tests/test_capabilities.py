"""Validate the shipped capabilities.json manifest."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = REPO_ROOT / "capabilities.json"


def test_capabilities_exists() -> None:
    assert CAPABILITIES.exists(), "capabilities.json must ship at repo root"


def test_capabilities_shape() -> None:
    data = json.loads(CAPABILITIES.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"

    server = data["server"]
    assert server["api"] == "openai-chat-completions-subset"
    assert server["supports_streaming"] is True
    assert server["supports_tools"] is False  # load-bearing: flocode suppresses tools
    assert server["auth"]["bearer_token_env"] == "FLOCODE_SERVE_TOKEN"

    assert "/healthz" in server["endpoints"]["public"]
    assert "/v1/chat/completions" in server["endpoints"]["authed"]

    models = data["models"]
    assert isinstance(models, list) and len(models) >= 1
    qwen = next(m for m in models if m["id"] == "micromodel-qwen3-4b-dflash")
    assert qwen["backend"] == "dflash"
    assert qwen["target"] == "mlx-community/Qwen3-4B-bf16"
    assert qwen["draft"] == "z-lab/Qwen3-4B-DFlash-b16"
    assert qwen["architecture"] == "apple_silicon_mlx"
    assert "apache-2.0" in qwen["licenses"]
