"""Regression test for the PlainMLXRuntime streaming path.

Verifies that the EOS chunk (GenerationResponse with ``text=""``) does NOT
leak through as a ``delta`` event containing the dataclass repr. Earlier
the code fell back to ``str(chunk)`` on a falsy ``.text`` value, which
rendered the final token as something like
``GenerationResponse(text='', token=2, finish_reason='stop', ...)`` on the
wire.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest


@dataclass
class FakeChunk:
    """Duck-types enough of mlx_lm.GenerationResponse for the streaming loop."""

    text: str
    token: int = 0
    finish_reason: str | None = None


class FakeTokenizer:
    def encode(self, s: str) -> list[int]:
        # Token count approximation is not under test; return something stable.
        return [0] * max(1, len(s.split()))


@pytest.fixture
def patched_mlx(monkeypatch):
    """Inject a fake ``stream_generate`` into the mlx_lm.generate *submodule*
    (not the function `mlx_lm.generate` that shadows it at package level) so
    PlainMLXRuntime's lazy ``from mlx_lm.generate import stream_generate``
    resolves to our stub.
    """
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        pytest.skip("mlx_lm not installed")

    submodule = sys.modules.get("mlx_lm.generate")
    if submodule is None or not isinstance(submodule, types.ModuleType):
        # Force-import the submodule so it's in sys.modules regardless of
        # what the package's __init__ re-exports under the same name.
        import importlib

        submodule = importlib.import_module("mlx_lm.generate")

    chunks: list[FakeChunk] = []

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, **kwargs):
        for c in chunks:
            yield c

    monkeypatch.setattr(submodule, "stream_generate", fake_stream_generate, raising=False)

    def set_chunks(values: list[FakeChunk]) -> None:
        chunks.clear()
        chunks.extend(values)

    return set_chunks


def test_empty_text_chunk_does_not_leak_repr(patched_mlx, monkeypatch):
    """The original bug: an EOS chunk with text='' emitted a delta containing
    the dataclass repr. After the fix, empty-text chunks are simply skipped."""
    from micromodel_ship import plain_mlx
    from micromodel_ship.runtime import GenerationRequest

    patched_mlx([
        FakeChunk(text="Hello"),
        FakeChunk(text=", "),
        FakeChunk(text="world"),
        FakeChunk(text=""),  # EOS chunk with empty text — the bug trigger
        FakeChunk(text="!"),  # and a final chunk to ensure we don't break early
    ])

    rt = plain_mlx.PlainMLXRuntime(target_model="fake-model")
    # Pre-populate model + tokenizer so _ensure_loaded is a no-op and we
    # never actually try to load weights.
    rt._model = object()
    rt._tokenizer = FakeTokenizer()

    events = list(rt.stream_generate(GenerationRequest(prompt="hi", max_new_tokens=8)))

    deltas = [e["text"] for e in events if e.get("type") == "delta"]
    done = [e for e in events if e.get("type") == "done"]

    assert deltas == ["Hello", ", ", "world", "!"], (
        f"empty-text chunk must be skipped, got deltas={deltas!r}"
    )
    for d in deltas:
        assert "GenerationResponse" not in d, (
            f"delta leaked dataclass repr: {d!r}"
        )
    assert len(done) == 1
    assert done[0]["text"] == "Hello, world!"


def test_all_empty_chunks_produces_no_deltas(patched_mlx):
    """Edge case: a stream that produces nothing but empty chunks yields
    zero delta events and still emits a done event."""
    from micromodel_ship import plain_mlx
    from micromodel_ship.runtime import GenerationRequest

    patched_mlx([FakeChunk(text=""), FakeChunk(text="")])

    rt = plain_mlx.PlainMLXRuntime(target_model="fake-model")
    rt._model = object()
    rt._tokenizer = FakeTokenizer()

    events = list(rt.stream_generate(GenerationRequest(prompt="hi", max_new_tokens=8)))

    assert not [e for e in events if e.get("type") == "delta"]
    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1
    assert done[0]["text"] == ""
    assert done[0]["generated_token_count"] == 0
