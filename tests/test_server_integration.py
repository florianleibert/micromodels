"""HTTP-level integration tests with a stubbed runtime.

These do not load MLX weights; they exercise the bind-before-warm state
machine, auth middleware, and warming-to-ready transition via a fake
runtime object duck-typed to ModelRuntime.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

import pytest

# NOTE: importing server pulls in runtime which imports mlx. On Apple Silicon
# dev boxes this is fine; on a non-Apple CI we'd need to stub mlx in sys.modules
# before import. Scoped to the collection here via an import guard.
try:
    from micromodel_ship.server import MicroModelServer
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"mlx stack not importable: {exc}", allow_module_level=True)


@dataclass
class FakeSpec:
    target: str = "fake/target"
    draft: str = "fake/draft"


class FakeRuntime:
    """Duck-typed ModelRuntime stub — no MLX weights involved."""

    def __init__(self, warm_delay: float = 0.0, warm_should_raise: bool = False) -> None:
        self.spec = FakeSpec()
        self._warm_delay = warm_delay
        self._warm_should_raise = warm_should_raise
        self.warmed = False

    def warm(self) -> None:
        time.sleep(self._warm_delay)
        if self._warm_should_raise:
            raise RuntimeError("synthetic warmup failure")
        self.warmed = True

    def generate(self, request: Any) -> dict[str, Any]:
        return {
            "text": "hello from fake",
            "metrics": {
                "num_input_tokens": 5,
                "num_generated_tokens": 3,
                "finish_reason": "stop",
                "generation_tps": 100.0,
                "end_to_end_tps": 90.0,
                "avg_acceptance_length": 8.0,
                "peak_memory_gb": 4.2,
            },
            "generated_token_count": 3,
            "target_path": self.spec.target,
            "draft_path": self.spec.draft,
        }

    def stream_generate(self, request: Any) -> Iterator[dict[str, Any]]:
        yield {"type": "delta", "text": "hello "}
        yield {"type": "delta", "text": "world"}
        yield {
            "type": "done",
            "metrics": {
                "num_input_tokens": 5,
                "num_generated_tokens": 2,
                "finish_reason": "stop",
                "generation_tps": 100.0,
                "end_to_end_tps": 90.0,
                "avg_acceptance_length": 8.0,
                "peak_memory_gb": 4.2,
            },
            "generated_token_count": 2,
            "target_path": self.spec.target,
            "draft_path": self.spec.draft,
        }


def _start_server(runtime: FakeRuntime, token: str = "") -> tuple[MicroModelServer, str, threading.Thread]:
    server = MicroModelServer(runtime=runtime, host="127.0.0.1", port=0, server_token=token)
    host, port = server.start()
    thread = threading.Thread(target=server.wait, daemon=True)
    thread.start()
    return server, f"http://{host}:{port}", thread


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class TestWarmingStateMachine:
    def test_healthz_transitions_warming_to_ready(self) -> None:
        runtime = FakeRuntime(warm_delay=0.3)
        server, base, thread = _start_server(runtime)
        try:
            # Immediately after start: warming
            status, body = _get(f"{base}/healthz")
            assert status == 503
            assert body["status"] == "warming"
            assert body["ok"] is False

            # Wait for warmup to complete
            deadline = time.time() + 3.0
            while time.time() < deadline:
                status, body = _get(f"{base}/healthz")
                if status == 200:
                    break
                time.sleep(0.05)
            assert status == 200
            assert body["status"] == "ready"
            assert body["ok"] is True
            assert runtime.warmed is True
        finally:
            server.stop()
            thread.join(timeout=2)

    def test_chat_completions_blocks_during_warming(self) -> None:
        runtime = FakeRuntime(warm_delay=0.5)
        server, base, thread = _start_server(runtime)
        try:
            status, body = _post(
                f"{base}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 503
            assert "warming" in body["error"]["message"].lower()
        finally:
            server.stop()
            thread.join(timeout=2)

    def test_warmup_error_surfaces_in_healthz(self) -> None:
        runtime = FakeRuntime(warm_delay=0.05, warm_should_raise=True)
        server, base, thread = _start_server(runtime)
        try:
            time.sleep(0.3)  # let warmup thread raise
            status, body = _get(f"{base}/healthz")
            assert status == 503
            assert body["status"] == "error"
            assert "synthetic warmup failure" in body["error"]
        finally:
            server.stop()
            thread.join(timeout=2)


class TestAuth:
    def test_public_paths_reachable_without_token(self) -> None:
        runtime = FakeRuntime()
        server, base, thread = _start_server(runtime, token="s3cret")
        try:
            time.sleep(0.1)  # let warmup finish
            status, _ = _get(f"{base}/healthz")
            assert status == 200
            status, _ = _get(f"{base}/v1/models")
            assert status == 200
        finally:
            server.stop()
            thread.join(timeout=2)

    def test_authed_paths_require_token(self) -> None:
        runtime = FakeRuntime()
        server, base, thread = _start_server(runtime, token="s3cret")
        try:
            time.sleep(0.1)
            # no header -> 401
            status, body = _get(f"{base}/metrics")
            assert status == 401
            assert body["error"]["message"] == "unauthorized"

            # wrong token -> 401
            status, _ = _get(f"{base}/metrics", headers={"Authorization": "Bearer wrong"})
            assert status == 401

            # correct token -> 200
            status, _ = _get(f"{base}/metrics", headers={"Authorization": "Bearer s3cret"})
            assert status == 200
        finally:
            server.stop()
            thread.join(timeout=2)

    def test_chat_completions_requires_token_after_warm(self) -> None:
        runtime = FakeRuntime()
        server, base, thread = _start_server(runtime, token="s3cret")
        try:
            time.sleep(0.1)
            status, _ = _post(
                f"{base}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 401

            status, body = _post(
                f"{base}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer s3cret"},
            )
            assert status == 200
            assert body["choices"][0]["message"]["content"] == "hello from fake"
        finally:
            server.stop()
            thread.join(timeout=2)

    def test_no_token_means_no_auth(self) -> None:
        runtime = FakeRuntime()
        server, base, thread = _start_server(runtime, token="")
        try:
            time.sleep(0.1)
            status, _ = _get(f"{base}/metrics")
            assert status == 200
            status, _ = _post(
                f"{base}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 200
        finally:
            server.stop()
            thread.join(timeout=2)


class TestEndpoints:
    def test_models_endpoint(self) -> None:
        runtime = FakeRuntime()
        server, base, thread = _start_server(runtime)
        try:
            status, body = _get(f"{base}/v1/models")
            assert status == 200
            assert body["object"] == "list"
            assert body["data"][0]["id"] == "micromodel-qwen3-4b-dflash"
        finally:
            server.stop()
            thread.join(timeout=2)

    def test_unknown_endpoint_returns_404(self) -> None:
        runtime = FakeRuntime()
        server, base, thread = _start_server(runtime)
        try:
            status, _ = _get(f"{base}/unknown")
            assert status == 404
        finally:
            server.stop()
            thread.join(timeout=2)
