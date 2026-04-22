from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .auth import check_bearer
from .config import (
    DEFAULT_HOST,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PORT,
    DEFAULT_TEMPERATURE,
    SERVE_TOKEN_ENV,
)
from .runtime import GenerationRequest, ModelRuntime, build_chat_prompt_from_messages

# Public (unauthenticated) endpoints. Everything else requires bearer auth when
# a token is configured. Kept narrow so supervisors can poll health/metadata
# without embedding the secret, while prompts and completion metrics stay gated.
PUBLIC_PATHS: frozenset[str] = frozenset({"/healthz", "/v1/models"})


class MicroModelServer:
    """Minimal OpenAI-compatible chat server.

    Binds the HTTP socket *before* warming the model so that supervisors can
    poll ``/healthz`` immediately and distinguish "still warming" from "dead".
    Warmup runs on a background thread and flips the ready flag when done; any
    attempt to generate before then returns 503.

    If ``server_token`` is non-empty, authenticated endpoints require a
    ``Authorization: Bearer <token>`` header (constant-time compared).

    TODO(unix-socket): Phase 2 should add AF_UNIX binding so the socket
    filesystem permissions gate access, rendering the bearer token redundant
    on single-user machines. Requires ThreadingUnixStreamServer + a client
    that can dial a unix socket (flocode's OpenAI provider does not yet).
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model_name: str = DEFAULT_MODEL_NAME,
        server_token: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.model_name = model_name
        self.server_token = server_token or ""
        self._ready = threading.Event()
        self._warm_error: BaseException | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._warm_thread: threading.Thread | None = None

    def _warm(self) -> None:
        try:
            self.runtime.warm()
        except BaseException as exc:
            self._warm_error = exc
            return
        self._ready.set()

    def _check_auth(self, header_value: str | None) -> bool:
        return check_bearer(self.server_token, header_value)

    def start(self) -> tuple[str, int]:
        """Bind the HTTP socket and start warmup; return the bound (host, port).

        Safe to call from tests: pair with ``wait()`` (blocking) or let the
        caller run ``self._httpd.serve_forever()`` on a thread and ``stop()``
        to shut it down.
        """
        self._build_httpd()
        assert self._httpd is not None
        warm_thread = threading.Thread(target=self._warm, name="micromodel-warm", daemon=True)
        warm_thread.start()
        self._warm_thread = warm_thread
        return self._httpd.server_address[0], self._httpd.server_address[1]

    def wait(self) -> None:
        """Serve requests until ``stop()`` is called."""
        if self._httpd is None:
            raise RuntimeError("start() must be called before wait()")
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()

    def serve_forever(self) -> None:
        """Bind, start warmup in the background, serve until shutdown."""
        host, port = self.start()
        auth_mode = "bearer" if self.server_token else "none"
        print(f"[micromodel-ship] bound http://{host}:{port} auth={auth_mode}")
        print(f"[micromodel-ship] model={self.model_name} target={self.runtime.spec.target}")
        print(f"[micromodel-ship] draft={self.runtime.spec.draft}")
        print("[micromodel-ship] warming up in background; /healthz reports status")
        self.wait()

    def _build_httpd(self) -> None:
        server = self
        model_name = self.model_name
        runtime = self.runtime
        server_state: dict[str, Any] = {"last_completion": None}

        class Handler(BaseHTTPRequestHandler):
            server_version = "micromodel-ship/0.1"

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _unauthorized(self) -> None:
                self.send_response(HTTPStatus.UNAUTHORIZED)
                self.send_header("Content-Type", "application/json")
                self.send_header("WWW-Authenticate", 'Bearer realm="micromodel-ship"')
                body = json.dumps({"error": {"message": "unauthorized"}}).encode("utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _require_auth(self) -> bool:
                if self.path in PUBLIC_PATHS:
                    return True
                if server._check_auth(self.headers.get("Authorization")):
                    return True
                self._unauthorized()
                return False

            def _healthz_payload(self) -> tuple[int, dict[str, Any]]:
                if server._warm_error is not None:
                    return (
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "status": "error",
                            "error": str(server._warm_error),
                            "model": model_name,
                        },
                    )
                if not server._ready.is_set():
                    return (
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "status": "warming",
                            "model": model_name,
                        },
                    )
                return (
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "status": "ready",
                        "model": model_name,
                        "target": runtime.spec.target,
                        "draft": runtime.spec.draft,
                    },
                )

            def do_GET(self) -> None:
                if not self._require_auth():
                    return
                if self.path == "/healthz":
                    status, payload = self._healthz_payload()
                    self._send_json(status, payload)
                    return
                if self.path == "/metrics":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": server._ready.is_set(),
                            "status": "ready" if server._ready.is_set() else "warming",
                            "model": model_name,
                            "target": runtime.spec.target,
                            "draft": runtime.spec.draft,
                            "last_completion": server_state["last_completion"],
                        },
                    )
                    return
                if self.path == "/v1/models":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": model_name,
                                    "object": "model",
                                    "owned_by": "local",
                                }
                            ],
                        },
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                if not self._require_auth():
                    return
                if self.path != "/v1/chat/completions":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                    return
                if not server._ready.is_set():
                    status = HTTPStatus.SERVICE_UNAVAILABLE
                    if server._warm_error is not None:
                        self._send_json(status, {"error": {"message": f"warmup failed: {server._warm_error}"}})
                    else:
                        self._send_json(status, {"error": {"message": "model warming, retry shortly"}})
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(content_length)
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": f"invalid json: {exc}"}})
                    return

                try:
                    prompt = build_chat_prompt_from_messages(payload.get("messages") or [])
                    max_new_tokens = int(payload.get("max_completion_tokens") or payload.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
                    temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                    return

                if payload.get("stream"):
                    self._handle_stream(
                        runtime=runtime,
                        server_state=server_state,
                        model_name=model_name,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        profile=bool(payload.get("profile", False)),
                    )
                    return

                try:
                    result = runtime.generate(
                        GenerationRequest(
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            profile=bool(payload.get("profile", False)),
                        )
                    )
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                    return

                metrics = result["metrics"]
                completion_tokens = int(
                    result.get("generated_token_count")
                    or metrics.get("num_generated_tokens", 0)
                )
                prompt_tokens = int(metrics.get("num_input_tokens", 0))
                server_state["last_completion"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_tps": metrics.get("generation_tps"),
                    "end_to_end_tps": metrics.get("end_to_end_tps"),
                    "avg_acceptance_length": metrics.get("avg_acceptance_length"),
                    "peak_memory_gb": metrics.get("peak_memory_gb"),
                    "target_path": result.get("target_path"),
                    "draft_path": result.get("draft_path"),
                    "updated_at": int(time.time()),
                }
                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result["text"].strip(),
                            },
                            "finish_reason": metrics.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "metrics": {
                        "generation_tps": metrics.get("generation_tps"),
                        "end_to_end_tps": metrics.get("end_to_end_tps"),
                        "avg_acceptance_length": metrics.get("avg_acceptance_length"),
                        "peak_memory_gb": metrics.get("peak_memory_gb"),
                    },
                }
                self._send_json(HTTPStatus.OK, response)

            def _write_sse(self, chunk: dict[str, Any]) -> None:
                self.wfile.write(b"data: ")
                self.wfile.write(json.dumps(chunk).encode("utf-8"))
                self.wfile.write(b"\n\n")
                self.wfile.flush()

            def _handle_stream(
                self,
                runtime: ModelRuntime,
                server_state: dict[str, Any],
                model_name: str,
                prompt: str,
                max_new_tokens: int,
                temperature: float,
                profile: bool,
            ) -> None:
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())

                def base_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
                    return {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }

                self.close_connection = True
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                stream = runtime.stream_generate(
                    GenerationRequest(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        profile=profile,
                    )
                )
                try:
                    self._write_sse(base_chunk({"role": "assistant"}))
                    done_event: dict[str, Any] | None = None
                    for event in stream:
                        etype = event.get("type")
                        if etype == "delta":
                            self._write_sse(base_chunk({"content": event["text"]}))
                        elif etype == "done":
                            done_event = event
                except (BrokenPipeError, ConnectionResetError):
                    stream.close()
                    return
                except Exception as exc:
                    stream.close()
                    err = base_chunk({}, finish_reason="stop")
                    err["error"] = {"message": str(exc)}
                    try:
                        self._write_sse(err)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except Exception:
                        pass
                    return

                if done_event is None:
                    return

                metrics = done_event["metrics"]
                completion_tokens = int(done_event.get("generated_token_count", 0))
                prompt_tokens = int(metrics.get("num_input_tokens", 0))
                server_state["last_completion"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_tps": metrics.get("generation_tps"),
                    "end_to_end_tps": metrics.get("end_to_end_tps"),
                    "avg_acceptance_length": metrics.get("avg_acceptance_length"),
                    "peak_memory_gb": metrics.get("peak_memory_gb"),
                    "target_path": done_event.get("target_path"),
                    "draft_path": done_event.get("draft_path"),
                    "updated_at": int(time.time()),
                }
                try:
                    final = base_chunk({}, finish_reason=metrics.get("finish_reason", "stop"))
                    final["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    self._write_sse(final)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)


def token_from_env(explicit: str | None = None) -> str | None:
    """Read the bearer token from env or the explicit arg. Empty = no auth."""
    if explicit is not None:
        return explicit or None
    return os.environ.get(SERVE_TOKEN_ENV) or None
