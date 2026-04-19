from __future__ import annotations

import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import (
    DEFAULT_HOST,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PORT,
    DEFAULT_TEMPERATURE,
)
from .runtime import GenerationRequest, ModelRuntime, build_chat_prompt_from_messages


class MicroModelServer:
    def __init__(
        self,
        runtime: ModelRuntime,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.model_name = model_name

    def serve_forever(self) -> None:
        runtime = self.runtime
        model_name = self.model_name
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

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "model": model_name,
                            "target": runtime.spec.target,
                            "draft": runtime.spec.draft,
                        },
                    )
                    return
                if self.path == "/metrics":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
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
                if self.path != "/v1/chat/completions":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(content_length)
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": f"invalid json: {exc}"}})
                    return

                if payload.get("stream"):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "streaming is not implemented"}})
                    return

                try:
                    prompt = build_chat_prompt_from_messages(payload.get("messages") or [])
                    max_new_tokens = int(payload.get("max_completion_tokens") or payload.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
                    temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
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

            def log_message(self, format: str, *args: Any) -> None:
                return

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"[micromodel-ship] serving {model_name} on http://{self.host}:{self.port}")
        print(f"[micromodel-ship] target={runtime.spec.target}")
        print(f"[micromodel-ship] draft={runtime.spec.draft}")
        httpd.serve_forever()
