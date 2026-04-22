"""Unix-socket integration tests. Same warming/auth contract as TCP."""

from __future__ import annotations

import json
import os
import secrets
import socket
import stat
import threading
import time
from pathlib import Path

import pytest

try:
    from micromodel_ship.server import MicroModelServer
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"mlx stack not importable: {exc}", allow_module_level=True)

from tests.test_server_integration import FakeRuntime


# macOS caps AF_UNIX paths at 104 bytes. pytest's tmp_path puts the socket
# under /private/var/folders/... which easily overflows. Put the socket in
# /tmp with a short random suffix so we stay under the limit.
def _short_socket_path() -> str:
    return f"/tmp/micromodel-test-{secrets.token_hex(4)}.sock"


@pytest.fixture
def sock_path():
    p = _short_socket_path()
    try:
        yield p
    finally:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def _uds_request(socket_path: str, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    """Send one HTTP request over a Unix socket and return (status, body, headers)."""
    req_lines = [f"{method} {path} HTTP/1.0", "Host: unix"]
    req_headers = dict(headers or {})
    if body:
        req_headers.setdefault("Content-Type", "application/json")
        req_headers["Content-Length"] = str(len(body))
    for k, v in req_headers.items():
        req_lines.append(f"{k}: {v}")
    req_lines.append("")
    req_lines.append("")
    raw = ("\r\n".join(req_lines)).encode("utf-8") + body

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(socket_path)
    sock.sendall(raw)
    chunks = []
    while True:
        try:
            c = sock.recv(4096)
        except socket.timeout:
            break
        if not c:
            break
        chunks.append(c)
    sock.close()

    data = b"".join(chunks)
    head, _, resp_body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status_line = lines[0]
    status = int(status_line.split(" ", 2)[1])
    hdrs: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            hdrs[k.strip().lower()] = v.strip()
    return status, resp_body, hdrs


def _start_uds_server(runtime: FakeRuntime, socket_path: str, token: str = "") -> tuple[MicroModelServer, threading.Thread]:
    server = MicroModelServer(
        runtime=runtime,
        unix_socket_path=socket_path,
        server_token=token,
    )
    server.start()
    thread = threading.Thread(target=server.wait, daemon=True)
    thread.start()
    # Wait briefly so the socket file is fully bound before the first probe.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if os.path.exists(socket_path):
            break
        time.sleep(0.01)
    return server, thread


def test_uds_socket_file_created_with_0600(sock_path: str) -> None:
    runtime = FakeRuntime()
    server, thread = _start_uds_server(runtime, sock_path)
    try:
        assert os.path.exists(sock_path), f"socket file not created at {sock_path}"
        mode = stat.S_IMODE(os.stat(sock_path).st_mode)
        assert mode == 0o600, f"expected mode 0o600, got {oct(mode)}"
    finally:
        server.stop()
        thread.join(timeout=2)


def test_uds_healthz_transitions(sock_path: str) -> None:
    runtime = FakeRuntime(warm_delay=0.2)
    server, thread = _start_uds_server(runtime, sock_path)
    try:
        status, body, _ = _uds_request(sock_path, "GET", "/healthz")
        assert status == 503
        assert json.loads(body)["status"] == "warming"
        deadline = time.time() + 3.0
        while time.time() < deadline:
            status, body, _ = _uds_request(sock_path, "GET", "/healthz")
            if status == 200:
                break
            time.sleep(0.05)
        assert status == 200
        assert json.loads(body)["status"] == "ready"
    finally:
        server.stop()
        thread.join(timeout=2)


def test_uds_auth_gated(sock_path: str) -> None:
    runtime = FakeRuntime()
    server, thread = _start_uds_server(runtime, sock_path, token="s3cret")
    try:
        time.sleep(0.1)
        status, body, _ = _uds_request(sock_path, "GET", "/metrics")
        assert status == 401
        status, body, _ = _uds_request(sock_path, "GET", "/metrics", headers={"Authorization": "Bearer s3cret"})
        assert status == 200
    finally:
        server.stop()
        thread.join(timeout=2)


def test_uds_socket_cleaned_up_on_stop(sock_path: str) -> None:
    runtime = FakeRuntime()
    server, thread = _start_uds_server(runtime, sock_path)
    assert os.path.exists(sock_path)
    server.stop()
    thread.join(timeout=2)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not os.path.exists(sock_path):
            break
        time.sleep(0.05)
    assert not os.path.exists(sock_path), f"socket file {sock_path} left behind after stop()"


def test_uds_stale_socket_cleaned_up(sock_path: str) -> None:
    """A leftover socket file from a prior run should be unlinked on bind."""
    with open(sock_path, "w") as f:
        f.write("stale")
    assert os.path.exists(sock_path)
    runtime = FakeRuntime()
    server, thread = _start_uds_server(runtime, sock_path)
    try:
        assert stat.S_ISSOCK(os.stat(sock_path).st_mode), "path should now be a socket"
    finally:
        server.stop()
        thread.join(timeout=2)
