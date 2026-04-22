"""Auth helpers split out so they are importable without pulling in MLX."""

from __future__ import annotations

import hmac


def check_bearer(expected_token: str, header_value: str | None) -> bool:
    """Return True if ``header_value`` presents the expected bearer token.

    An empty ``expected_token`` disables auth and always returns True.
    Uses constant-time comparison to avoid timing side channels.
    """
    if not expected_token:
        return True
    if not header_value:
        return False
    prefix = "Bearer "
    if not header_value.startswith(prefix):
        return False
    presented = header_value[len(prefix):]
    return hmac.compare_digest(presented, expected_token)


def parse_bind(value: str) -> tuple[str, int]:
    """Parse ``host:port`` into (host, port). IPv6 bracket form not supported."""
    if ":" not in value:
        raise ValueError(f"expected host:port, got {value!r}")
    host, _, port_str = value.rpartition(":")
    if not host or not port_str:
        raise ValueError(f"expected host:port, got {value!r}")
    port = int(port_str)
    if port < 1 or port > 65535:
        raise ValueError(f"port out of range: {port}")
    return host, port
