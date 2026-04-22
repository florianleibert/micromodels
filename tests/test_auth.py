"""Tests for auth helpers. No MLX required."""

from __future__ import annotations

import pytest

from micromodel_ship.auth import check_bearer, parse_bind


class TestCheckBearer:
    def test_empty_token_allows_all(self) -> None:
        assert check_bearer("", None) is True
        assert check_bearer("", "") is True
        assert check_bearer("", "Bearer anything") is True

    def test_missing_header_when_token_set(self) -> None:
        assert check_bearer("secret", None) is False
        assert check_bearer("secret", "") is False

    def test_wrong_scheme(self) -> None:
        assert check_bearer("secret", "Basic secret") is False
        assert check_bearer("secret", "secret") is False
        assert check_bearer("secret", "bearer secret") is False  # case sensitive

    def test_wrong_token(self) -> None:
        assert check_bearer("secret", "Bearer wrong") is False
        assert check_bearer("secret", "Bearer ") is False
        assert check_bearer("secret", "Bearer secret-with-suffix") is False

    def test_correct_token(self) -> None:
        assert check_bearer("secret", "Bearer secret") is True
        assert check_bearer("x" * 64, "Bearer " + "x" * 64) is True


class TestParseBind:
    def test_valid(self) -> None:
        assert parse_bind("127.0.0.1:8051") == ("127.0.0.1", 8051)
        assert parse_bind("0.0.0.0:1") == ("0.0.0.0", 1)
        assert parse_bind("localhost:65535") == ("localhost", 65535)

    def test_missing_colon(self) -> None:
        with pytest.raises(ValueError, match="expected host:port"):
            parse_bind("127.0.0.1")

    def test_empty_host(self) -> None:
        with pytest.raises(ValueError, match="expected host:port"):
            parse_bind(":8051")

    def test_empty_port(self) -> None:
        with pytest.raises(ValueError, match="expected host:port"):
            parse_bind("127.0.0.1:")

    def test_non_numeric_port(self) -> None:
        with pytest.raises(ValueError):
            parse_bind("127.0.0.1:abc")

    def test_port_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="port out of range"):
            parse_bind("127.0.0.1:0")
        with pytest.raises(ValueError, match="port out of range"):
            parse_bind("127.0.0.1:65536")
