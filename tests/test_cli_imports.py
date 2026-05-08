from __future__ import annotations

import subprocess
import sys


def test_cli_import_does_not_initialize_mlx_runtime() -> None:
    script = """
import sys
import micromodel_ship.cli
for name in ("micromodel_ship.runtime", "micromodel_ship.server", "mlx.core"):
    if name in sys.modules:
        raise SystemExit(f"unexpected eager import: {name}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
