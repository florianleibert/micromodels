#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE' >&2
Usage:
  scripts/build-release.sh <version-or-tag>

Build the micromodel-ship managed-runtime release artifact for the current
platform. This is intentionally a prepared-machine flow: the archive must
contain .venv/ and models/, so run it on the platform you intend Hydra to
install.

Examples:
  uv python install 3.11
  PYTHON_BIN="$(uv python find 3.11)"
  PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
  UV_PROJECT_ENVIRONMENT=.venv uv sync --frozen --python "$PYTHON_BIN"
  ./scripts/prefetch.sh
  MICROMODEL_BUNDLED_PYTHON="$PYTHON_PREFIX" scripts/build-release.sh v0.2.1-alpha.2
USAGE
}

if [[ "$#" -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && exit 0
  exit 1
fi

VERSION="${1#v}"
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.+][0-9A-Za-z._-]+)?$ ]]; then
  echo "ERROR: invalid release version '$1'." >&2
  echo "       expected SemVer-like, e.g. v0.2.1-alpha.2" >&2
  exit 1
fi

./scripts/check-no-secrets.sh
./scripts/package.sh "$VERSION"

echo
echo "Built micromodel-ship release assets:"
sed 's/^/  /' dist/checksums.txt
echo "  dist/release-manifest.json"
