#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE' >&2
Usage:
  scripts/package.sh [version-or-tag]

Build a self-contained, platform-specific micromodel-ship archive plus
checksums.txt and release-manifest.json. The optional version may be a
SemVer-like release tag such as v0.2.1-alpha.2. When omitted, the project
version from pyproject.toml is used.

Environment:
  MICROMODEL_RELEASE_PLATFORM  Override platform, default: <os>-<arch>.
  MICROMODEL_BUILD_DATE        RFC3339 timestamp for release-manifest.json.
  MICROMODEL_BUNDLED_PYTHON    Python prefix to copy into the archive,
                               for example an `uv python install` directory.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  usage
  exit 1
fi

if [[ "$#" -eq 1 ]]; then
  VERSION="${1#v}"
else
  VERSION="$(python3 - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as f:
    print(tomllib.load(f)["project"]["version"])
PY
)"
fi
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.+][0-9A-Za-z._-]+)?$ ]]; then
  echo "ERROR: invalid release version '$VERSION'." >&2
  echo "       expected SemVer-like, e.g. 0.2.1 or 0.2.1-alpha.2" >&2
  exit 1
fi

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$arch" in
  arm64|aarch64) arch="arm64" ;;
  x86_64|amd64) arch="amd64" ;;
esac
platform="${MICROMODEL_RELEASE_PLATFORM:-${os}-${arch}}"
if ! [[ "$platform" =~ ^[a-z0-9]+-[a-z0-9]+$ ]]; then
  echo "ERROR: invalid release platform '$platform'." >&2
  exit 1
fi

COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
BUILT_AT="${MICROMODEL_BUILD_DATE:-$(date -u +"%Y-%m-%dT%H:%M:%SZ")}"
RUNTIME_DIR=".micromodel-runtime"
BUNDLED_PYTHON_DIR="$RUNTIME_DIR/python"

if [[ ! -x ".venv/bin/micromodel-ship" ]]; then
  cat >&2 <<'EOF'
ERROR: self-contained release artifact requires .venv/bin/micromodel-ship.

Run `uv sync --frozen` on the target release platform before packaging.
Hydra release artifacts must not require users to run uv after download.
EOF
  exit 1
fi

if [[ -n "${MICROMODEL_BUNDLED_PYTHON:-}" ]]; then
  if [[ ! -d "$MICROMODEL_BUNDLED_PYTHON" ]]; then
    echo "ERROR: MICROMODEL_BUNDLED_PYTHON does not exist: $MICROMODEL_BUNDLED_PYTHON" >&2
    exit 1
  fi
  rm -rf "$BUNDLED_PYTHON_DIR"
  mkdir -p "$BUNDLED_PYTHON_DIR"
  cp -a "$MICROMODEL_BUNDLED_PYTHON"/. "$BUNDLED_PYTHON_DIR"/
fi

bundled_python=""
for candidate in "$BUNDLED_PYTHON_DIR/bin/python3" "$BUNDLED_PYTHON_DIR/bin/python" "$BUNDLED_PYTHON_DIR"/bin/python3.*; do
  if [[ -x "$candidate" ]]; then
    bundled_python="$candidate"
    break
  fi
done
if [[ -z "$bundled_python" ]]; then
  cat >&2 <<'EOF'
ERROR: self-contained release artifact requires a bundled Python runtime.

Set MICROMODEL_BUNDLED_PYTHON to a prepared Python prefix before packaging,
for example:

  uv python install 3.11
  PYTHON_BIN="$(uv python find 3.11)"
  PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
  MICROMODEL_BUNDLED_PYTHON="$PYTHON_PREFIX" scripts/build-release.sh v0.2.1-alpha.2

Hydra alpha testers must not need Homebrew Python or uv after extraction.
EOF
  exit 1
fi

python_version="$("$bundled_python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
site_packages=".venv/lib/python${python_version}/site-packages"
if [[ ! -d "$site_packages" ]]; then
  cat >&2 <<EOF
ERROR: .venv was not built for bundled Python $python_version.

Expected site-packages at:
  $site_packages

Rebuild the environment with the same Python used for MICROMODEL_BUNDLED_PYTHON,
then run scripts/package.sh again.
EOF
  exit 1
fi

cat > ".venv/bin/micromodel-ship" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd)"
PY="\$ROOT/$BUNDLED_PYTHON_DIR/bin/python3"
if [[ ! -x "\$PY" ]]; then
  PY="\$ROOT/$BUNDLED_PYTHON_DIR/bin/python"
fi
if [[ ! -x "\$PY" ]]; then
  for candidate in "\$ROOT/$BUNDLED_PYTHON_DIR"/bin/python3.*; do
    if [[ -x "\$candidate" ]]; then
      PY="\$candidate"
      break
    fi
  done
fi
if [[ ! -x "\$PY" ]]; then
  echo "bundled Python runtime missing under \$ROOT/$BUNDLED_PYTHON_DIR" >&2
  exit 127
fi
export PYTHONHOME="\$ROOT/$BUNDLED_PYTHON_DIR"
export PYTHONPATH="\$ROOT/$site_packages:\$ROOT"
cd "\$ROOT"
exec "\$PY" -m micromodel_ship.cli "\$@"
EOF
chmod +x ".venv/bin/micromodel-ship"

if [[ ! -d "models" ]]; then
  echo "ERROR: self-contained release artifact requires bundled models/ payload." >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256_hex() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_hex() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "ERROR: neither sha256sum nor shasum is on PATH." >&2
  exit 1
fi

mkdir -p dist
archive="dist/micromodel-ship-${VERSION}-${platform}.tar.gz"
legacy="dist/micromodel-ship-offline.tar.gz"

tar -czf "$archive" \
  --exclude='.git' \
  --exclude='dist' \
  -C "$(dirname "$ROOT_DIR")" \
  "$(basename "$ROOT_DIR")"
cp "$archive" "$legacy"

checksum_file="dist/checksums.txt"
: > "$checksum_file"
for file in "$archive" "$legacy"; do
  digest="$(sha256_hex "$file")"
  printf 'sha256:%s  %s\n' "$digest" "$(basename "$file")" >> "$checksum_file"
done

archive_digest="$(sha256_hex "$archive")"
archive_size="$(wc -c < "$archive" | tr -d ' ')"
manifest="dist/release-manifest.json"
cat > "$manifest" <<EOF
{
  "schema_version": 1,
  "component": "micromodel-ship",
  "version": "$VERSION",
  "commit": "$COMMIT",
  "repository": "florianleibert/micromodels",
  "built_at": "$BUILT_AT",
  "artifacts": [
    {
      "platform": "$platform",
      "filename": "$(basename "$archive")",
      "sha256": "sha256:$archive_digest",
      "size": $archive_size,
      "content_type": "application/gzip"
    }
  ]
}
EOF

echo "wrote $archive"
echo "wrote $legacy"
echo "wrote $checksum_file"
echo "wrote $manifest"
