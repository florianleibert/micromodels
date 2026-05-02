#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="$(python3 - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as f:
    print(tomllib.load(f)["project"]["version"])
PY
)"

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

if [[ ! -x ".venv/bin/micromodel-ship" ]]; then
  cat >&2 <<'EOF'
ERROR: self-contained release artifact requires .venv/bin/micromodel-ship.

Run `uv sync --frozen` on the target release platform before packaging.
Hydra release artifacts must not require users to run uv after download.
EOF
  exit 1
fi

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
