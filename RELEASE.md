# Release Guide

This repo publishes a self-contained, platform-specific runtime archive for
Hydra-managed installs. The archive must contain the Python environment,
`micromodel-ship` console script, and model payload. Hydra users must not need
`uv` or source checkout access after download.

## Pre-release checks

From the repo root:

```bash
cd ~/dev/models/micromodel-ship
./scripts/check-no-secrets.sh
uv python install 3.11
PYTHON_BIN="$(uv python find 3.11)"
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
UV_PROJECT_ENVIRONMENT=.venv uv sync --frozen \
  --python "$PYTHON_BIN"
```

For a train release, pass the exact release tag/version so the archive name and
`release-manifest.json` line up with the GitHub Release:

```bash
./scripts/prefetch.sh
MICROMODEL_BUNDLED_PYTHON="$PYTHON_PREFIX" \
  ./scripts/build-release.sh v0.2.1-alpha.2
```

`MICROMODEL_BUNDLED_PYTHON` is required. The package script copies that Python
prefix into `.micromodel-runtime/python` inside the archive and rewrites
`.venv/bin/micromodel-ship` to a relative launcher. This is the property Hydra
Desktop depends on: after extraction on a clean Mac, the runtime starts without
Homebrew Python or `uv`.

Optional local sanity flow:

```bash
uv sync
./scripts/paths.sh
./scripts/run.sh "Explain speculative decoding simply."
./scripts/serve.sh
```

## Git LFS setup

Install and initialize Git LFS once:

```bash
git lfs install
```

This repo already contains:

```text
models/** filter=lfs diff=lfs merge=lfs -text
dist/*.tar.gz filter=lfs diff=lfs merge=lfs -text
```

in `.gitattributes`.

## Publish steps

```bash
cd ~/dev/models/micromodel-ship
git init
git add .gitattributes
git add .
git commit -m "Initial micromodel-ship release"
git branch -M master
git remote add origin <your-github-url>
git push -u origin master
```

> The shipped repo uses `master` as the default branch; substitute `main` above
> if your fork uses the newer convention.

## What is actually being published

Code:

- CLI
- runtime wrapper
- local server
- scripts
- docs

Bundled model payload:

- `models/target/`
- `models/draft/`

Release artifacts:

- `dist/micromodel-ship-<version>-<platform>.tar.gz`
- `dist/micromodel-ship-offline.tar.gz` compatibility copy
- `dist/checksums.txt`
- `dist/release-manifest.json`

`release-manifest.json` is consumed by the OpenFlo suite-manifest assembler. It
records the versioned platform archive, source commit, SHA-256, size, content
type, and build timestamp. The legacy compatibility archive remains in
`checksums.txt` but is not listed as a suite artifact because the suite lockfile
expects one active artifact per component platform.

## GitHub Release workflow

The workflow in `.github/workflows/release.yml` is intentionally bound to a
self-hosted `macOS`, `ARM64`, `micromodel-release` runner. The managed-runtime
artifact includes `.venv/` and model weights, so a generic hosted Linux runner
would produce the wrong platform and an incomplete artifact. Until a prepared
runner is available, operators can run `scripts/build-release.sh <tag>` locally
and attach `dist/micromodel-ship-*.tar.gz`, `dist/checksums.txt`, and
`dist/release-manifest.json` to the release by hand.

## Notes

- There is no single “model binary.” The shipped weights are a directory bundle.
- The release tarball is platform-specific because it includes `.venv`, model
  weights, and a bundled Python runtime.
- If GitHub LFS quotas are a concern, publish a code-only branch and keep the
  offline tarball in releases or private object storage.
