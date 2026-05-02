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
uv sync --frozen
./scripts/package.sh
```

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

## Notes

- There is no single “model binary.” The shipped weights are a directory bundle.
- The release tarball is platform-specific because it includes `.venv`.
- If GitHub LFS quotas are a concern, publish a code-only branch and keep the
  offline tarball in releases or private object storage.
