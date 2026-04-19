# Release Guide

This repo is designed to be publishable to GitHub, but the bundled model payload
must go through Git LFS.

## Pre-release checks

From the repo root:

```bash
cd ~/dev/models/micromodel-ship
./scripts/check-no-secrets.sh
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
git branch -M main
git remote add origin <your-github-url>
git push -u origin main
```

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

Release artifact:

- `dist/micromodel-ship-offline.tar.gz`

## Notes

- There is no single “model binary.” The shipped weights are a directory bundle.
- If GitHub LFS quotas are a concern, publish a code-only branch and keep the
  offline tarball in releases or private object storage.
