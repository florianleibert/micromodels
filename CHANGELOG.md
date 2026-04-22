# Changelog

## 0.2.0 — 2026-04-22

Server-side half of the flocode local-model integration. Companion to
[florianleibert/openflo v0.1.29-flomac](https://github.com/florianleibert/openflo/releases/tag/v0.1.29-flomac).

### Added

- **Bind-before-warm `/healthz`** — HTTP socket binds immediately; the
  model loads in a background thread. `/healthz` returns
  `503 {"status":"warming"}` until ready, then `200 {"status":"ready"}`,
  and `503 {"status":"error"}` if warmup fails. Makes supervisors (like
  `flocode localmodel`) distinguish "still warming" from "dead".
- **Bearer-token auth** — set `FLOCODE_SERVE_TOKEN` or `--token` to
  require `Authorization: Bearer <token>` on `/v1/chat/completions` and
  `/metrics`. `/healthz` and `/v1/models` stay public so supervisors
  can probe without the secret. Constant-time compare.
- **Unix-socket transport** — `--unix-socket PATH` binds AF_UNIX
  (chmod 0600, stale-file cleanup, umask during bind). On single-user
  machines this replaces bearer auth because filesystem perms gate
  access. Mutually exclusive with `--bind`.
- **Plain-MLX backend** — new `PlainMLXRuntime` wraps `mlx_lm.generate`
  so models without a DFlash draft can be served through the same
  OpenAI-compatible API. `--model-id` picks between backends.
- **Model registry** — `micromodel_ship.registry` maps short model IDs
  to `(backend, target, draft, context, ...)`. Two entries:
  `qwen3-4b-dflash` (default, DFlash), `gemma-3n-e2b` (plain MLX).
- **`--no-hf-fallback`** — hard-fail if the bundled target/draft
  directories are missing instead of silently triggering an HF download.
- **`capabilities.json` manifest** — machine-readable description of
  the API surface + shipped models. Consumed by flocode. Dumped via
  the new `micromodel-ship capabilities` subcommand.
- **`LICENSE`** — Apache-2.0. The repo was previously license-less,
  which blocked redistribution.

### Tested

- 33 pytest cases covering auth helpers, bind parsing, capabilities
  shape, HTTP warming→ready state machine, Unix-socket lifecycle, and
  auth gating. No MLX weights required for the test suite.
- Live end-to-end on Apple M5 Max via flocode: cold start (8 GB HF
  download) in ~2 min; warm start in ~5 s; first completion ~2 s.

### Changed

- `serve_forever()` split into composable `start()` + `wait()` + `stop()`
  lifecycle methods. The public single-call behavior is unchanged; the
  new methods exist so supervisors and tests can control lifecycle.

## 0.1.0 — 2026-04-21

Initial public release. See `git log v0.1.0` (no tag) for the pre-0.2 history.
