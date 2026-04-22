from __future__ import annotations

import argparse
import json

from .auth import parse_bind
from .config import (
    CAPABILITIES_PATH,
    DEFAULT_DRAFT_REPO,
    DEFAULT_HOST,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PORT,
    DEFAULT_SEED,
    DEFAULT_TARGET_REPO,
    DEFAULT_TEMPERATURE,
    DEFAULT_VERIFY_CHUNK_SIZE,
    DEFAULT_VERIFY_MODE,
    bundled_model_spec,
)
from . import registry
from .plain_mlx import PlainMLXRuntime
from .runtime import GenerationRequest, ModelRuntime, prefetch_models
from .server import MicroModelServer, token_from_env


def _make_runtime(
    model_id: str | None,
    target_override: str | None,
    draft_override: str | None,
    seed: int,
):
    """Dispatch to the DFlash or plain-MLX runtime based on the model registry.

    ``model_id`` picks an entry in micromodel_ship.registry; target/draft
    overrides still win so a user can point either backend at a local
    directory without touching the registry. Returns a runtime object whose
    public interface (warm/generate/stream_generate/spec) server.py calls.
    """
    if model_id is None:
        model_id = registry.DEFAULT_MODEL_ID
    entry = registry.get(model_id)
    if entry.backend == "dflash":
        return ModelRuntime(
            target_override or entry.target,
            draft_override or entry.draft or None,
            seed=seed,
        )
    if entry.backend == "plain_mlx":
        return PlainMLXRuntime(
            target_override or entry.target,
            seed=seed,
        )
    raise ValueError(f"registry: unknown backend {entry.backend!r} for model {entry.id!r}")


def _parse_bind(value: str) -> tuple[str, int]:
    try:
        return parse_bind(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shippable local Qwen3-4B DFlash runtime.")
    sub = parser.add_subparsers(dest="command", required=True)

    prefetch = sub.add_parser("prefetch", help="Download the target and draft into local models/ directories.")
    prefetch.add_argument("--target-repo", default=None)
    prefetch.add_argument("--draft-repo", default=None)

    paths = sub.add_parser("paths", help="Show which target and draft paths would be used.")
    paths.add_argument("--target-model", default=None)
    paths.add_argument("--draft-model", default=None)

    sub.add_parser("capabilities", help="Print the capabilities.json manifest.")

    run = sub.add_parser("run", help="Run one prompt locally.")
    run.add_argument("--prompt", required=True)
    run.add_argument("--target-model", default=None)
    run.add_argument("--draft-model", default=None)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run.add_argument("--verify-mode", default=DEFAULT_VERIFY_MODE)
    run.add_argument("--verify-chunk-size", type=int, default=DEFAULT_VERIFY_CHUNK_SIZE)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--profile", action="store_true")
    run.add_argument("--json", action="store_true")
    run.add_argument(
        "--no-hf-fallback",
        action="store_true",
        help="Fail if bundled target/draft directories are missing instead of pulling from Hugging Face.",
    )

    chat = sub.add_parser("chat", help="Interactive chat loop.")
    chat.add_argument("--target-model", default=None)
    chat.add_argument("--draft-model", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    chat.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    chat.add_argument("--seed", type=int, default=DEFAULT_SEED)
    chat.add_argument("--show-stats", action="store_true")
    chat.add_argument("--max-turns", type=int, default=6)

    serve = sub.add_parser("serve", help="Serve a minimal OpenAI-compatible chat endpoint.")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument(
        "--bind",
        type=_parse_bind,
        default=None,
        help="Shorthand for --host/--port, e.g. 127.0.0.1:8051. Overrides --host/--port.",
    )
    serve.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    serve.add_argument(
        "--model-id",
        default=None,
        help=(
            "Entry in the built-in model registry (qwen3-4b-dflash | gemma-3n-e2b). "
            "Defaults to " + registry.DEFAULT_MODEL_ID + ". Picks the backend and default weights."
        ),
    )
    serve.add_argument("--target-model", default=None)
    serve.add_argument("--draft-model", default=None)
    serve.add_argument("--seed", type=int, default=DEFAULT_SEED)
    serve.add_argument(
        "--token",
        default=None,
        help=(
            "Bearer token required on /v1/chat/completions and /metrics. "
            "If omitted, falls back to $FLOCODE_SERVE_TOKEN. Empty disables auth."
        ),
    )
    serve.add_argument(
        "--no-hf-fallback",
        action="store_true",
        help="Fail if bundled target/draft directories are missing instead of pulling from Hugging Face.",
    )
    serve.add_argument(
        "--unix-socket",
        default=None,
        help=(
            "Bind an AF_UNIX socket at PATH (mutually exclusive with --bind/--host/--port). "
            "File permissions are set to 0600 so only the same user can connect; "
            "bearer auth becomes optional on single-user machines."
        ),
    )

    return parser


def _ensure_local_paths(target_override: str | None, draft_override: str | None) -> None:
    spec = bundled_model_spec(target_override, draft_override)
    if not spec.target_is_local:
        raise SystemExit(
            f"refusing to fall back to Hugging Face: target not local ({spec.target}). "
            f"Run `micromodel-ship prefetch` or pass explicit --target-model / --draft-model."
        )
    if not spec.draft_is_local:
        raise SystemExit(
            f"refusing to fall back to Hugging Face: draft not local ({spec.draft}). "
            f"Run `micromodel-ship prefetch` or pass explicit --target-model / --draft-model."
        )


def cmd_prefetch(args: argparse.Namespace) -> int:
    paths = prefetch_models(
        target_repo=args.target_repo or DEFAULT_TARGET_REPO,
        draft_repo=args.draft_repo or DEFAULT_DRAFT_REPO,
    )
    print(json.dumps(paths, indent=2))
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    spec = bundled_model_spec(args.target_model, args.draft_model)
    print(json.dumps({
        "target": spec.target,
        "draft": spec.draft,
        "target_is_local": spec.target_is_local,
        "draft_is_local": spec.draft_is_local,
    }, indent=2))
    return 0


def cmd_capabilities(_args: argparse.Namespace) -> int:
    if not CAPABILITIES_PATH.exists():
        raise SystemExit(f"capabilities.json not found at {CAPABILITIES_PATH}")
    print(CAPABILITIES_PATH.read_text(encoding="utf-8"), end="")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.no_hf_fallback:
        _ensure_local_paths(args.target_model, args.draft_model)
    runtime = ModelRuntime(args.target_model, args.draft_model, seed=args.seed)
    result = runtime.generate(
        GenerationRequest(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
            profile=args.profile,
        )
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["text"].strip())
        print()
        print(json.dumps(result["metrics"], indent=2))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    runtime = ModelRuntime(args.target_model, args.draft_model, seed=args.seed)
    history: list[tuple[str, str]] = []
    print("Type a message. Use /exit or Ctrl-D to quit.")
    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            break
        if user_message == "/clear":
            history.clear()
            print("[cleared]")
            continue

        lines = []
        if history:
            lines.append("Continue this conversation and answer the latest user message.")
            for old_user, old_assistant in history[-max(1, args.max_turns):]:
                lines.append(f"User: {old_user}")
                lines.append(f"Assistant: {old_assistant}")
        lines.append(f"User: {user_message}")
        lines.append("Assistant:")
        prompt = "\n".join(lines)
        result = runtime.generate(
            GenerationRequest(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )
        answer = result["text"].strip()
        print(f"assistant> {answer}\n")
        if args.show_stats:
            metrics = result["metrics"]
            print(
                "[stats] "
                f"gen_tps={metrics.get('generation_tps'):.2f} "
                f"e2e_tps={metrics.get('end_to_end_tps'):.2f} "
                f"accept={metrics.get('avg_acceptance_length'):.2f}\n"
            )
        history.append((user_message, answer))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    # --no-hf-fallback only applies to the DFlash path (which has a bundled
    # models/target, models/draft convention). Plain-MLX models always pull
    # from HF or an explicit --target-model path.
    model_id = args.model_id or registry.DEFAULT_MODEL_ID
    entry = registry.get(model_id)
    if args.no_hf_fallback and entry.backend == "dflash":
        _ensure_local_paths(args.target_model, args.draft_model)
    host = args.host
    port = args.port
    if args.bind is not None:
        host, port = args.bind
    if args.unix_socket and args.bind is not None:
        raise SystemExit("--unix-socket and --bind/--host/--port are mutually exclusive")
    runtime = _make_runtime(
        model_id=model_id,
        target_override=args.target_model,
        draft_override=args.draft_model,
        seed=args.seed,
    )
    model_name = args.model_name
    if model_name == DEFAULT_MODEL_NAME and model_id != registry.DEFAULT_MODEL_ID:
        # Surface the actual served model in /v1/models rather than always
        # reporting "micromodel-qwen3-4b-dflash".
        model_name = f"micromodel-{entry.id}"
    server = MicroModelServer(
        runtime=runtime,
        host=host,
        port=port,
        model_name=model_name,
        server_token=token_from_env(args.token),
        unix_socket_path=args.unix_socket,
    )
    server.serve_forever()
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prefetch":
        raise SystemExit(cmd_prefetch(args))
    if args.command == "paths":
        raise SystemExit(cmd_paths(args))
    if args.command == "capabilities":
        raise SystemExit(cmd_capabilities(args))
    if args.command == "run":
        raise SystemExit(cmd_run(args))
    if args.command == "chat":
        raise SystemExit(cmd_chat(args))
    if args.command == "serve":
        raise SystemExit(cmd_serve(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
