#!/usr/bin/env python3
"""
bcg/apps/run.py
==========
Single command-line driver for BOTH belief-context-graph construction
backends. Pick one with the first positional argument:

  python bcg/apps/run.py light      --input data.json [...]
  python bcg/apps/run.py api_based  --input data.json [...]

Each subcommand mirrors the CLI of its original standalone project exactly
(same flags, same defaults) — only the dispatch is new. Run with
``light -h`` / ``api_based -h`` to see each backend's full option list.

Examples
--------
  # light backend: local embeddings + small generative model
  python bcg/apps/run.py light --input data.json --model-key gpt-5.5 --embedding-key embedding

  # api_based backend: one large API-based chat model does extraction + relations
  python bcg/apps/run.py api_based --input data.json \\
      --evidence-mode sentence --incremental-merge --incremental-merge-threshold 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bcg.apps.cli_help import RichArgumentParser
from bcg.apps.cli_options import add_run_options
from bcg.config.runtime import RuntimeConfig, resolve_runtime_config
from bcg.construct.dispatch import DEFAULT_BACKEND, split_backend_args


def _bootstrap_env() -> None:
    """Load the project root .env explicitly (import no longer does this)."""
    from bcg.core.env import load_project_env

    load_project_env()


def _add_common_args(p: argparse.ArgumentParser, runtime: RuntimeConfig) -> None:
    settings = runtime.settings
    p.add_argument("--input", "-i", required=True,
                   help="Input JSON/TXT (a trajectory, or multi-session QA items).")
    p.add_argument("--config", "-c", default=runtime.config_path,
                   help="Model config path (nested by model name; reserved key "
                        "'embedding' holds the embedding endpoint).")
    p.add_argument("--output-dir", "-o", default="outputs",
                   help="Output root; each item gets its own subdirectory.")
    p.add_argument("--model-key", default=settings.model_key,
                   help="Which chat-model entry of the config to use "
                        f"(default: {settings.model_key}).")
    p.add_argument("--embedding-key", default=settings.embedding_key,
                   help="Which config entry holds the embedding endpoint.")
    p.add_argument("--item", default=None,
                   help="Process only this item (id or 0-based index).")
    p.add_argument("--keep-order", default=False, action="store_true",
                   help="For multi-session inputs, do NOT sort sessions by date when "
                        "flattening; keep the input array order.")


def _run_light(argv: list[str]) -> None:
    from bcg.construct.light.pipeline import run_input

    runtime = resolve_runtime_config(argv)
    p = argparse.ArgumentParser(
        prog="bcg/apps/run.py light",
        description="construct_beliefs v3 streaming pipeline driver (light backend: "
                    "local embeddings + small generative model).",
    )
    _add_common_args(p, runtime)
    args = p.parse_args(argv)

    run_input(
        args.input, args.config, Path(args.output_dir),
        model_key=args.model_key, embedding_key=args.embedding_key,
        item_selector=args.item, keep_order=args.keep_order,
    )


def _run_api_based(argv: list[str]) -> None:
    from bcg.construct.api_based.pipeline import run_input
    from bcg.construct.api_based.stream import StreamOptions

    runtime = resolve_runtime_config(argv)
    p = argparse.ArgumentParser(
        prog="bcg/apps/run.py api_based",
        description="construct_beliefs v3 streaming pipeline driver (api_based backend: "
                    "one large API-based chat model).",
    )
    _add_common_args(p, runtime)

    add_run_options(p, runtime.settings.runner)

    args = p.parse_args(argv)

    options = StreamOptions(
        evidence_mode=args.evidence_mode,
        incremental_merge=args.incremental_merge,
        incremental_merge_threshold=args.incremental_merge_threshold,
        verify_merge=args.verify_merge,
        context_chars=args.context_chars,
        min_content_len=args.min_content_len,
    )

    run_input(
        args.input, args.config, Path(args.output_dir),
        model_key=args.model_key, embedding_key=args.embedding_key,
        options=options, item_selector=args.item, keep_order=args.keep_order,
    )


_BACKENDS = {"light": _run_light, "api_based": _run_api_based}


def main(argv: list[str] | None = None) -> None:
    _bootstrap_env()
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        parser = RichArgumentParser(
            prog="bcg construct run",
            description="construct_beliefs v3 streaming pipeline driver "
                        "(builds belief graphs from a trajectory or dataset).",
            epilog="Run 'bcg construct run <backend> --help' for a backend's "
                   "full option list. If omitted, the backend defaults to "
                   f"{DEFAULT_BACKEND!r} for compatibility.",
        )
        parser.add_argument(
            "backend",
            choices=list(_BACKENDS),
            nargs="?",
            default=DEFAULT_BACKEND,
            help=f"Which construct backend to use (default: {DEFAULT_BACKEND}).",
        )
        parser.print_help()
        raise SystemExit(0)

    try:
        runtime_argv = argv[1:] if argv and argv[0] in _BACKENDS else argv
        runtime = resolve_runtime_config(runtime_argv)
        backend, rest = split_backend_args(
            argv,
            backends=_BACKENDS,
            default=runtime.settings.backend,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    _BACKENDS[backend](rest)


if __name__ == "__main__":
    main()


__all__ = [
    "main",
]
