#!/usr/bin/env python3
"""
bcg/run.py
==========
Single command-line driver for BOTH belief-context-graph construction
backends. Pick one with the first positional argument:

  python bcg/run.py light      --input data.json [...]
  python bcg/run.py api_based  --input data.json [...]

Each subcommand mirrors the CLI of its original standalone project exactly
(same flags, same defaults) — only the dispatch is new. Run with
``light -h`` / ``api_based -h`` to see each backend's full option list.

Examples
--------
  # light backend: local embeddings + small generative model
  python bcg/run.py light --input data.json --model-key gpt-5.5 --embedding-key embedding

  # api_based backend: one large API-based chat model does extraction + relations
  python bcg/run.py api_based --input data.json \\
      --evidence-mode sentence --incremental-merge --incremental-merge-threshold 0.86
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python bcg/run.py ...` from the project root (the parent
# directory of this `bcg` package), matching both original projects' scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", required=True,
                   help="Input JSON/TXT (a trajectory, or multi-session QA items).")
    p.add_argument("--config", "-c", default="bcg/model_config.json",
                   help="Model config path (nested by model name; reserved key "
                        "'embedding' holds the embedding endpoint).")
    p.add_argument("--output-dir", "-o", default="outputs",
                   help="Output root; each item gets its own subdirectory.")
    p.add_argument("--model-key", default="gpt-5.5",
                   help="Which chat-model entry of the config to use "
                        "(default: gpt-5.5, matching the online server).")
    p.add_argument("--embedding-key", default="embedding",
                   help="Which config entry holds the embedding endpoint.")
    p.add_argument("--item", default=None,
                   help="Process only this item (id or 0-based index).")
    p.add_argument("--keep-order", default=False, action="store_true",
                   help="For multi-session inputs, do NOT sort sessions by date when "
                        "flattening; keep the input array order.")


def _run_light(argv: list[str]) -> None:
    from bcg.construct.light.pipeline import run_input

    p = argparse.ArgumentParser(
        prog="bcg/run.py light",
        description="construct_beliefs v3 streaming pipeline driver (light backend: "
                    "local embeddings + small generative model).",
    )
    _add_common_args(p)
    args = p.parse_args(argv)

    run_input(
        args.input, args.config, Path(args.output_dir),
        model_key=args.model_key, embedding_key=args.embedding_key,
        item_selector=args.item, keep_order=args.keep_order,
    )


def _run_api_based(argv: list[str]) -> None:
    from bcg.construct.api_based.pipeline import run_input
    from bcg.construct.api_based.stream import StreamOptions

    p = argparse.ArgumentParser(
        prog="bcg/run.py api_based",
        description="construct_beliefs v3 streaming pipeline driver (api_based backend: "
                    "one large API-based chat model).",
    )
    _add_common_args(p)

    # evidence mode (HP1)
    p.add_argument("--evidence-mode", choices=["sentence", "excerpt"], default="sentence",
                   help="'sentence' = evidence is always a complete sentence (split + indices); "
                        "'excerpt' = model quotes verbatim spans (may be fragments). Default: sentence.")

    # incremental per-turn merge (embedding-only, no LLM verification)
    p.add_argument("--incremental-merge", dest="incremental_merge",
                   default=True, action="store_true",
                   help="After each turn's new nodes/edges, run an embedding-only "
                        "merge (no LLM verification). Default: ON. Needs the embedding entry.")
    p.add_argument("--no-incremental-merge", dest="incremental_merge",
                   action="store_false",
                   help="Disable the per-turn incremental merge.")
    p.add_argument("--incremental-merge-threshold", type=float, default=0.86,
                   help="Cosine threshold for the per-turn incremental merge. Default 0.86 "
                        "(matching the online server).")
    p.add_argument("--verify-merge", dest="verify_merge",
                   default=True, action="store_true",
                   help="Add an LLM check to the per-turn incremental merge: for each "
                        "embedding-flagged candidate group, call the LLM once to verify the "
                        "merge is reasonable (apply-time gate) AND rewrite the surviving "
                        "node's content to cover all merged nodes' meaning. Default: ON. "
                        "Needs the embedding entry.")

    p.add_argument("--context-chars", type=int, default=100000,
                   help="Char budget of the existing-nodes context block. Default 100000 "
                        "(matching the online server).")
    p.add_argument("--min-content-len", type=int, default=0,
                   help="Skip turns whose content is shorter than this. Default 0.")

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
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print(f"Available backends: {', '.join(_BACKENDS)}")
        return

    backend, rest = argv[0], argv[1:]
    if backend not in _BACKENDS:
        print(f"error: unknown backend {backend!r}; choose one of: "
              f"{', '.join(_BACKENDS)}", file=sys.stderr)
        raise SystemExit(2)

    _BACKENDS[backend](rest)


if __name__ == "__main__":
    main()
