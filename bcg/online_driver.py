#!/usr/bin/env python3
"""Replay JSONL turns through either construction backend."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcg.cli_help import RichArgumentParser  # noqa: E402
from bcg.construct.dispatch import DEFAULT_BACKEND, split_backend_args  # noqa: E402


def iter_jsonl(stream: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from non-blank lines, reporting malformed input."""

    for lineno, raw in enumerate(stream, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"[warn] skipping malformed JSON on line {lineno}: {exc}",
                file=sys.stderr,
            )
            continue
        if not isinstance(obj, dict):
            print(
                f"[warn] skipping non-object on line {lineno}",
                file=sys.stderr,
            )
            continue
        yield obj


def drive(
    manager: Any,
    turns: Iterable[dict[str, Any]],
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Push all turns and finalize trajectories left open at end-of-file."""

    count = 0
    finalized: list[str] = []
    for turn in turns:
        snapshot = manager.push(turn)
        count += 1
        if not quiet:
            print(
                f"  push #{count:>4}  problem={snapshot.get('problem_id')}  "
                f"stage={snapshot.get('stage')}  "
                f"beliefs={snapshot.get('n_beliefs')}",
                file=sys.stderr,
            )
        if snapshot.get("finalized"):
            finalized.append(snapshot.get("problem_id"))

    for problem_id in list(manager.active_problem_ids()):
        if not quiet:
            print(
                f"  [finalize-on-eof] problem={problem_id} (no is_trajectory_end seen)",
                file=sys.stderr,
            )
        manager.finalize(problem_id)
        finalized.append(problem_id)

    return {
        "n_turns_pushed": count,
        "problems": manager.all_problem_ids(),
        "finalized": finalized,
    }


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="JSONL file of turn objects (default: read standard input).",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="bcg/model_config.json",
        help="Shared model configuration path.",
    )
    parser.add_argument("--output-dir", "-o", default="outputs_stream")
    parser.add_argument("--model-key", default="gpt-5.5")
    parser.add_argument("--embedding-key", default="embedding")
    parser.add_argument("--quiet", "-q", action="store_true")


def _run_light(argv: list[str]) -> None:
    from bcg.construct.light.online import SessionManager

    parser = argparse.ArgumentParser(
        prog="bcg construct replay light",
        description="Replay JSONL turns through the light construction backend.",
    )
    _add_common_args(parser)
    args = parser.parse_args(argv)
    manager = SessionManager(
        config_path=args.config,
        model_key=args.model_key,
        embedding_key=args.embedding_key,
        output_root=Path(args.output_dir),
    )
    _run_stream(manager, args)


def _run_api_based(argv: list[str]) -> None:
    from bcg.construct.api_based.online import SessionManager
    from bcg.construct.api_based.stream import StreamOptions

    parser = argparse.ArgumentParser(
        prog="bcg construct replay api_based",
        description="Replay JSONL turns through the API-based construction backend.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--evidence-mode",
        choices=["sentence", "excerpt"],
        default="sentence",
    )
    parser.add_argument(
        "--incremental-merge",
        dest="incremental_merge",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-incremental-merge",
        dest="incremental_merge",
        action="store_false",
    )
    parser.add_argument("--incremental-merge-threshold", type=float, default=0.86)
    parser.add_argument(
        "--verify-merge",
        dest="verify_merge",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-verify-merge",
        dest="verify_merge",
        action="store_false",
    )
    parser.add_argument("--context-chars", type=int, default=100000)
    parser.add_argument("--min-content-len", type=int, default=0)
    args = parser.parse_args(argv)

    options = StreamOptions(
        evidence_mode=args.evidence_mode,
        incremental_merge=args.incremental_merge,
        incremental_merge_threshold=args.incremental_merge_threshold,
        verify_merge=args.verify_merge,
        context_chars=args.context_chars,
        min_content_len=args.min_content_len,
    )
    manager = SessionManager(
        config_path=args.config,
        model_key=args.model_key,
        embedding_key=args.embedding_key,
        output_root=Path(args.output_dir),
        options=options,
    )
    _run_stream(manager, args)


def _run_stream(manager: Any, args: argparse.Namespace) -> None:
    if args.input:
        with open(args.input, encoding="utf-8") as stream:
            summary = drive(manager, iter_jsonl(stream), quiet=args.quiet)
    else:
        summary = drive(manager, iter_jsonl(sys.stdin), quiet=args.quiet)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


_BACKENDS = {"light": _run_light, "api_based": _run_api_based}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        parser = RichArgumentParser(
            prog="bcg construct replay",
            description="Replay JSONL turns through a belief-graph backend.",
            epilog="If omitted, the backend defaults to "
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
        backend, rest = split_backend_args(args, backends=_BACKENDS)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    _BACKENDS[backend](rest)


if __name__ == "__main__":
    main()
