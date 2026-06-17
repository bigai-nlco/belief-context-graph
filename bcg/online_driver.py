#!/usr/bin/env python3
"""
scripts/online_driver.py
========================
Thin driver for the v3 streaming (online) interface.

It reads a stream of turn dicts — one compact JSON object per line (JSONL) —
from STDIN or a file, and feeds each to a SessionManager, which routes by
problem_id and maintains one live belief graph per trajectory. Per-turn and
final graphs, the raw stream log, and the reconstructed trajectory are written
under  <output-dir>/<problem_id>/  (see construct_beliefs.online for the full
list of artifacts).

Each input line is a dict:

    {"problem_id": "...", "role": "user", "content": "..."}
    {"problem_id": "...", "role": "assistant", "content": "<think>...</think>..."}
    {"problem_id": "...", "role": "assistant", "content": "...", "is_trajectory_end": true}

Optional per-line flags: "is_message_end" (default true) and
"is_trajectory_end" (default false). See construct_beliefs/online.py.

Examples
--------
  # pipe a JSONL stream straight from your generator:
  my_agent --stream | python scripts/online_driver.py --config model_config.json

  # replay a recorded stream file:
  python scripts/online_driver.py -i examples/research_stream_example.jsonl \
      --config model_config.json --model-key gpt-5.5 --output-dir outputs_stream

  # whole-sentence evidence + clustering + embedding-verified merge at finalize:
  python scripts/online_driver.py -i stream.jsonl --use-clustering \
      --merge-strategy embedding --merge-threshold 0.86
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construct_beliefs.online import SessionManager          # noqa: E402
from construct_beliefs.stream import StreamOptions           # noqa: E402


def iter_jsonl(stream: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Yield one dict per non-blank line; skip blanks, report bad lines."""
    for lineno, raw in enumerate(stream, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[warn] skipping malformed JSON on line {lineno}: {e}", file=sys.stderr)
            continue
        if not isinstance(obj, dict):
            print(f"[warn] skipping non-object on line {lineno}", file=sys.stderr)
            continue
        yield obj


def drive(manager: SessionManager, turns: Iterable[Dict[str, Any]],
          *, quiet: bool = False) -> Dict[str, Any]:
    """
    Push every turn into the manager. Returns a small run summary. This is the
    core loop, separated from CLI wiring so it can be unit-tested with an
    injected (e.g. fake-backed) manager.
    """
    n = 0
    finalized = []
    for turn in turns:
        snap = manager.push(turn)
        n += 1
        if not quiet:
            pid = snap.get("problem_id")
            stage = snap.get("stage")
            nb = snap.get("n_beliefs")
            print(f"  push #{n:>4}  problem={pid}  stage={stage:<8}  beliefs={nb}",
                  file=sys.stderr)
        if snap.get("finalized"):
            finalized.append(snap.get("problem_id"))
    # finalize any trajectory that never sent is_trajectory_end
    for pid in list(manager.active_problem_ids()):
        if not quiet:
            print(f"  [finalize-on-eof] problem={pid} (no is_trajectory_end seen)",
                  file=sys.stderr)
        manager.finalize(pid)
        finalized.append(pid)
    return {"n_turns_pushed": n,
            "problems": manager.all_problem_ids(),
            "finalized": finalized}


def build_manager(args) -> SessionManager:
    options = StreamOptions(
        evidence_mode=args.evidence_mode,
        use_clustering=args.use_clustering,
        cluster_threshold=args.cluster_threshold,
        cluster_min_sentences=args.cluster_min_sentences,
        cluster_buffer=args.cluster_buffer,
        merge_strategy=args.merge_strategy,
        merge_threshold=args.merge_threshold,
        context_chars=args.context_chars,
    )
    return SessionManager(
        config_path=args.config,
        model_key=args.model_key,
        embedding_key=args.embedding_key,
        output_root=Path(args.output_dir),
        options=options,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="construct_beliefs v3 streaming driver")
    p.add_argument("--input", "-i", default=None,
                   help="JSONL file of turn dicts (default: read STDIN).")
    p.add_argument("--config", "-c", default="model_config.json",
                   help="Model config path (same format as scripts/run.py).")
    p.add_argument("--output-dir", "-o", default="outputs_stream",
                   help="Output root; each problem_id gets its own subdirectory.")
    p.add_argument("--model-key", default=None,
                   help="Chat-model config entry (default: first non-reserved key).")
    p.add_argument("--embedding-key", default="embedding",
                   help="Config entry holding the embedding endpoint.")
    p.add_argument("--evidence-mode", choices=["sentence", "excerpt"], default="sentence")
    p.add_argument("--use-clustering", default=False, action="store_true")
    p.add_argument("--cluster-threshold", type=float, default=0.6)
    p.add_argument("--cluster-min-sentences", type=int, default=4)
    p.add_argument("--cluster-buffer", type=int, default=0)
    p.add_argument("--merge-strategy", choices=["embedding", "llm", "off"], default="embedding",
                   help="Trajectory-end (finalize) dedup strategy.")
    p.add_argument("--merge-threshold", type=float, default=0.86)
    p.add_argument("--context-chars", type=int, default=9000)
    p.add_argument("--quiet", "-q", default=False, action="store_true")
    args = p.parse_args()

    manager = build_manager(args)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            summary = drive(manager, iter_jsonl(f), quiet=args.quiet)
    else:
        summary = drive(manager, iter_jsonl(sys.stdin), quiet=args.quiet)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
