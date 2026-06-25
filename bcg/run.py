#!/usr/bin/env python3
"""
scripts/run.py  (v3)
====================
Command-line driver for the streaming `construct_beliefs` pipeline.

No scenarios, no sessions. Any input (a trajectory, or multi-session QA data) is
normalised into items, each a flat list of role-tagged turns; each turn is
processed by role in ONE LLM call (new nodes + new forward edges).

Examples
--------
  # A trajectory or conversation file (auto-normalised), default options:
  python scripts/run.py --input data.json

  # Whole-sentence evidence (default) with topic clustering + embedding merge:
  python scripts/run.py --input data.json \
      --evidence-mode sentence --use-clustering \
      --merge-strategy embedding --merge-threshold 0.86

  # Free-span evidence (model quotes excerpts; no sentence splitting):
  python scripts/run.py --input data.json --evidence-mode excerpt

  # Pick the chat model / embedding entry from model_config.json:
  python scripts/run.py --input data.json \
      --model-key deepseek-v4-flash-260425 --embedding-key embedding
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belief_graph.pipeline import run_input
from belief_graph.stream import StreamOptions


def main():
    p = argparse.ArgumentParser(description="construct_beliefs v3 streaming pipeline driver")
    p.add_argument("--input", "-i", required=True,
                   help="Input JSON/TXT (a trajectory, or multi-session QA items).")
    p.add_argument("--config", "-c", default="bcg/model_config.json",
                   help="Model config path (nested by model name; reserved key "
                        "'embedding' holds the embedding endpoint).")
    p.add_argument("--output-dir", "-o", default="outputs_stream",
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

    # evidence mode (HP1)
    p.add_argument("--evidence-mode", choices=["sentence", "excerpt"], default="sentence",
                   help="'sentence' = evidence is always a complete sentence (split + indices); "
                        "'excerpt' = model quotes verbatim spans (may be fragments). Default: sentence.")
    # clustering (HP2) — sentence mode only; needs the embedding entry
    p.add_argument("--use-clustering", default=False, action="store_true",
                   help="Group a turn's sentences by topic cluster inside the single "
                        "extraction call (sentence mode only; requires the embedding entry).")
    p.add_argument("--cluster-threshold", type=float, default=0.6,
                   help="Cosine-similarity floor for merging sentence clusters. Default 0.6.")
    p.add_argument("--cluster-min-sentences", type=int, default=4,
                   help="Turns with fewer sentences skip clustering. Default 4.")
    p.add_argument("--cluster-buffer", type=int, default=0,
                   help="Neighbour window used ONLY for sentence embedding. Default 0.")

    # merge / dedup
    p.add_argument("--merge-strategy", choices=["embedding", "llm", "off"], default="embedding",
                   help="Trajectory-end dedup strategy. Default: embedding + LLM verification.")
    p.add_argument("--merge-threshold", type=float, default=0.86,
                   help="Cosine-similarity threshold for merge candidate pairs. Default 0.86.")

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

    p.add_argument("--context-chars", type=int, default=100000,
                   help="Char budget of the existing-nodes context block. Default 100000 "
                        "(matching the online server).")
    p.add_argument("--min-content-len", type=int, default=0,
                   help="Skip turns whose content is shorter than this. Default 0.")

    args = p.parse_args()

    options = StreamOptions(
        evidence_mode=args.evidence_mode,
        use_clustering=args.use_clustering,
        cluster_threshold=args.cluster_threshold,
        cluster_min_sentences=args.cluster_min_sentences,
        cluster_buffer=args.cluster_buffer,
        merge_strategy=args.merge_strategy,
        merge_threshold=args.merge_threshold,
        incremental_merge=args.incremental_merge,
        incremental_merge_threshold=args.incremental_merge_threshold,
        context_chars=args.context_chars,
        min_content_len=args.min_content_len,
    )

    run_input(
        args.input, args.config, Path(args.output_dir),
        model_key=args.model_key, embedding_key=args.embedding_key,
        options=options, item_selector=args.item, keep_order=args.keep_order,
    )


if __name__ == "__main__":
    main()
