"""
pipeline.py  (v3)
=================
Thin batch drivers over the streaming engine (stream.py). No scenarios, no
sessions: every input is normalised by loaders.iter_items into items, each a
flat list of role-tagged turns, and each item is built into one belief graph.

    run_item(...)   — one item: ingest turns in order, applying semantic
                      chunking, local summarisation, incremental merge and
                      Qwen relation generation, then finalize.
    run_input(...)  — load a file, wire client/embedder from config, process
                      all selected items into <out_dir>/<item_id>/.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

from .._shared.loaders import iter_items, load_input_file, sanitize_name, select_items
from .llm import (
    USAGE,
    load_belief_graph_config,
    load_config,
    load_embedding_config,
    make_client,
    make_embedder,
)
from .stream import StreamingBeliefBuilder, StreamOptions


def run_item(
    item: dict[str, Any],
    *,
    client,
    model: str,
    out_dir: Path,
    options: StreamOptions,
    embedder=None,
    edge_generator=None,
    max_tokens: int | None = None,
    pricing: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the belief graph for ONE normalised item."""
    builder = StreamingBeliefBuilder(
        client=client, model=model, item_id=item["item_id"],
        item_meta=item.get("meta"), out_dir=out_dir, options=options,
        embedder=embedder, edge_generator=edge_generator,
        max_tokens=max_tokens,
    )
    for turn in item["turns"]:
        builder.ingest_turn(turn["role"], turn["content"],
                            date=turn.get("date"), has_answer=turn.get("has_answer"))
    meta: dict[str, Any] = {"order_sorted": item.get("order_sorted", False)}
    if extra_meta:
        meta.update(extra_meta)
    return builder.finalize(extra_meta=meta, pricing=pricing)


def run_input(
    input_path: str,
    config_path: str,
    output_dir: Path,
    *,
    model_key: str | None = None,
    embedding_key: str = "embedding",
    options: StreamOptions | None = None,
    item_selector: str | None = None,
    keep_order: bool = False,
) -> None:
    options = options or StreamOptions()
    data = load_input_file(input_path)

    cfg = load_config(config_path, model_key=model_key)
    bg_cfg = load_belief_graph_config(config_path, model_key=model_key)
    if bg_cfg:
        options = copy.deepcopy(options)
        options.apply_belief_graph_config(bg_cfg)
    # else: no belief_graph section supplied — keep the caller-provided
    # StreamOptions() defaults as-is. Do NOT call
    # apply_belief_graph_config({}) here: an empty/missing config is a
    # normal, documented case (see model_config.example.json), and that
    # call raises ValueError("belief_graph.runtime must be an object")
    # because {} has no "runtime" key.
    client = make_client(cfg)
    model = cfg["model"]
    max_tokens = cfg.get("max_tokens")
    pricing = cfg.get("pricing")

    masked = (cfg.get("api_key", "") or "")
    masked = (masked[:6] + "…" + masked[-3:]) if len(masked) > 10 else "***"
    print(f"[info] model={model}  base_url={cfg['base_url']}  api_key={masked}"
          + (f"  max_tokens={max_tokens}" if max_tokens else ""))
    extractor_cfg = options.to_dict()["extractor"]
    if extractor_cfg.get("enabled", True):
        print(
            f"[info] extractor={extractor_cfg['model']}  "
            f"base_url={extractor_cfg['base_url']}  "
            f"max_concurrency={extractor_cfg['max_concurrency']}  "
            f"context_scope={extractor_cfg['context_scope']}"
        )
    else:
        print("[warn] generative extractor disabled; turns will produce no nodes", file=sys.stderr)
    entity_cfg = options.to_dict()["entities"]
    print(
        f"[info] entities method={entity_cfg['method']}  "
        f"spaCy={entity_cfg['spacy_model']}  "
        "stage=post-merge"
    )
    edge_cfg = options.to_dict()["edge_generation"]
    print(
        f"[info] edge_generator={edge_cfg['model']}  "
        f"base_url={edge_cfg['base_url']}  non_thinking={not edge_cfg['enable_thinking']}"
    )
    stance_cfg = options.to_dict()["stance"]
    print(
        f"[info] stance_model={stance_cfg['model_path']}  "
        f"device={stance_cfg['device']}  labels={','.join(stance_cfg['labels'])} "
        "(weights load lazily on first use)"
    )

    emb_cfg = load_embedding_config(config_path, embedding_key=embedding_key)
    embedder = None
    if emb_cfg is not None:
        embedder = make_embedder(emb_cfg)
        if emb_cfg.get("provider") == "local":
            print(f"[info] embedding provider=local  model={emb_cfg['model']}  "
                  f"(weights load lazily on first use)")
        else:
            print(f"[info] embedding model={emb_cfg['model']}  base_url={emb_cfg['base_url']}")
    else:
        if options.chunking_enabled:
            print(
                f"[warn] semantic chunking needs an {embedding_key!r} entry in "
                f"{config_path}; each turn will fall back to one chunk",
                file=sys.stderr,
            )
        if options.incremental_merge:
            print(
                f"[warn] incremental merge needs an {embedding_key!r} entry; "
                "incremental merge passes will be skipped",
                file=sys.stderr,
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = select_items(iter_items(data, keep_order=keep_order), item_selector)
    print(f"[info] {len(items)} item(s) to process")
    for item in items:
        sub = output_dir / sanitize_name(item["item_id"])
        print(f"\n=== item {item['item_id']} -> {sub} ===")
        USAGE.reset()
        if embedder is not None:
            embedder.clear_cache()
        run_item(item, client=client, model=model, out_dir=sub,
                 options=options, embedder=embedder, max_tokens=max_tokens,
                 pricing=pricing, extra_meta={"input_path": input_path})

    print("\n[done]")
