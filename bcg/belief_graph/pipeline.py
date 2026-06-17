"""
pipeline.py  (v3)
=================
Thin batch drivers over the streaming engine (stream.py). No scenarios, no
sessions: every input is normalised by loaders.iter_items into items, each a
flat list of role-tagged turns, and each item is built into one belief graph.

    run_item(...)   — one item: ingest its turns in order, then finalize
                      (backward + confidence + merge run once at the end).
    run_input(...)  — load a file, wire client/embedder from config, process
                      all selected items into <out_dir>/<item_id>/.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import (
    USAGE,
    load_config,
    load_embedding_config,
    make_client,
    make_embedder,
)
from .loaders import iter_items, load_input_file, sanitize_name, select_items
from .stream import StreamingBeliefBuilder, StreamOptions


def run_item(
    item: Dict[str, Any],
    *,
    client,
    model: str,
    out_dir: Path,
    options: StreamOptions,
    embedder=None,
    max_tokens: Optional[int] = None,
    pricing: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the belief graph for ONE normalised item."""
    builder = StreamingBeliefBuilder(
        client=client, model=model, item_id=item["item_id"],
        item_meta=item.get("meta"), out_dir=out_dir, options=options,
        embedder=embedder, max_tokens=max_tokens,
    )
    for turn in item["turns"]:
        builder.ingest_turn(turn["role"], turn["content"],
                            date=turn.get("date"), has_answer=turn.get("has_answer"))
    meta: Dict[str, Any] = {"order_sorted": item.get("order_sorted", False)}
    if extra_meta:
        meta.update(extra_meta)
    return builder.finalize(extra_meta=meta, pricing=pricing)


def run_input(
    input_path: str,
    config_path: str,
    output_dir: Path,
    *,
    model_key: Optional[str] = None,
    embedding_key: str = "embedding",
    options: Optional[StreamOptions] = None,
    item_selector: Optional[str] = None,
    keep_order: bool = False,
) -> None:
    options = options or StreamOptions()
    data = load_input_file(input_path)

    cfg = load_config(config_path, model_key=model_key)
    client = make_client(cfg)
    model = cfg.get("model") or cfg.get("model_name") or "gpt-4o-mini"
    max_tokens = cfg.get("max_tokens")
    pricing = cfg.get("pricing")

    masked = (cfg.get("api_key", "") or "")
    masked = (masked[:6] + "…" + masked[-3:]) if len(masked) > 10 else "***"
    print(f"[info] model={model}  base_url={cfg['base_url']}  api_key={masked}"
          + (f"  max_tokens={max_tokens}" if max_tokens else ""))

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
        if options.use_clustering:
            print(f"[warn] --use-clustering requires an {embedding_key!r} entry in "
                  f"{config_path} — clustering DISABLED for this run", file=sys.stderr)
            options = copy.deepcopy(options)
            options.use_clustering = False
        if options.merge_strategy == "embedding":
            print(f"[warn] merge-strategy=embedding requires an {embedding_key!r} entry "
                  f"in {config_path} — falling back to merge-strategy=llm", file=sys.stderr)
            options = copy.deepcopy(options)
            options.merge_strategy = "llm"

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
