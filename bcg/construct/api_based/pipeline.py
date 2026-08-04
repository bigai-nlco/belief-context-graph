"""
pipeline.py  (v3)
=================
Batch and async SDK drivers over the streaming engine (stream.py). Every input
is normalised into a flat list of role-tagged turns and built by the same
canonical graph implementation.

    run_item(...)   — one item: ingest its turns in order, then finalize
                      (backward + confidence + merge run once at the end).
    run_input(...)  — load a file, wire client/embedder from config, process
                      all selected items into <out_dir>/<item_id>/.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # NOTE: BeliefGraphPipeline/BeliefGraphRunResult below are a legacy SDK-style
    # wrapper kept from the original project for reference; they target an
    # external ``bcg.graph`` / ``bcg.memory`` / ``bcg.runner`` module set that is
    # not part of this repo and is not wired into run_input/run_item (the
    # actual entry points used by run.py / online_server.py). Rather than
    # delete this code, the import is made optional so the rest of this
    # subpackage keeps working even without those external modules; only
    # calling BeliefGraphPipeline.run(...) requires them to be importable.
    from bcg.graph import BCG
except ImportError:  # pragma: no cover - see note above
    BCG = Any  # type: ignore[assignment]

from .llm import (
    USAGE,
    load_belief_graph_config,
    load_config,
    load_embedding_config,
    make_client,
    make_embedder,
)
from .loaders import iter_items, load_input_file, sanitize_name, select_items
from .stream import StreamingBeliefBuilder, StreamOptions
from .utils import new_run_id


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
    bg_cfg = load_belief_graph_config(config_path, model_key=model_key)
    if bg_cfg:
        options = copy.deepcopy(options)
        options.apply_belief_graph_config(bg_cfg)
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
        print(f"[warn] no {embedding_key!r} entry in {config_path} — the incremental "
              "merge pass will be skipped", file=sys.stderr)

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


@dataclass(frozen=True, slots=True)
class BeliefGraphRunPaths:
    """SDK paths alongside the construct engine's native artifacts."""

    run_dir: Path
    artifacts_dir: Path
    graph: Path
    memory: Path
    token_usage: Path
    events: Path
    segments: Path
    io_beliefs: Path
    reasoning_beliefs: Path
    forward_relations: Path
    backward_relations: Path
    merges: Path
    result: Path
    final_graph: Path
    trajectory: Path
    graph_stream: Path

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BeliefGraphRunResult:
    run_id: str
    graph: BCG
    memory: dict[str, Any]
    output_paths: BeliefGraphRunPaths
    token_usage: dict[str, Any]
    counts: dict[str, Any]
    construct_result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BeliefGraphOptions:
    """Public SDK options mapped onto :class:`StreamOptions`."""

    evidence_mode: str = "sentence"
    incremental_merge: bool = True
    incremental_merge_threshold: float = 0.8
    verify_merge: bool = False
    context_chars: int = 9000
    io_context_chars: int = 6000
    min_content_len: int = 0

    def to_stream_options(self) -> StreamOptions:
        return StreamOptions(
            evidence_mode=self.evidence_mode,
            incremental_merge=self.incremental_merge,
            incremental_merge_threshold=self.incremental_merge_threshold,
            verify_merge=self.verify_merge,
            context_chars=self.context_chars,
            min_content_len=self.min_content_len,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_stream_options().to_dict(),
            "io_context_chars": self.io_context_chars,
        }


class BeliefGraphPipeline:
    """Async SDK entry point for the canonical streaming engine."""

    def __init__(
        self,
        llm: Any,
        *,
        output_root: str | Path = ".bcg/runs",
        run_id: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        scenario: str = "research",
        item_id: str = "trajectory",
        evidence_mode: str = "sentence",
        incremental_merge: bool = True,
        incremental_merge_threshold: float = 0.8,
        verify_merge: bool = False,
        context_chars: int = 9000,
        io_context_chars: int = 6000,
        min_content_len: int = 0,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: Any | None = None,
    ) -> None:
        del min_segment_len, confidence_config
        self.llm = llm
        self.output_root = Path(output_root)
        self.run_id = run_id or new_run_id()
        self.model = model
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.item_id = item_id
        self.options = BeliefGraphOptions(
            evidence_mode=evidence_mode,
            incremental_merge=incremental_merge,
            incremental_merge_threshold=incremental_merge_threshold,
            verify_merge=verify_merge,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_content_len=min_content_len,
        )
        self.embedder = embedder

    async def run(
        self,
        trajectory: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> BeliefGraphRunResult:
        from bcg.memory import BCGMemory
        from bcg.runner import BCGRunner

        runner = BCGRunner(
            memory=BCGMemory(graph=BCG(metadata={"run_id": self.run_id})),
            llm=self.llm,
            output_root=self.output_root,
        )
        return await runner.observe_trajectory(
            trajectory,
            run_id=self.run_id,
            model=self.model,
            max_tokens=self.max_tokens,
            scenario=self.scenario,
            item_id=self.item_id,
            embedder=self.embedder,
            metadata=metadata,
            options=self.options,
        )


__all__ = [
    "BeliefGraphOptions",
    "BeliefGraphPipeline",
    "BeliefGraphRunPaths",
    "BeliefGraphRunResult",
    "run_input",
    "run_item",
]
