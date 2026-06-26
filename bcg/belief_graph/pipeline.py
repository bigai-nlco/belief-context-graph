"""Incremental belief-graph construction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bcg.belief_graph.confidence import ConfidenceConfig
from bcg.belief_graph.constants import DEFAULT_MIN_SEGMENT_LEN
from bcg.belief_graph.utils import new_run_id
from bcg.graph import BCG


@dataclass(frozen=True, slots=True)
class BeliefGraphRunPaths:
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

    def to_dict(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "graph": str(self.graph),
            "memory": str(self.memory),
            "token_usage": str(self.token_usage),
            "events": str(self.events),
            "segments": str(self.segments),
            "io_beliefs": str(self.io_beliefs),
            "reasoning_beliefs": str(self.reasoning_beliefs),
            "forward_relations": str(self.forward_relations),
            "backward_relations": str(self.backward_relations),
            "merges": str(self.merges),
        }


@dataclass(frozen=True, slots=True)
class BeliefGraphRunResult:
    run_id: str
    graph: BCG
    memory: dict[str, Any]
    output_paths: BeliefGraphRunPaths
    token_usage: dict[str, Any]
    counts: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BeliefGraphOptions:
    use_split: bool = False
    split_threshold: float = 0.6
    split_min_sentences: int = 4
    split_buffer: int = 0
    merge_strategy: str = "off"
    merge_threshold: float = 0.86
    context_chars: int = 9000
    io_context_chars: int = 6000
    min_segment_len: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MIN_SEGMENT_LEN)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_split": self.use_split,
            "split_threshold": self.split_threshold,
            "split_min_sentences": self.split_min_sentences,
            "split_buffer": self.split_buffer,
            "merge_strategy": self.merge_strategy,
            "merge_threshold": self.merge_threshold,
            "context_chars": self.context_chars,
            "io_context_chars": self.io_context_chars,
            "min_segment_len": dict(self.min_segment_len),
        }


class BeliefGraphPipeline:
    """Convenience wrapper that feeds a trajectory through BCGMemory semantics."""

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
        use_split: bool = False,
        split_threshold: float = 0.6,
        split_min_sentences: int = 4,
        split_buffer: int = 0,
        merge_strategy: str = "off",
        merge_threshold: float = 0.86,
        context_chars: int = 9000,
        io_context_chars: int = 6000,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: ConfidenceConfig | None = None,
    ) -> None:
        self.llm = llm
        self.output_root = Path(output_root)
        self.run_id = run_id or new_run_id()
        self.model = model
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.item_id = item_id
        segment_lengths = dict(DEFAULT_MIN_SEGMENT_LEN)
        if min_segment_len:
            segment_lengths.update(min_segment_len)
        self.options = BeliefGraphOptions(
            use_split=use_split,
            split_threshold=split_threshold,
            split_min_sentences=split_min_sentences,
            split_buffer=split_buffer,
            merge_strategy=merge_strategy,
            merge_threshold=merge_threshold,
            context_chars=context_chars,
            io_context_chars=io_context_chars,
            min_segment_len=segment_lengths,
        )
        self.embedder = embedder
        self.confidence_config = confidence_config

    async def run(
        self,
        trajectory: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> BeliefGraphRunResult:
        """Run incremental ingestion over a trajectory and write run outputs."""

        from bcg.memory import BCGMemory
        from bcg.runner import BCGRunner

        memory = BCGMemory(
            graph=BCG(metadata={"run_id": self.run_id}),
            confidence_config=self.confidence_config,
        )
        runner = BCGRunner(
            memory=memory,
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
            use_split=self.options.use_split,
            split_threshold=self.options.split_threshold,
            split_min_sentences=self.options.split_min_sentences,
            split_buffer=self.options.split_buffer,
            merge_strategy=self.options.merge_strategy,
            merge_threshold=self.options.merge_threshold,
            context_chars=self.options.context_chars,
            io_context_chars=self.options.io_context_chars,
            min_segment_len=self.options.min_segment_len,
            embedder=self.embedder,
            confidence_config=self.confidence_config,
            metadata=metadata,
        )


def run_item(
    item: dict[str, Any],
    *,
    client: Any,
    model: str,
    out_dir: Path,
    options: Any,
    embedder: Any = None,
    max_tokens: int | None = None,
    pricing: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one normalized item with the legacy streaming engine."""

    from bcg.belief_graph.stream import StreamingBeliefBuilder

    builder = StreamingBeliefBuilder(
        client=client,
        model=model,
        item_id=item["item_id"],
        item_meta=item.get("meta"),
        out_dir=out_dir,
        options=options,
        embedder=embedder,
        max_tokens=max_tokens,
    )
    for turn in item["turns"]:
        builder.ingest_turn(
            turn["role"],
            turn["content"],
            date=turn.get("date"),
            has_answer=turn.get("has_answer"),
        )
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
    options: Any | None = None,
    item_selector: str | None = None,
    keep_order: bool = False,
) -> None:
    """Load an input file and process it with the legacy streaming engine."""

    import copy
    import sys

    from bcg.belief_graph.llm import (
        USAGE,
        load_config,
        load_embedding_config,
        make_client,
        make_embedder,
    )
    from bcg.belief_graph.loaders import (
        iter_items,
        load_input_file,
        sanitize_name,
        select_items,
    )
    from bcg.belief_graph.stream import StreamOptions

    options = options or StreamOptions()
    data = load_input_file(input_path)

    cfg = load_config(config_path, model_key=model_key)
    client = make_client(cfg)
    model = cfg.get("model") or cfg.get("model_name") or "gpt-4o-mini"
    max_tokens = cfg.get("max_tokens")
    pricing = cfg.get("pricing")

    masked = cfg.get("api_key", "") or ""
    masked = (masked[:6] + "…" + masked[-3:]) if len(masked) > 10 else "***"
    print(
        f"[info] model={model}  base_url={cfg['base_url']}  api_key={masked}"
        + (f"  max_tokens={max_tokens}" if max_tokens else "")
    )

    emb_cfg = load_embedding_config(config_path, embedding_key=embedding_key)
    embedder = None
    if emb_cfg is not None:
        embedder = make_embedder(emb_cfg)
        if emb_cfg.get("provider") == "local":
            print(
                f"[info] embedding provider=local  model={emb_cfg['model']}  "
                "(weights load lazily on first use)"
            )
        else:
            print(
                f"[info] embedding model={emb_cfg['model']}  "
                f"base_url={emb_cfg['base_url']}"
            )
    else:
        if options.use_clustering:
            print(
                f"[warn] --use-clustering requires an {embedding_key!r} entry in "
                f"{config_path} — clustering DISABLED for this run",
                file=sys.stderr,
            )
            options = copy.deepcopy(options)
            options.use_clustering = False
        if options.merge_strategy == "embedding":
            print(
                f"[warn] merge-strategy=embedding requires an {embedding_key!r} entry "
                f"in {config_path} — falling back to merge-strategy=llm",
                file=sys.stderr,
            )
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
        run_item(
            item,
            client=client,
            model=model,
            out_dir=sub,
            options=options,
            embedder=embedder,
            max_tokens=max_tokens,
            pricing=pricing,
            extra_meta={"input_path": input_path},
        )

    print("\n[done]")
