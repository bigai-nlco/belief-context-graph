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
