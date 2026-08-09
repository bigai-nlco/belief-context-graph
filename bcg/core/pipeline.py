"""Backend-neutral SDK pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bcg.core.contracts import RunOptions, RunResult


class BeliefGraphPipelineBase:
    """Compatibility pipeline implemented on top of the protocol-based runner."""

    options_type: type[RunOptions] = RunOptions

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
        incremental_merge_threshold: float = 0.86,
        verify_merge: bool = True,
        context_chars: int = 100000,
        io_context_chars: int = 6000,
        min_content_len: int = 0,
        min_segment_len: dict[str, int] | None = None,
        embedder: Any | None = None,
        confidence_config: Any | None = None,
    ) -> None:
        from bcg.core.utils import new_run_id

        del min_segment_len, confidence_config
        self.llm = llm
        self.output_root = Path(output_root)
        self.run_id = run_id or new_run_id()
        self.model = model
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.item_id = item_id
        self.options = self.options_type(
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
    ) -> RunResult:
        # Delayed imports avoid package-initialization cycles.
        from bcg.core.graph import BCG
        from bcg.core.memory import BCGMemory
        from bcg.core.runner import BCGRunner

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


__all__ = ["BeliefGraphPipelineBase"]
