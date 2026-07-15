"""Lightweight runtime configuration for BeliefTracer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by using the available tools when needed.
When you have enough information, call the finish tool with the final answer clearly stated in \\boxed{} format."""


def model_tag_for_path(model: str) -> str:
    """Return the stable filesystem/display tag for a model path or name."""
    return Path(model).name.replace("/", "_")


def thinking_state_label(enable_thinking: bool) -> str:
    return "thinking" if enable_thinking else "no thinking"


def thinking_state_dir_suffix(enable_thinking: bool) -> str:
    return "thinking" if enable_thinking else "no-thinking"


def model_display_name(model: str, enable_thinking: bool) -> str:
    return f"{model_tag_for_path(model)} ({thinking_state_label(enable_thinking)})"


def model_output_dir_name(
    model: str,
    *,
    enable_thinking: bool,
    save_alias: str = "",
) -> str:
    alias = f"_{save_alias}" if save_alias else ""
    return (
        f"{model_tag_for_path(model)}_"
        f"{thinking_state_dir_suffix(enable_thinking)}"
        f"{alias}"
    )


@dataclass
class AgentRolloutConfig:
    """Configuration for a single agent workflow rollout run."""

    model: str
    backend: str = "vllm"
    harness: str = ""
    base_url: str = ""
    api_key: str = ""

    ray_num_replicas: int = 1
    ray_address: str = ""
    data_parallel_size: int = 1
    data_parallel_devices: str = ""

    tasks: list[str] = field(default_factory=lambda: ["gpqa_diamond"])
    artifacts_dir: str = ""
    max_problems: int | None = None
    task_ids: list[str] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)
    shuffle: bool = True
    shuffle_seed: int = 0

    tools: list[str] = field(default_factory=list)
    parser_name: str = "qwen"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Layered mode only: detailed AVeriTeC rules (label definitions, search
    # policy, decision rules) routed into the user message instead of system.
    user_rules_prompt: str = ""
    max_steps: int = 10
    max_response_length: int = 8192
    max_prompt_length: int = 32768
    max_new_tokens: int = 2048

    retrieval_server_url: str = ""
    retrieval_max_results: int = 10
    retrieval_timeout: float = 3600.0

    # Live Google web search through Serper. Its key is loaded from the root
    # .env into the environment and is never serialized into run artifacts.
    serper_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "SERPER_ENDPOINT", "https://google.serper.dev/search"
        )
    )
    serper_country: str = field(
        default_factory=lambda: os.environ.get("SERPER_COUNTRY", "us")
    )
    serper_language: str = field(
        default_factory=lambda: os.environ.get("SERPER_LANGUAGE", "en")
    )
    serper_timeout: float = 30.0
    serper_max_output_chars: int = 12000
    serper_scrape_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "SERPER_SCRAPE_ENDPOINT", "https://scrape.serper.dev"
        )
    )
    serper_scrape_timeout: float = field(
        default_factory=lambda: float(os.environ.get("SERPER_SCRAPE_TIMEOUT", "30"))
    )
    serper_scrape_max_output_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("SERPER_SCRAPE_MAX_OUTPUT_CHARS", "30000")
        )
    )

    # BrowseComp official-style answer judge. The grader key is deliberately
    # environment-only so serialized run configs never contain it.
    browsecomp_grader_model: str = field(
        default_factory=lambda: os.environ.get("BROWSECOMP_GRADER_MODEL", "")
    )
    browsecomp_grader_base_url: str = field(
        default_factory=lambda: os.environ.get("BROWSECOMP_GRADER_BASE_URL", "")
    )
    browsecomp_grader_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BROWSECOMP_GRADER_TIMEOUT", "120")
        )
    )
    browsecomp_grader_max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get("BROWSECOMP_GRADER_MAX_TOKENS", "2048")
        )
    )
    browsecomp_grader_max_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("BROWSECOMP_GRADER_MAX_RETRIES", "2")
        )
    )

    # BrowseComp-Plus local dense retrieval
    bcp_index_dir: str = ""
    bcp_max_output_chars: int = 6000

    # HerO retrieval configuration
    retrieval_method: str = "bm25"  # "bm25", "hero", or "hero4"
    hero_bm25_top_k: int = 10  # BM25 candidate pool size (reduced for CPU)
    hero_embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "HERO_EMBEDDING_MODEL", "SFR-Embedding-2_R"
        )
    )
    hero_embedding_device: str = "cpu"
    hero_batch_size: int = 16
    hero_embedding_url: str = field(
        default_factory=lambda: os.environ.get("HERO_EMBEDDING_URL", "")
    )  # empty = local SentenceTransformer; set URL for remote API
    hyde: bool = True  # whether to use HyDE (hypothetical document expansion) in queries

    # Four-stage retrieval ("hero4"): BM25 -> embedding -> reranker -> LLM judge
    stage1_bm25_k: int = 1000      # BM25 candidate pool (actual = min(stage1_bm25_k, N))
    # embedding rerank survivors, also the batch size handed to the rerank
    # service in one call. Capped at 32 (down from 64), matching
    # HerOSearchTool._RERANK_SAFE_BATCH, so rerank always runs as a single
    # batch — no split + final-rerank merge needed. jina-reranker-v3's vLLM
    # serving path has a confirmed bug at n=64 in one call (non-deterministic,
    # tail-biased ranking); the split-batch path in
    # _rerank_via_rerank_api_batched avoids that specific bug, but merging
    # survivors across batches has its own real cost: a batch's scores aren't
    # comparable to another batch's (a weak batch's strongest candidate can
    # outscore a strong batch's 3rd/4th-place candidate purely because there
    # was less competition in that batch), which needs a second rerank pass
    # over the merged survivors to correct. Keeping embedding output at <=32
    # sidesteps needing that correction at all. See
    # jina-embedding-reranker-migration memory for the full investigation.
    stage2_embed_k: int = 32
    stage3_rerank_k: int = 10      # reranker survivors (also final top_k upper bound)
    rerank_url: str = field(
        default_factory=lambda: os.environ.get(
            "RERANK_URL", "http://127.0.0.1:8010"
        )
    )
    rerank_model: str = field(
        default_factory=lambda: os.environ.get(
            "RERANK_MODEL", "Qwen3-Reranker-0.6B"
        )
    )
    enable_judge: bool = True      # LLM relevance judge as stage 4
    judge_model: str = ""          # empty = fall back to $MODEL / cfg.model
    judge_base_url: str = ""       # empty = fall back to $OPENAI_BASE_URL
    judge_api_key: str = ""        # empty = fall back to $OPENAI_API_KEY
    judge_max_workers: int = 10    # concurrency for per-item judge calls
    judge_max_items: int = 10      # cap on items sent to the judge

    # Sandboxed file-read tool + two-layer archive
    file_tool_root: str = ""       # empty = $BELIEF_TRACER_FILE_ROOT or ai_workspace/
    enable_file_read: bool = False  # expose read_file tool (auto-on when enable_archive)
    enable_archive: bool = False   # write two-layer archive + manifest refs
    layered_context: bool = False  # graph/rules split into context blocks (implied by enable_archive)
    recent_turns: int = 0          # 0 = keep all (no trimming); >0 = keep last N turns

    # Context-memory baselines. "belief_graph" preserves the existing service
    # path; other modes replace the graph prompt slot and never call the graph
    # service.
    context_memory_mode: str = "belief_graph"  # belief_graph|claude_pipeline|codex_handoff|opencode_marker|none
    context_memory_recent_observations: int = 3
    context_memory_tail_turns: int = 2
    context_memory_max_chars: int = 8000
    context_memory_tool_summary_chars: int = 200
    context_memory_interval: int = 1
    context_memory_summarizer: str = "local"  # local|llm
    context_memory_summarizer_max_tokens: int = 2048
    context_memory_summarizer_timeout: float = 120.0
    context_memory_summarizer_failure_limit: int = 3
    context_memory_log_preview_chars: int = 0

    # Parallel tool calls: when a turn's response contains multiple non-finish
    # <tool_call> blocks, run up to this many concurrently in a thread pool.
    # 1 (default) preserves the historical sequential execution order exactly.
    max_tool_workers: int = 1

    num_samples: int = 1
    passk: int = 1
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None

    n_parallel_tasks: int = 32
    retry_limit: int = 2
    mixed_rollouts: bool = False

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int | None = None
    vllm_dtype: str = "auto"
    vllm_enforce_eager: bool = False
    vllm_trust_remote_code: bool = True
    enable_thinking: bool = False

    output_dir: str = "artifacts/belief_tracer"
    save_alias: str = ""
    overwrite: bool = False
    auto_ui: bool = True

    # Belief Graph service
    belief_graph_url: str = field(default_factory=lambda: os.environ.get("BELIEF_GRAPH_URL", ""))
    belief_graph_timeout: float = 300.0
    belief_graph_scenario: str = "research"
    belief_graph_mode: str = "augment"  # "none", "augment", "only"
    graph_format: str = "structured"  # "structured", "narrative", "markdown", "xml"
    graph_include_relations: bool = True
    belief_graph_placement: str = "user"  # layered mode: "user" or "system"
    # Rebuild the graph every N model turns instead of every turn. 1 (default)
    # pushes/refreshes every turn (no behavior change). Turns in between are
    # buffered and flushed together on the triggering turn; the prompt keeps
    # showing the last built snapshot until then. Always flushed on done=True
    # so no buffered content is lost at trajectory end.
    belief_graph_interval: int = 1

    # TongGraph Server persistence for belief graph snapshots.
    tonggraph_sync: bool = field(
        default_factory=lambda: os.environ.get("TONGGRAPH_SYNC", "").lower()
        in {"1", "true", "yes", "on"}
    )
    tonggraph_base_url: str = field(default_factory=lambda: os.environ.get("TONGGRAPH_BASE_URL", ""))
    tonggraph_token: str = field(
        default_factory=lambda: os.environ.get("TONGGRAPH_TOKEN")
        or os.environ.get("TONGGRAPH_AGENT_WRITER_TOKEN", "")
    )
    tonggraph_graph: str = field(default_factory=lambda: os.environ.get("TONGGRAPH_GRAPH", "agent_workspace"))
    tonggraph_logical_graph_id: str = field(default_factory=lambda: os.environ.get("TONGGRAPH_LOGICAL_GRAPH_ID", ""))
    tonggraph_timeout: float = 30.0
    tonggraph_text_index: str = "agent_text"
    tonggraph_embedding_url: str = field(
        default_factory=lambda: os.environ.get("TONGGRAPH_EMBEDDING_URL")
        or os.environ.get("HERO_EMBEDDING_URL", "")
    )
    tonggraph_embedding_model: str = field(
        default_factory=lambda: os.environ.get("TONGGRAPH_EMBEDDING_MODEL")
        or os.environ.get("HERO_EMBEDDING_MODEL", "")
    )
    tonggraph_embedding_index: str = field(default_factory=lambda: os.environ.get("TONGGRAPH_EMBEDDING_INDEX", "agent_embedding"))
    tonggraph_embedding_batch_size: int = 16

    def resolved_base_url(self) -> str:
        return (
            self.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:30000/v1"
        )

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    def resolved_browsecomp_grader_model(self) -> str:
        return self.browsecomp_grader_model or self.model

    def resolved_browsecomp_grader_base_url(self) -> str:
        return self.browsecomp_grader_base_url or self.resolved_base_url()

    def resolved_browsecomp_grader_api_key(self) -> str:
        return (
            os.environ.get("BROWSECOMP_GRADER_API_KEY")
            or self.resolved_api_key()
        )

    @property
    def mixed_evals(self) -> bool:
        """Backward-compatible spelling for older callers."""
        return self.mixed_rollouts

    @mixed_evals.setter
    def mixed_evals(self, value: bool) -> None:
        self.mixed_rollouts = value


def default_rollout_config() -> AgentRolloutConfig:
    """Return the single source of default values used by agent entry points.

    CLI parsing and launch scripts must derive their values from this factory
    instead of introducing a second set of literals. Environment-backed fields
    are intentionally evaluated on each call so a caller can provide a
    project-specific ``.env`` before constructing the configuration.
    """

    return AgentRolloutConfig(model="")


AgenticEvalConfig = AgentRolloutConfig
