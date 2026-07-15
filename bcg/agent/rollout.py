"""CLI entry point for agent workflow rollouts.

Default backend is in-process vLLM (no HTTP server needed). Use
``--backend ray_vllm --ray-num-replicas N`` to run N parallel vLLM
replicas on separate GPUs under Ray, or ``--backend openai`` to talk to
an external OpenAI-compatible server.

Usage (see scripts/rollout.sh for a batteries-included wrapper)::

    bcg agent run \
        --model /share/nlp/share/plm/Qwen3-4B-Thinking-2507 \
        --tasks bamboogle hotpotqa 2wiki musique gaia \
        --num-samples 1 --max-steps 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the project root importable when the module is run directly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
for p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bcg.agent.benchmark_loader import AVAILABLE_BENCHMARKS  # noqa: E402
from bcg.agent.context_memory import CONTEXT_MEMORY_BASELINE_MODES, CONTEXT_MEMORY_MODES  # noqa: E402
from bcg.agent.harness import HARNESS_REGISTRY, get_harness  # noqa: E402
from bcg.cli_help import RichArgumentParser  # noqa: E402

_PROMPTS_DIR = _THIS_DIR / "prompts"


def _parse_args(argv: list[str] | None = None, prog: str | None = None):
    parser = RichArgumentParser(
        prog=prog,
        description="Run BeliefTracer agent workflows over supported benchmarks.",
    )

    parser.add_argument(
        "--model", required=True, help="HF model path / name (local path or hub id)"
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "sglang", "sglang_dp", "ray_vllm", "openai", "api"),
        default="vllm",
        help="Inference backend. 'vllm' loads the model in-process (default); "
        "'sglang' loads the model through SGLang's in-process engine; "
        "'sglang_dp' spawns multiple SGLang worker processes for data-parallel "
        "agent trajectories; "
        "'ray_vllm' spawns multiple Ray actors, each hosting its own vLLM "
        "engine, for concurrent agents across GPUs; "
        "'openai' talks to an external OpenAI-compatible server (via rllm); "
        "'api' talks to any OpenAI-compatible API directly (no vLLM/rllm dep).",
    )
    parser.add_argument(
        "--harness",
        choices=sorted(HARNESS_REGISTRY),
        default=None,
        help="Agent harness preset. Selects a bundled system prompt, parser, "
        "and tool list. Explicit --prompt, --parser-name, or --tools flags "
        "override the harness defaults.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenAI-compatible base URL (only used with --backend openai)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Temporary API key override (default: OPENAI_API_KEY from root .env; "
        "EMPTY for unauthenticated local servers)",
    )

    # Ray-distributed vLLM knobs (ignored unless --backend ray_vllm)
    parser.add_argument(
        "--ray-num-replicas",
        type=int,
        default=1,
        help="Number of Ray-hosted vLLM replicas. Each owns --tensor-parallel-size "
        "GPUs. Aggregate GPU count = replicas * TP.",
    )
    parser.add_argument(
        "--ray-address",
        default="",
        help="Ray cluster address (defaults to $RAY_ADDRESS or a local cluster).",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of SGLang worker processes when --backend sglang_dp.",
    )
    parser.add_argument(
        "--data-parallel-devices",
        default="",
        help="Comma-separated device IDs for --backend sglang_dp. Defaults to "
        "$CUDA_VISIBLE_DEVICES or 0..N-1.",
    )

    # Local engine knobs (ignored when --backend openai)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-model-len",
        "--vllm-max-model-len",
        dest="max_model_len",
        type=int,
        default=None,
        help="Maximum model context length for local vLLM/SGLang backends.",
    )
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking (extended reasoning) for the inference model",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["gpqa_diamond"],
        choices=sorted(AVAILABLE_BENCHMARKS),
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--artifacts-dir",
        "--benchmarks-dir",
        dest="artifacts_dir",
        default="",
        help="Override the benchmark data directory",
    )
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only run specific task IDs (e.g. --task-ids 450 455).",
    )
    parser.add_argument(
        "--exclude-ids",
        nargs="+",
        default=None,
        help="Skip specific task IDs (e.g. --exclude-ids 450 455).",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="Shuffle benchmark tasks with --shuffle-seed before truncation (default).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Preserve benchmark file order (disables the default shuffle).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed for the pre-rollout task shuffle. Same seed -> same ordering.",
    )

    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help="rllm tool names to expose to the agent. 'local_search' routes to "
        "the local dense retrieval server (RETRIEVAL_SERVER_URL), matching the "
        "fused-agent training setup. Overrides harness default when set.",
    )
    parser.add_argument(
        "--retrieval-server-url",
        default="",
        help="Dense retrieval server URL for 'local_search' (overrides RETRIEVAL_SERVER_URL)",
    )
    parser.add_argument("--retrieval-max-results", type=int, default=10)
    parser.add_argument("--retrieval-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--serper-endpoint",
        default=os.environ.get("SERPER_ENDPOINT", "https://google.serper.dev/search"),
        help="Serper search endpoint.",
    )
    parser.add_argument(
        "--serper-country",
        default=os.environ.get("SERPER_COUNTRY", "us"),
        help="Serper gl country code (default: us).",
    )
    parser.add_argument(
        "--serper-language",
        default=os.environ.get("SERPER_LANGUAGE", "en"),
        help="Serper hl language code (default: en).",
    )
    parser.add_argument("--serper-timeout", type=float, default=30.0)
    parser.add_argument(
        "--serper-max-output-chars",
        type=int,
        default=12000,
        help="Maximum formatted characters returned by one serper_search call.",
    )
    parser.add_argument(
        "--serper-scrape-endpoint",
        default=os.environ.get("SERPER_SCRAPE_ENDPOINT", "https://scrape.serper.dev"),
        help="Serper page-content extraction endpoint.",
    )
    parser.add_argument(
        "--serper-scrape-timeout",
        type=float,
        default=float(os.environ.get("SERPER_SCRAPE_TIMEOUT", "30")),
        help="Timeout in seconds for one serper_scrape call.",
    )
    parser.add_argument(
        "--serper-scrape-max-output-chars",
        type=int,
        default=int(os.environ.get("SERPER_SCRAPE_MAX_OUTPUT_CHARS", "30000")),
        help="Maximum page-content characters returned by one serper_scrape call.",
    )
    parser.add_argument(
        "--browsecomp-grader-model",
        default=os.environ.get("BROWSECOMP_GRADER_MODEL", ""),
        help="LLM used for official-style BrowseComp judging (default: --model).",
    )
    parser.add_argument(
        "--browsecomp-grader-base-url",
        default=os.environ.get("BROWSECOMP_GRADER_BASE_URL", ""),
        help="OpenAI-compatible grader base URL (default: --base-url / OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--browsecomp-grader-timeout",
        type=float,
        default=float(os.environ.get("BROWSECOMP_GRADER_TIMEOUT", "120")),
        help="BrowseComp grader request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--browsecomp-grader-max-tokens",
        type=int,
        default=int(os.environ.get("BROWSECOMP_GRADER_MAX_TOKENS", "2048")),
        help="BrowseComp grader completion budget (default: 2048).",
    )
    parser.add_argument(
        "--browsecomp-grader-max-retries",
        type=int,
        default=int(os.environ.get("BROWSECOMP_GRADER_MAX_RETRIES", "2")),
        help="Retries after a failed or unparsable BrowseComp judge call (default: 2).",
    )
    parser.add_argument(
        "--bcp-index-dir",
        default="",
        help="BrowseComp-Plus local index artifact directory for bcp_search.",
    )
    parser.add_argument(
        "--bcp-max-output-chars",
        type=int,
        default=6000,
        help="Maximum characters returned by each bcp_search tool call.",
    )
    parser.add_argument(
        "--retrieval-method",
        choices=["bm25", "hero", "hero4"],
        default="bm25",
        help="Retrieval method: 'bm25' (default), 'hero' (BM25 + embedding rerank), "
        "or 'hero4' (BM25 -> embedding -> reranker -> LLM judge).",
    )
    parser.add_argument(
        "--hero-bm25-top-k",
        type=int,
        default=10,
        help="HerO: BM25 candidate pool size before embedding reranking (default: 10)",
    )
    parser.add_argument(
        "--hero-embedding-model",
        default="Qwen3-Embedding-8B",
        help="HerO: Embedding model name (default: SFR-Embedding-2_R)",
    )
    parser.add_argument(
        "--hero-embedding-url",
        default="http://10.2.152.9:8008/v1/embeddings",
        help="HerO: Remote embedding API URL (default: http://10.2.152.9:8008/v1/embeddings)",
    )
    parser.add_argument(
        "--hero-embedding-device",
        default="cpu",
        help="HerO: Device for embedding model - 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--hero-batch-size",
        type=int,
        default=16,
        help="HerO: Batch size for embedding computation (default: 16)",
    )
    parser.add_argument(
        "--hyde",
        dest="hyde",
        action="store_true",
        default=True,
        help="Enable HyDE: model generates hypothetical passages in query (default).",
    )
    parser.add_argument(
        "--no-hyde",
        dest="hyde",
        action="store_false",
        help="Disable HyDE: model sends plain natural-language queries.",
    )

    # Four-stage retrieval ("hero4")
    parser.add_argument("--stage1-bm25-k", type=int, default=1000,
                        help="hero4: BM25 candidate pool (actual = min(k, N)). Default 1000.")
    parser.add_argument("--stage2-embed-k", type=int, default=32,
                        help="hero4: embedding-rerank survivors, also the rerank service's "
                        "batch size. Default 32 — kept at a single safe batch so rerank "
                        "never needs the split+merge path (jina-reranker-v3's vLLM serving "
                        "path misranks at n=64 in one call; merging survivors across split "
                        "batches also has its own cost, since scores aren't comparable "
                        "across batches). Raise only if the service-side bug is fixed.")
    parser.add_argument("--stage3-rerank-k", type=int, default=10,
                        help="hero4: reranker survivors / final top_k upper bound. Default 10.")
    parser.add_argument("--rerank-url", default="http://10.2.152.9:8010",
                        help="hero4: reranker server base URL. Default uses the "
                        "model's /v1/completions + official template (P(yes)). "
                        "Pass a .../v1/rerank URL to use the native rerank API.")
    parser.add_argument("--rerank-model", default="Qwen3-Reranker-0.6B",
                        help="hero4: reranker model name.")
    parser.add_argument("--enable-judge", dest="enable_judge", action="store_true",
                        default=True, help="hero4: enable LLM relevance judge (default).")
    parser.add_argument("--no-judge", dest="enable_judge", action="store_false",
                        help="hero4: disable the LLM relevance judge stage.")
    parser.add_argument("--judge-model", default="",
                        help="hero4: judge model (default: --model / $MODEL).")
    parser.add_argument("--judge-base-url", default="",
                        help="hero4: judge OpenAI-compatible base URL (default: $OPENAI_BASE_URL).")
    parser.add_argument("--judge-api-key", default="",
                        help="hero4: judge API key (default: $OPENAI_API_KEY).")
    parser.add_argument("--judge-max-workers", type=int, default=10,
                        help="hero4: concurrency for per-item judge calls. Default 10.")
    parser.add_argument("--judge-max-items", type=int, default=10,
                        help="hero4: cap on items sent to the judge. Default 10.")

    # Sandboxed file-read tool + two-layer archive
    parser.add_argument("--file-tool-root", default="",
                        help="Sandbox root for read_file / archive "
                        "(default: $BELIEF_TRACER_FILE_ROOT or ai_workspace/).")
    parser.add_argument("--enable-file-read", action="store_true",
                        help="Expose the read_file tool (auto-on with --enable-archive).")
    parser.add_argument("--enable-archive", action="store_true",
                        help="Write a two-layer archive of tool results and inject "
                        "manifest refs into the system prompt (implies --enable-file-read).")
    parser.add_argument("--layered-context", action="store_true",
                        help="Use layered message assembly: belief graph as a standalone "
                        "user message (not in system prompt), detailed rules in user msg. "
                        "Implied by --enable-archive; use this flag alone to get the "
                        "message layout without archiving.")
    parser.add_argument("--recent-turns", type=int, default=0,
                        help="Archive mode: keep only the last N conversation turns in "
                        "context (0 = keep all, no trimming). Default 0.")
    parser.add_argument(
        "--context-memory-mode",
        choices=sorted(CONTEXT_MEMORY_MODES),
        default="belief_graph",
        help="Context memory backend. 'belief_graph' preserves the existing "
        "Belief Graph service path; claude_pipeline/codex_handoff/"
        "opencode_marker replace the graph slot without calling the graph "
        "service; 'none' disables the extra context slot.",
    )
    parser.add_argument("--context-memory-recent-observations", type=int, default=3,
                        help="Baseline modes: number of recent tool observations to keep verbatim.")
    parser.add_argument("--context-memory-tail-turns", type=int, default=2,
                        help="OpenCode-style baseline: number of recent turns to retain in tail.")
    parser.add_argument("--context-memory-max-chars", type=int, default=8000,
                        help="Baseline modes: max rendered context-memory block characters.")
    parser.add_argument("--context-memory-tool-summary-chars", type=int, default=200,
                        help="Baseline modes: max chars per tool summary.")
    parser.add_argument("--context-memory-interval", type=int, default=1,
                        help="Baseline modes: compact/update interval in model turns.")
    parser.add_argument(
        "--context-memory-summarizer",
        choices=["local", "llm"],
        default="local",
        help="Baseline summarizer policy. 'local' uses extractive summaries; "
        "'llm' calls the same OpenAI-compatible model/base URL/API key as the "
        "main agent and falls back to local summaries on repeated failures.",
    )
    parser.add_argument("--context-memory-summarizer-max-tokens", type=int, default=2048,
                        help="Baseline LLM summarizer: max completion tokens per compact call.")
    parser.add_argument("--context-memory-summarizer-timeout", type=float, default=120.0,
                        help="Baseline LLM summarizer: HTTP timeout in seconds.")
    parser.add_argument("--context-memory-summarizer-failure-limit", type=int, default=3,
                        help="Baseline LLM summarizer: disable LLM summarization after this many consecutive failures.")
    parser.add_argument("--context-memory-log-preview-chars", type=int, default=0,
                        help="Log this many characters of rendered context-memory/summary previews (0 = metadata only).")
    parser.add_argument("--max-tool-workers", type=int, default=1,
                        help="When a turn issues multiple non-finish tool calls, run up "
                        "to this many concurrently in a thread pool (default 1 = "
                        "sequential, identical to legacy behavior).")
    parser.add_argument(
        "--parser-name",
        default=None,
        help="rllm ToolParser name (qwen, r1, llama, ...). Overrides harness default when set.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=2048)

    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--passk", type=int, default=1)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults are model-aware; Qwen3.5 uses "
        "1.0 with --enable-thinking and 0.7 otherwise.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling probability. Defaults are model-aware; Qwen3.5 "
        "uses 0.95 with --enable-thinking and 0.8 otherwise.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling override. Qwen3.5 defaults to 20.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling override. Qwen3.5 defaults to 0.0.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty override. Qwen3.5 defaults to 1.5.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty override. Qwen3.5 defaults to 1.0.",
    )

    parser.add_argument("--n-parallel-tasks", type=int, default=32)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument(
        "--mixed-rollouts",
        action="store_true",
        dest="mixed_rollouts",
        help="Pool rollouts across every benchmark in --tasks into a single "
        "workflow engine, so slow-tail problems on one benchmark overlap with "
        "fast finishers on another. Per-benchmark scoring and outputs are "
        "unchanged; only the dispatching changes. Falls back to the "
        "one-benchmark-at-a-time path when only one benchmark is loaded.",
    )
    parser.add_argument(
        "--mixed-evals",
        action="store_true",
        dest="mixed_rollouts",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--prompt",
        default="",
        help="Path to a .txt file whose contents override the agent system prompt. "
        "Relative paths are resolved from bcg/agent/prompts/. "
        "Defaults to the built-in SEARCH_SYSTEM_PROMPT.",
    )
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/belief_tracer")
    parser.add_argument("--save-alias", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-auto-ui",
        action="store_true",
        help="Do not auto-start the BeliefTracer UI before running rollouts.",
    )
    parser.add_argument(
        "--belief-graph-url",
        default="",
        help="Belief Graph service URL (e.g. http://10.1.101.147:8848). Empty = disabled.",
    )
    parser.add_argument(
        "--belief-graph-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for belief graph HTTP calls (default: 300s).",
    )
    parser.add_argument(
        "--belief-graph-mode",
        choices=["none", "augment", "only"],
        default="augment",
        help="Belief graph ablation mode: 'none' = no belief graph, original context only; "
        "'augment' = original context + belief graph (default); "
        "'only' = belief graph replaces intermediate conversation history.",
    )
    parser.add_argument(
        "--graph-format",
        choices=[
            "structured", "narrative", "markdown", "xml", "triplet", "yaml",
            "json", "deepseek_v4",
        ],
        default="structured",
        help="Belief graph prompt format: 'structured' (default), 'narrative', "
        "'markdown', 'xml', 'triplet' (KG triples), 'yaml', 'json', "
        "'deepseek_v4' (DeepSeek-V4 encoded fake dialogue).",
    )
    parser.add_argument(
        "--no-graph-relations",
        dest="graph_include_relations",
        action="store_false",
        default=True,
        help="Exclude relation edges from belief graph context (only include belief nodes).",
    )
    parser.add_argument(
        "--belief-graph-placement",
        choices=["user", "system"],
        default="user",
        help="Where to place the belief graph in layered context mode. "
        "'user' keeps the existing standalone user message; 'system' appends "
        "the <belief_graph> block to the system prompt. Legacy non-layered "
        "mode already embeds the graph in the system prompt.",
    )
    parser.add_argument(
        "--belief-graph-interval", type=int, default=1,
        help="Rebuild the belief graph every N model turns instead of every turn "
        "(default 1 = every turn, no behavior change). Turns in between are "
        "buffered and pushed together on the triggering turn; the prompt keeps "
        "showing the last built snapshot until then.",
    )
    parser.add_argument(
        "--tonggraph-sync",
        action="store_true",
        default=os.environ.get("TONGGRAPH_SYNC", "").lower() in {"1", "true", "yes", "on"},
        help="Sync each belief graph snapshot into TongGraph Server and verify read-back.",
    )
    parser.add_argument(
        "--tonggraph-url",
        default=os.environ.get("TONGGRAPH_BASE_URL", ""),
        help="TongGraph Server URL, e.g. http://10.2.152.51:8719.",
    )
    parser.add_argument(
        "--tonggraph-token",
        default=os.environ.get("TONGGRAPH_TOKEN") or os.environ.get("TONGGRAPH_AGENT_WRITER_TOKEN", ""),
        help="TongGraph writer token; defaults to $TONGGRAPH_TOKEN or $TONGGRAPH_AGENT_WRITER_TOKEN.",
    )
    parser.add_argument(
        "--tonggraph-graph",
        default=os.environ.get("TONGGRAPH_GRAPH", "agent_workspace"),
        help="TongGraph physical graph name (default: agent_workspace).",
    )
    parser.add_argument(
        "--tonggraph-logical-graph-id",
        default=os.environ.get("TONGGRAPH_LOGICAL_GRAPH_ID", ""),
        help="Optional fixed TongGraph logical_graph_id; defaults to each problem id.",
    )
    parser.add_argument(
        "--tonggraph-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for TongGraph sync requests.",
    )
    parser.add_argument(
        "--tonggraph-index-name",
        default="agent_text",
        help="Fulltext index name to create for synced graphs.",
    )
    parser.add_argument(
        "--no-tonggraph-index",
        action="store_true",
        help="Disable fulltext index creation during TongGraph sync.",
    )
    parser.add_argument(
        "--tonggraph-embedding-url",
        default=os.environ.get("TONGGRAPH_EMBEDDING_URL") or os.environ.get("HERO_EMBEDDING_URL", ""),
        help="Embedding API URL for TongGraph vectors; defaults to $TONGGRAPH_EMBEDDING_URL or $HERO_EMBEDDING_URL.",
    )
    parser.add_argument(
        "--tonggraph-embedding-model",
        default=os.environ.get("TONGGRAPH_EMBEDDING_MODEL") or os.environ.get("HERO_EMBEDDING_MODEL", ""),
        help="Embedding model for TongGraph vectors; defaults to $TONGGRAPH_EMBEDDING_MODEL or $HERO_EMBEDDING_MODEL.",
    )
    parser.add_argument(
        "--tonggraph-embedding-index",
        default=os.environ.get("TONGGRAPH_EMBEDDING_INDEX", "agent_embedding"),
        help="TongGraph vector index name (default: agent_embedding).",
    )
    parser.add_argument(
        "--no-tonggraph-embedding",
        action="store_true",
        help="Disable TongGraph embedding/vector sync.",
    )
    parser.add_argument(
        "--tonggraph-embedding-batch-size",
        type=int,
        default=int(os.environ.get("TONGGRAPH_EMBEDDING_BATCH_SIZE", "16")),
        help="Batch size for TongGraph embedding API calls.",
    )

    args = parser.parse_args(argv)

    # Resolve harness defaults — explicit flags override harness values.
    harness_cfg = get_harness(args.harness) if args.harness else None

    # System prompt resolution: --no-system-prompt > --prompt > harness > built-in
    system_prompt = None
    if args.no_system_prompt:
        system_prompt = ""
    elif args.prompt:
        p = Path(args.prompt)
        if not p.is_absolute():
            candidates = [Path.cwd() / p, _PROMPTS_DIR / p, _REPO_ROOT / p]
            p = next((c for c in candidates if c.is_file()), candidates[0])
        system_prompt = p.read_text(encoding="utf-8").strip()
    elif harness_cfg:
        system_prompt = harness_cfg.system_prompt

    # Layered mode (archive on or --layered-context): the resolved prompt text
    # holds the detailed AVeriTeC rules (label definitions, search policy,
    # decision rules + query examples), which belong in the user message — keep
    # the system prompt minimal.
    _uses_context_memory_baseline = args.context_memory_mode in CONTEXT_MEMORY_BASELINE_MODES
    _layered = args.enable_archive or args.layered_context or _uses_context_memory_baseline
    user_rules_prompt = ""
    if _layered and system_prompt:
        from bcg.agent.config import DEFAULT_SYSTEM_PROMPT

        user_rules_prompt = system_prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Parser resolution: explicit --parser-name > harness > "qwen"
    parser_name = args.parser_name
    if parser_name is None:
        parser_name = harness_cfg.parser_name if harness_cfg else "qwen"

    # Tools resolution: explicit --tools > harness > []
    tools = args.tools
    if tools is None:
        tools = list(harness_cfg.tools) if harness_cfg else []

    from bcg.agent.config import AgentRolloutConfig

    return AgentRolloutConfig(
        model=args.model,
        backend=args.backend,
        harness=args.harness or "",
        base_url=args.base_url,
        api_key=args.api_key,
        ray_num_replicas=args.ray_num_replicas,
        ray_address=args.ray_address,
        data_parallel_size=args.data_parallel_size,
        data_parallel_devices=args.data_parallel_devices,
        tasks=list(args.tasks),
        artifacts_dir=args.artifacts_dir,
        max_problems=args.max_problems,
        task_ids=list(args.task_ids or []),
        exclude_ids=list(args.exclude_ids or []),
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        tools=list(tools),
        retrieval_server_url=args.retrieval_server_url,
        retrieval_max_results=args.retrieval_max_results,
        retrieval_timeout=args.retrieval_timeout,
        serper_endpoint=args.serper_endpoint,
        serper_country=args.serper_country,
        serper_language=args.serper_language,
        serper_timeout=args.serper_timeout,
        serper_max_output_chars=args.serper_max_output_chars,
        serper_scrape_endpoint=args.serper_scrape_endpoint,
        serper_scrape_timeout=args.serper_scrape_timeout,
        serper_scrape_max_output_chars=args.serper_scrape_max_output_chars,
        browsecomp_grader_model=args.browsecomp_grader_model,
        browsecomp_grader_base_url=args.browsecomp_grader_base_url,
        browsecomp_grader_timeout=args.browsecomp_grader_timeout,
        browsecomp_grader_max_tokens=args.browsecomp_grader_max_tokens,
        browsecomp_grader_max_retries=args.browsecomp_grader_max_retries,
        bcp_index_dir=args.bcp_index_dir,
        bcp_max_output_chars=args.bcp_max_output_chars,
        retrieval_method=args.retrieval_method,
        hero_bm25_top_k=args.hero_bm25_top_k,
        hero_embedding_model=args.hero_embedding_model,
        hero_embedding_url=args.hero_embedding_url,
        hero_embedding_device=args.hero_embedding_device,
        hero_batch_size=args.hero_batch_size,
        hyde=args.hyde,
        stage1_bm25_k=args.stage1_bm25_k,
        stage2_embed_k=args.stage2_embed_k,
        stage3_rerank_k=args.stage3_rerank_k,
        rerank_url=args.rerank_url,
        rerank_model=args.rerank_model,
        enable_judge=args.enable_judge,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        judge_max_workers=args.judge_max_workers,
        judge_max_items=args.judge_max_items,
        file_tool_root=args.file_tool_root,
        enable_file_read=args.enable_file_read or args.enable_archive,
        enable_archive=args.enable_archive,
        layered_context=_layered,
        recent_turns=args.recent_turns,
        context_memory_mode=args.context_memory_mode,
        context_memory_recent_observations=args.context_memory_recent_observations,
        context_memory_tail_turns=args.context_memory_tail_turns,
        context_memory_max_chars=args.context_memory_max_chars,
        context_memory_tool_summary_chars=args.context_memory_tool_summary_chars,
        context_memory_interval=args.context_memory_interval,
        context_memory_summarizer=args.context_memory_summarizer,
        context_memory_summarizer_max_tokens=args.context_memory_summarizer_max_tokens,
        context_memory_summarizer_timeout=args.context_memory_summarizer_timeout,
        context_memory_summarizer_failure_limit=args.context_memory_summarizer_failure_limit,
        context_memory_log_preview_chars=args.context_memory_log_preview_chars,
        max_tool_workers=args.max_tool_workers,
        parser_name=parser_name,
        max_steps=args.max_steps,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        passk=args.passk,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        n_parallel_tasks=args.n_parallel_tasks,
        retry_limit=args.retry_limit,
        mixed_rollouts=args.mixed_rollouts,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.max_model_len,
        vllm_dtype=args.vllm_dtype,
        vllm_enforce_eager=args.vllm_enforce_eager,
        enable_thinking=args.enable_thinking,
        **({"system_prompt": system_prompt} if system_prompt is not None else {}),
        user_rules_prompt=user_rules_prompt,
        output_dir=args.output_dir,
        save_alias=args.save_alias,
        overwrite=args.overwrite,
        auto_ui=not args.no_auto_ui,
        belief_graph_url=args.belief_graph_url,
        belief_graph_timeout=args.belief_graph_timeout,
        belief_graph_mode=args.belief_graph_mode,
        graph_format=args.graph_format,
        graph_include_relations=args.graph_include_relations,
        belief_graph_placement=args.belief_graph_placement,
        belief_graph_interval=args.belief_graph_interval,
        tonggraph_sync=args.tonggraph_sync,
        tonggraph_base_url=args.tonggraph_url,
        tonggraph_token=args.tonggraph_token,
        tonggraph_graph=args.tonggraph_graph,
        tonggraph_logical_graph_id=args.tonggraph_logical_graph_id,
        tonggraph_timeout=args.tonggraph_timeout,
        tonggraph_text_index="" if args.no_tonggraph_index else args.tonggraph_index_name,
        tonggraph_embedding_url="" if args.no_tonggraph_embedding else args.tonggraph_embedding_url,
        tonggraph_embedding_model="" if args.no_tonggraph_embedding else args.tonggraph_embedding_model,
        tonggraph_embedding_index="" if args.no_tonggraph_embedding else args.tonggraph_embedding_index,
        tonggraph_embedding_batch_size=args.tonggraph_embedding_batch_size,
    )


class _ColorFormatter(logging.Formatter):
    """ANSI-colored log formatter that highlights by component tag."""

    RESET = "\033[0m"
    GREY = "\033[90m"
    # Component → color
    _TAG_COLORS = {
        "[BeliefGraph]": "\033[35m",   # magenta
        "[Agent]":       "\033[36m",   # cyan
        "[APIEngine]":   "\033[34m",   # blue
        "[Workflow]":    "\033[32m",   # green
        "[ContextMemory]": "\033[36m", # cyan
        "[HerO":         "\033[33m",   # yellow
    }
    _LEVEL_COLORS = {
        "WARNING":  "\033[33m",        # yellow
        "ERROR":    "\033[31m",        # red
        "CRITICAL": "\033[1;31m",      # bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = self.GREY + self.formatTime(record, "%H:%M:%S") + self.RESET
        msg = record.getMessage()

        # Color by level for warnings/errors
        level_color = self._LEVEL_COLORS.get(record.levelname, "")
        if level_color:
            return f"{ts} {level_color}[{record.levelname}]{self.RESET} {level_color}{msg}{self.RESET}"

        # Color by component tag
        for tag, color in self._TAG_COLORS.items():
            if tag in msg:
                return f"{ts} {color}{msg}{self.RESET}"

        return f"{ts} {msg}"


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    import logging
    import os

    log_level = os.environ.get("LOGLEVEL", "WARNING").upper()
    if os.environ.get("BELIEF_GRAPH_DEBUG", ""):
        log_level = "INFO"

    # Console handler: concise, with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.WARNING))
    console_handler.setFormatter(_ColorFormatter())

    handlers = [console_handler]

    # File handler: full detail (always INFO when belief graph is enabled)
    log_file = os.environ.get("BELIEF_GRAPH_LOG", "")
    if log_file or os.environ.get("BELIEF_GRAPH_DEBUG", ""):
        log_file = log_file or "belief_graph.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        handlers.append(file_handler)

    logging.basicConfig(level=logging.WARNING, handlers=handlers)

    # Only set our loggers to DEBUG, not third-party ones
    for name in ("bcg.agent",):
        logging.getLogger(name).setLevel(logging.DEBUG)

    cfg = _parse_args(argv, prog=prog)
    if cfg.auto_ui:
        from bcg.agent.ui import ensure_ui_running

        ensure_ui_running(artifacts_dir=Path(cfg.output_dir))

    from bcg.agent.runner import run_agent_rollouts

    run_agent_rollouts(cfg)


if __name__ == "__main__":
    main()
