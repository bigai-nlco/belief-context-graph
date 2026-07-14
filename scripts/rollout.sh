#!/usr/bin/env bash
# Launch agent workflow rollout across benchmarks in experiments/artifacts/benchmarks/.
#
# All parameters are passed via explicit flags (see Usage below). Env vars
# of the same name still act as defaults so existing CI wrappers keep
# working, but flags win when both are set.
#
# The default backend talks to an OpenAI-compatible model service. Use
# --backend vllm for a single in-process engine, or --backend ray_vllm to
# load the model once per replica inside the rollout process.
#
# Web search defaults to the local dense retrieval server, matching the
# fused-agent training setup (rllm/experiments/fused/train_fused_agent.sh).
#
# Vitabench (benchmarks/vitabench) is dispatched through `vita run` — name
# the domain as `vitabench:<domain>` where domain is delivery, ota, or
# instore, or use the bare alias `vitabench` to fan out across all three.
# The assistant agent is served by --model-path (vllm_local, or the
# --openai-base-url when --backend openai). user/evaluator agents default
# to gpt-4.1 via the shared litellm gateway; override with --vita-user-model
# / --vita-evaluator-model (and --vita-litellm-* knobs).
#
# Usage:
#   ./scripts/rollout.sh \
#       --model-path /data/user/fukeshu/model/Qwen3-8B \
#       --benchmarks bamboogle,hotpotqa \
#       --backend openai --openai-base-url http://localhost:8001/v1 \
#       --n-parallel 2048 --max-problems 128
#
#   # vitabench against a local HF checkpoint
#   ./scripts/rollout.sh \
#       --model-path /path/to/hf_model --benchmarks vitabench \
#       --backend vllm --vllm-tp 1 --vllm-mem-fraction 0.22 \
#       --save-alias ckpt40
#
#   ./scripts/rollout.sh --help

set -euo pipefail

# ---------------------------------------------------------------------------
# Load .env file if present (provides default env vars for API keys, URLs, etc.)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# ---------------------------------------------------------------------------
# Defaults (env vars override the compile-time default; flags override env)
# ---------------------------------------------------------------------------

# Model service / backend
MODEL="${MODEL:-/data/user/fukeshu/model/Qwen3-8B}"
BACKEND="${BACKEND:-openai}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# Reasoning / thinking mode
ENABLE_THINKING="${ENABLE_THINKING:-1}"

# Local vLLM / Ray backend
RAY_REPLICAS="${RAY_REPLICAS:-8}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MEM_FRACTION="${VLLM_MEM_FRACTION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-}}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

# Benchmarks / rollout scale
BENCHMARKS_CSV="averitec"
MAX_PROBLEMS="${MAX_PROBLEMS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
PASSK="${PASSK:-1}"
N_PARALLEL="${N_PARALLEL:-8}"
MIXED_ROLLOUTS="${MIXED_ROLLOUTS:-${MIXED_EVALS:-0}}"

# Agent behavior
PROMPT_FILE="${PROMPT_FILE:-}"
NO_SYSTEM_PROMPT="${NO_SYSTEM_PROMPT:-0}"
PARSER_NAME="${PARSER_NAME:-qwen}"
TOOLS_CSV="${TOOLS:-averitec_search}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"

# Sampling
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

# Retrieval tool
RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}"
RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-10}"
RLLM_RETRIEVAL_SUMMARIZE="${RLLM_RETRIEVAL_SUMMARIZE:-0}"

# Live Web Search (Serper). SERPER_API_KEY is loaded from the root .env above;
# the launcher never puts it on the command line.
SERPER_ENDPOINT="${SERPER_ENDPOINT:-https://google.serper.dev/search}"
SERPER_COUNTRY="${SERPER_COUNTRY:-us}"
SERPER_LANGUAGE="${SERPER_LANGUAGE:-en}"
SERPER_TIMEOUT="${SERPER_TIMEOUT:-30}"
SERPER_MAX_OUTPUT_CHARS="${SERPER_MAX_OUTPUT_CHARS:-12000}"

# BrowseComp answer judge (official-style LLM binary verdict).
BROWSECOMP_GRADER_MODEL="${BROWSECOMP_GRADER_MODEL:-}"
BROWSECOMP_GRADER_BASE_URL="${BROWSECOMP_GRADER_BASE_URL:-}"
BROWSECOMP_GRADER_TIMEOUT="${BROWSECOMP_GRADER_TIMEOUT:-120}"
BROWSECOMP_GRADER_MAX_TOKENS="${BROWSECOMP_GRADER_MAX_TOKENS:-2048}"
BROWSECOMP_GRADER_MAX_RETRIES="${BROWSECOMP_GRADER_MAX_RETRIES:-2}"

# HerO retrieval configuration
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-bm25}"
HERO_BM25_TOP_K="${HERO_BM25_TOP_K:-10}"
HERO_EMBEDDING_MODEL="${HERO_EMBEDDING_MODEL:-SFR-Embedding-2_R}"
HERO_EMBEDDING_URL="${HERO_EMBEDDING_URL:-}"
HERO_EMBEDDING_DEVICE="${HERO_EMBEDDING_DEVICE:-cpu}"
HERO_BATCH_SIZE="${HERO_BATCH_SIZE:-16}"
HYDE="${HYDE:-1}"

# Four-stage retrieval ("hero4"): BM25 -> embedding -> reranker -> LLM judge
STAGE1_BM25_K="${STAGE1_BM25_K:-1000}"
STAGE2_EMBED_K="${STAGE2_EMBED_K:-32}"
STAGE3_RERANK_K="${STAGE3_RERANK_K:-10}"
RERANK_URL="${RERANK_URL:-http://10.2.152.9:8010}"
RERANK_MODEL="${RERANK_MODEL:-Qwen3-Reranker-0.6B}"
ENABLE_JUDGE="${ENABLE_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MAX_WORKERS="${JUDGE_MAX_WORKERS:-10}"
JUDGE_MAX_ITEMS="${JUDGE_MAX_ITEMS:-10}"

# Sandboxed file-read tool + two-layer archive
FILE_TOOL_ROOT="${FILE_TOOL_ROOT:-}"
ENABLE_FILE_READ="${ENABLE_FILE_READ:-0}"
ENABLE_ARCHIVE="${ENABLE_ARCHIVE:-0}"
RECENT_TURNS="${RECENT_TURNS:-0}"

# Context-memory baselines (empty = defer to Python defaults)
CONTEXT_MEMORY_MODE="${CONTEXT_MEMORY_MODE:-}"
CONTEXT_MEMORY_RECENT_OBSERVATIONS="${CONTEXT_MEMORY_RECENT_OBSERVATIONS:-}"
CONTEXT_MEMORY_TAIL_TURNS="${CONTEXT_MEMORY_TAIL_TURNS:-}"
CONTEXT_MEMORY_MAX_CHARS="${CONTEXT_MEMORY_MAX_CHARS:-}"
CONTEXT_MEMORY_TOOL_SUMMARY_CHARS="${CONTEXT_MEMORY_TOOL_SUMMARY_CHARS:-}"
CONTEXT_MEMORY_INTERVAL="${CONTEXT_MEMORY_INTERVAL:-}"
CONTEXT_MEMORY_SUMMARIZER="${CONTEXT_MEMORY_SUMMARIZER:-}"
CONTEXT_MEMORY_SUMMARIZER_MAX_TOKENS="${CONTEXT_MEMORY_SUMMARIZER_MAX_TOKENS:-}"
CONTEXT_MEMORY_SUMMARIZER_TIMEOUT="${CONTEXT_MEMORY_SUMMARIZER_TIMEOUT:-}"
CONTEXT_MEMORY_SUMMARIZER_FAILURE_LIMIT="${CONTEXT_MEMORY_SUMMARIZER_FAILURE_LIMIT:-}"
CONTEXT_MEMORY_LOG_PREVIEW_CHARS="${CONTEXT_MEMORY_LOG_PREVIEW_CHARS:-}"

# Output / ordering / logging
OUTPUT_DIR="${OUTPUT_DIR:-output}"
SAVE_ALIAS="${SAVE_ALIAS:-}"
OVERWRITE="${OVERWRITE:-1}"
SHUFFLE="${SHUFFLE:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
TASK_IDS=()
EXCLUDE_IDS=()
VERBOSE="${VERBOSE:-0}"

# Belief Graph service
BELIEF_GRAPH_URL="${BELIEF_GRAPH_URL:-http://10.1.100.76:8848}"
BELIEF_GRAPH_TIMEOUT="${BELIEF_GRAPH_TIMEOUT:-600}"
BELIEF_GRAPH_MODE="${BELIEF_GRAPH_MODE:-augment}"
GRAPH_FORMAT="${GRAPH_FORMAT:-structured}"
BELIEF_GRAPH_PLACEMENT="${BELIEF_GRAPH_PLACEMENT:-user}"

# vitabench-specific knobs (only read when --benchmarks includes vitabench[:domain]).
VITA_LITELLM_BASE_URL="${VITA_LITELLM_BASE_URL:-https://litellm.mybigai.ac.cn/}"
VITA_LITELLM_API_KEY="${VITA_LITELLM_API_KEY:-${OPENAI_API_KEY:-}}"
VITA_USER_MODEL="${VITA_USER_MODEL:-gpt-4.1}"
VITA_EVALUATOR_MODEL="${VITA_EVALUATOR_MODEL:-gpt-4.1}"
VITA_MAX_STEPS="${VITA_MAX_STEPS:-128}"
VITA_MAX_CONCURRENCY="${VITA_MAX_CONCURRENCY:-1}"
VITA_LANGUAGE="${VITA_LANGUAGE:-english}"
VITA_SAVE_PREFIX="${VITA_SAVE_PREFIX:-}"
VITA_CONDA_ENV="${VITA_CONDA_ENV:-vita}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/rollout.sh \
      --model-path /data/user/fukeshu/model/Qwen3-8B \
      --benchmarks bamboogle,hotpotqa --prompt bcg/agent/prompts/react.txt \
      --backend openai --openai-base-url http://localhost:8001/v1 \
      --n-parallel 2048 --max-problems 128

Flags:
  Model service / backend:
  --model-path PATH                 HF model path or OpenAI model id (default: /data/user/fukeshu/model/Qwen3-8B)
  --backend {vllm|ray_vllm|openai}  Inference backend (default: openai)
  --openai-base-url URL             OpenAI-compatible endpoint (openai only; default: http://localhost:8001/v1)
  --openai-api-key KEY              Temporary override for root .env OPENAI_API_KEY
  --enable-thinking                 Treat the model as a Qwen-style reasoning model
                                    and request thinking when the backend supports it

  Local vLLM / Ray backend:
  --ray-replicas N                  Ray vLLM replicas (ray_vllm only)
  --ray-address ADDR                Existing Ray cluster address
  --vllm-tp N                       vLLM tensor-parallel size per replica
  --vllm-mem-fraction F             gpu-memory-utilization
  --max-model-len N                 Override local engine context length
  --vllm-dtype STR                  auto|bfloat16|float16|...
  --vllm-enforce-eager              Disable CUDA graph capture

  Benchmarks / rollout scale:
  --benchmarks LIST                 Comma-separated benchmarks
                                    (e.g. bamboogle,hotpotqa,2wiki,musique)
  --max-problems N                  Cap problems per benchmark
  --num-samples N                   Samples per problem
  --passk N                         Report pass@N
  --n-parallel N                    Concurrent agent rollouts
  --mixed-rollouts                  Pool rollouts across every benchmark into
                                    one workflow engine so slow tails on one
                                    benchmark overlap with fast finishers on
                                    another (keeps GPUs saturated end-to-end).

  Agent behavior:
  --prompt FILE                     Path to system-prompt .txt file
                                    (relative to the project root, e.g. bcg/agent/prompts/gem.txt)
  --no-system-prompt                Disable system prompt entirely
  --parser-name STR                 rllm ToolParser (qwen, r1, llama, ...)
  --tools LIST                      Comma-separated rllm tool names, or "none"
                                    for no tools (default: local_search)
  --max-steps N                     Max tool-use turns per rollout
  --max-new-tokens N                Per-turn decode budget
  --max-response-length N           Max response tokens per turn (default: 32768)
  --max-prompt-length N             Max prompt tokens (default: 32768)

  Sampling:
  --temperature F                   Sampling temperature (default: model-aware)
  --top-p F                         Nucleus sampling probability (default: model-aware)
  --top-k N                         Top-k sampling override (default: model-aware)

  Retrieval tool:
  --retrieval-server-url URL        Override RETRIEVAL_SERVER_URL
  --retrieval-max-results N         Top-k docs returned per retrieval call
  --retrieval-method {bm25|hero}    Retrieval method: 'bm25' or 'hero' (default: bm25)
  --retrieval-summarize             Call the retrieval server /summarize
                                    endpoint after search (default: off)
  --no-retrieval-summarize          Return raw retrieved passages without
                                    summarization
  --serper-endpoint URL             Serper search endpoint
  --serper-country CODE             Serper gl country code (default: us)
  --serper-language CODE            Serper hl language code (default: en)
  --serper-timeout SECONDS          Serper HTTP timeout (default: 30)
  --serper-max-output-chars N       Max formatted chars per search (default: 12000)
  --browsecomp-grader-model MODEL   Answer judge model (default: agent model)
  --browsecomp-grader-base-url URL  Judge API base URL (default: agent base URL)
  --browsecomp-grader-timeout SEC   Judge request timeout (default: 120)
  --browsecomp-grader-max-tokens N  Judge completion budget (default: 2048)
  --browsecomp-grader-max-retries N Retries after failure/unparsable output (default: 2)
                                    Optional separate key: BROWSECOMP_GRADER_API_KEY

  HerO retrieval configuration (only used when --retrieval-method hero):
  --hero-bm25-top-k N               BM25 candidate pool size before embedding reranking (default: 10)
  --hero-embedding-model NAME       Embedding model name (default: SFR-Embedding-2_R)
  --hero-embedding-url URL          Remote embedding API URL (default: http://10.2.152.9:8008/v1/embeddings)
  --hero-embedding-device {cpu|cuda} Device for embedding model (default: cpu)
  --hero-batch-size N               Batch size for embedding computation (default: 16)
  --hyde / --no-hyde                Enable/disable HyDE hypothetical passage generation (default: enabled)

  Four-stage retrieval (only used when --retrieval-method hero4):
  --stage1-bm25-k N                 BM25 candidate pool, actual = min(N, corpus) (default: 1000)
  --stage2-embed-k N                Embedding-rerank survivors (default: 32; kept at a single
                                    safe rerank batch, see jina-reranker-v3 bug notes in code)
  --stage3-rerank-k N               Reranker survivors / final top_k (default: 10)
  --rerank-url URL                  Reranker base URL; bare base uses /v1/completions+template,
                                    a .../v1/rerank URL uses the native rerank API
                                    (default: http://10.2.152.9:8010)
  --rerank-model NAME               Reranker model (default: Qwen3-Reranker-0.6B)
  --enable-judge / --no-judge       LLM relevance judge stage (default: enabled)
  --judge-model NAME                Judge model (default: --model / $MODEL)
  --judge-base-url URL              Judge OpenAI-compatible base URL (default: $OPENAI_BASE_URL)
  --judge-api-key KEY               Judge API key (default: $OPENAI_API_KEY)
  --judge-max-workers N             Concurrency for per-item judge calls (default: 10)
  --judge-max-items N               Cap on items sent to the judge (default: 10)

  File-read tool + two-layer archive:
  --enable-archive                  Write two-layer archive + inject manifest refs
                                    (implies --enable-file-read; moves tool usage into
                                    the user message and belief graph into its own message)
  --enable-file-read                Expose the read_file tool (sandboxed)
  --file-tool-root DIR              Sandbox root (default: $BELIEF_TRACER_FILE_ROOT or ai_workspace/)
  --recent-turns N                  Keep only the last N conversation turns (0 = keep all)

  Context-memory baselines:
  --context-memory-mode MODE        belief_graph|none|claude_pipeline|codex_handoff|opencode_marker
  --context-memory-summarizer MODE  local|llm
  --context-memory-recent-observations N
                                    Recent tool observations kept verbatim
  --context-memory-tail-turns N     OpenCode-style retained tail turns
  --context-memory-max-chars N      Max rendered <context_memory> block chars
  --context-memory-tool-summary-chars N
                                    Max chars per local tool summary
  --context-memory-interval N       Compact/update interval in model turns
  --context-memory-summarizer-max-tokens N
                                    LLM summarizer completion token budget
  --context-memory-summarizer-timeout SECONDS
                                    LLM summarizer HTTP timeout
  --context-memory-summarizer-failure-limit N
                                    Disable LLM summarizer after N consecutive failures
  --context-memory-log-preview-chars N
                                    Log N chars of summary/render previews (0 = metadata only)

  Output / ordering / logging:
  --output-dir DIR                  Where per-benchmark results.json lands
  --save-alias STR                  Suffix appended to the output dir name
  --overwrite                       Overwrite existing results.json
  --shuffle / --no-shuffle          Randomize tasks before truncation
                                    (default: on)
  --shuffle-seed N                  Seed for the pre-rollout shuffle
  --verbose                         Stream all rollout logs (default: only print
                                    the final summary table)

  Belief Graph service:
  --belief-graph-url URL            Belief Graph service URL (default: http://10.1.100.76:8848)
                                    Set to empty string to disable.
  --belief-graph-timeout N          Timeout in seconds for belief graph calls (default: 600)
  --no-belief-graph                 Disable belief graph integration entirely
  --belief-graph-mode {none|augment|only}
                                    Ablation mode: 'none' = no belief graph;
                                    'augment' = context + graph (default);
                                    'only' = graph replaces conversation history
  --graph-format {structured|narrative|markdown|xml|triplet|yaml|json|deepseek_v4}
                                    Belief graph prompt format (default: structured)
  --belief-graph-placement {user|system}
                                    Layered-context graph placement (default: user);
                                    use system to append <belief_graph> to system prompt

Vitabench flags (only used when --benchmarks lists vitabench[:domain]):
  --vita-litellm-base-url URL       Gateway URL for user/evaluator agents
                                    (default: https://litellm.mybigai.ac.cn/)
  --vita-litellm-api-key KEY        Gateway API key (defaults to --openai-api-key,
                                    then $OPENAI_API_KEY)
  --vita-user-model NAME            Model id for user-agent     (default: gpt-4.1)
  --vita-evaluator-model NAME       Model id for evaluator-agent(default: gpt-4.1)
  --vita-max-steps N                vita run --max-steps        (default: 128)
  --vita-max-concurrency SPEC       vita run --max-concurrency  (default: 1)
                                    Either a single int (applied to every domain)
                                    or a comma-separated map, e.g.
                                    delivery=32,ota=16,instore=8. Domains missing
                                    from the map fall back to a `default=<N>`
                                    entry if present, otherwise to 1.
  --vita-language LANG              vita run --language         (default: english)
  --vita-save-prefix STR            Override the per-domain save-to prefix
                                    (default: derived from --save-alias)
  --vita-conda-env NAME             Conda env that has `vita`   (default: vita)

  -h, --help                        Show this help
EOF
}

die() { echo "[rollout] ERROR: $*" >&2; exit 1; }

require_val() {
  [[ -n "${2:-}" ]] || die "flag '$1' requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    # Model service / backend
    --model-path)            require_val "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
    --backend)               require_val "$1" "${2:-}"; BACKEND="$2"; shift 2 ;;
    --openai-base-url)       require_val "$1" "${2:-}"; OPENAI_BASE_URL="$2"; shift 2 ;;
    --openai-api-key)        require_val "$1" "${2:-}"; OPENAI_API_KEY="$2"; shift 2 ;;

    # Local vLLM / Ray backend
    --ray-replicas)          require_val "$1" "${2:-}"; RAY_REPLICAS="$2"; shift 2 ;;
    --ray-address)           require_val "$1" "${2:-}"; RAY_ADDRESS="$2"; shift 2 ;;
    --vllm-tp)               require_val "$1" "${2:-}"; VLLM_TP="$2"; shift 2 ;;
    --vllm-mem-fraction)     require_val "$1" "${2:-}"; VLLM_MEM_FRACTION="$2"; shift 2 ;;
    --max-model-len|--vllm-max-model-len) require_val "$1" "${2:-}"; MAX_MODEL_LEN="$2"; shift 2 ;;
    --vllm-dtype)            require_val "$1" "${2:-}"; VLLM_DTYPE="$2"; shift 2 ;;
    --vllm-enforce-eager)    VLLM_ENFORCE_EAGER=1; shift ;;
    --enable-thinking)       ENABLE_THINKING=1; shift ;;

    # Benchmarks / rollout scale
    --benchmarks)            require_val "$1" "${2:-}"; BENCHMARKS_CSV="$2"; shift 2 ;;
    --max-problems)          require_val "$1" "${2:-}"; MAX_PROBLEMS="$2"; shift 2 ;;
    --num-samples)           require_val "$1" "${2:-}"; NUM_SAMPLES="$2"; shift 2 ;;
    --passk)                 require_val "$1" "${2:-}"; PASSK="$2"; shift 2 ;;
    --n-parallel)            require_val "$1" "${2:-}"; N_PARALLEL="$2"; shift 2 ;;
    --mixed-rollouts|--mixed-evals) MIXED_ROLLOUTS=1; shift ;;

    # Agent behavior
    --prompt)                require_val "$1" "${2:-}"; PROMPT_FILE="$2"; shift 2 ;;
    --no-system-prompt)      NO_SYSTEM_PROMPT=1; shift ;;
    --parser-name)           require_val "$1" "${2:-}"; PARSER_NAME="$2"; shift 2 ;;
    --tools)                 require_val "$1" "${2:-}"; TOOLS_CSV="$2"; shift 2 ;;
    --max-steps)             require_val "$1" "${2:-}"; MAX_STEPS="$2"; shift 2 ;;
    --max-new-tokens)        require_val "$1" "${2:-}"; MAX_NEW_TOKENS="$2"; shift 2 ;;
    --max-response-length)   require_val "$1" "${2:-}"; MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --max-prompt-length)     require_val "$1" "${2:-}"; MAX_PROMPT_LENGTH="$2"; shift 2 ;;

    # Sampling
    --temperature)           require_val "$1" "${2:-}"; TEMPERATURE="$2"; shift 2 ;;
    --top-p)                 require_val "$1" "${2:-}"; TOP_P="$2"; shift 2 ;;
    --top-k)                 require_val "$1" "${2:-}"; TOP_K="$2"; shift 2 ;;

    # Retrieval tool
    --retrieval-server-url)  require_val "$1" "${2:-}"; RETRIEVAL_SERVER_URL="$2"; shift 2 ;;
    --retrieval-max-results) require_val "$1" "${2:-}"; RETRIEVAL_MAX_RESULTS="$2"; shift 2 ;;
    --retrieval-method)      require_val "$1" "${2:-}"; RETRIEVAL_METHOD="$2"; shift 2 ;;
    --retrieval-summarize)   RLLM_RETRIEVAL_SUMMARIZE=1; shift ;;
    --no-retrieval-summarize) RLLM_RETRIEVAL_SUMMARIZE=0; shift ;;
    --serper-endpoint)       require_val "$1" "${2:-}"; SERPER_ENDPOINT="$2"; shift 2 ;;
    --serper-country)        require_val "$1" "${2:-}"; SERPER_COUNTRY="$2"; shift 2 ;;
    --serper-language)       require_val "$1" "${2:-}"; SERPER_LANGUAGE="$2"; shift 2 ;;
    --serper-timeout)        require_val "$1" "${2:-}"; SERPER_TIMEOUT="$2"; shift 2 ;;
    --serper-max-output-chars) require_val "$1" "${2:-}"; SERPER_MAX_OUTPUT_CHARS="$2"; shift 2 ;;
    --browsecomp-grader-model) require_val "$1" "${2:-}"; BROWSECOMP_GRADER_MODEL="$2"; shift 2 ;;
    --browsecomp-grader-base-url) require_val "$1" "${2:-}"; BROWSECOMP_GRADER_BASE_URL="$2"; shift 2 ;;
    --browsecomp-grader-timeout) require_val "$1" "${2:-}"; BROWSECOMP_GRADER_TIMEOUT="$2"; shift 2 ;;
    --browsecomp-grader-max-tokens) require_val "$1" "${2:-}"; BROWSECOMP_GRADER_MAX_TOKENS="$2"; shift 2 ;;
    --browsecomp-grader-max-retries) require_val "$1" "${2:-}"; BROWSECOMP_GRADER_MAX_RETRIES="$2"; shift 2 ;;

    # HerO retrieval configuration
    --hero-bm25-top-k)       require_val "$1" "${2:-}"; HERO_BM25_TOP_K="$2"; shift 2 ;;
    --hero-embedding-model)  require_val "$1" "${2:-}"; HERO_EMBEDDING_MODEL="$2"; shift 2 ;;
    --hero-embedding-url)    require_val "$1" "${2:-}"; HERO_EMBEDDING_URL="$2"; shift 2 ;;
    --hero-embedding-device) require_val "$1" "${2:-}"; HERO_EMBEDDING_DEVICE="$2"; shift 2 ;;
    --hero-batch-size)       require_val "$1" "${2:-}"; HERO_BATCH_SIZE="$2"; shift 2 ;;
    --hyde)                  HYDE=1; shift ;;
    --no-hyde)              HYDE=0; shift ;;

    # Four-stage retrieval (hero4)
    --stage1-bm25-k)         require_val "$1" "${2:-}"; STAGE1_BM25_K="$2"; shift 2 ;;
    --stage2-embed-k)        require_val "$1" "${2:-}"; STAGE2_EMBED_K="$2"; shift 2 ;;
    --stage3-rerank-k)       require_val "$1" "${2:-}"; STAGE3_RERANK_K="$2"; shift 2 ;;
    --rerank-url)            require_val "$1" "${2:-}"; RERANK_URL="$2"; shift 2 ;;
    --rerank-model)          require_val "$1" "${2:-}"; RERANK_MODEL="$2"; shift 2 ;;
    --enable-judge)          ENABLE_JUDGE=1; shift ;;
    --no-judge)              ENABLE_JUDGE=0; shift ;;
    --judge-model)           require_val "$1" "${2:-}"; JUDGE_MODEL="$2"; shift 2 ;;
    --judge-base-url)        require_val "$1" "${2:-}"; JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-api-key)         require_val "$1" "${2:-}"; JUDGE_API_KEY="$2"; shift 2 ;;
    --judge-max-workers)     require_val "$1" "${2:-}"; JUDGE_MAX_WORKERS="$2"; shift 2 ;;
    --judge-max-items)       require_val "$1" "${2:-}"; JUDGE_MAX_ITEMS="$2"; shift 2 ;;

    # Sandboxed file-read tool + two-layer archive
    --file-tool-root)        require_val "$1" "${2:-}"; FILE_TOOL_ROOT="$2"; shift 2 ;;
    --enable-file-read)      ENABLE_FILE_READ=1; shift ;;
    --enable-archive)        ENABLE_ARCHIVE=1; shift ;;
    --recent-turns)          require_val "$1" "${2:-}"; RECENT_TURNS="$2"; shift 2 ;;

    # Context-memory baselines
    --context-memory-mode)   require_val "$1" "${2:-}"; CONTEXT_MEMORY_MODE="$2"; shift 2 ;;
    --context-memory-summarizer) require_val "$1" "${2:-}"; CONTEXT_MEMORY_SUMMARIZER="$2"; shift 2 ;;
    --context-memory-recent-observations) require_val "$1" "${2:-}"; CONTEXT_MEMORY_RECENT_OBSERVATIONS="$2"; shift 2 ;;
    --context-memory-tail-turns) require_val "$1" "${2:-}"; CONTEXT_MEMORY_TAIL_TURNS="$2"; shift 2 ;;
    --context-memory-max-chars) require_val "$1" "${2:-}"; CONTEXT_MEMORY_MAX_CHARS="$2"; shift 2 ;;
    --context-memory-tool-summary-chars) require_val "$1" "${2:-}"; CONTEXT_MEMORY_TOOL_SUMMARY_CHARS="$2"; shift 2 ;;
    --context-memory-interval) require_val "$1" "${2:-}"; CONTEXT_MEMORY_INTERVAL="$2"; shift 2 ;;
    --context-memory-summarizer-max-tokens) require_val "$1" "${2:-}"; CONTEXT_MEMORY_SUMMARIZER_MAX_TOKENS="$2"; shift 2 ;;
    --context-memory-summarizer-timeout) require_val "$1" "${2:-}"; CONTEXT_MEMORY_SUMMARIZER_TIMEOUT="$2"; shift 2 ;;
    --context-memory-summarizer-failure-limit) require_val "$1" "${2:-}"; CONTEXT_MEMORY_SUMMARIZER_FAILURE_LIMIT="$2"; shift 2 ;;
    --context-memory-log-preview-chars) require_val "$1" "${2:-}"; CONTEXT_MEMORY_LOG_PREVIEW_CHARS="$2"; shift 2 ;;

    # Output / ordering / logging
    --output-dir)            require_val "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --save-alias)            require_val "$1" "${2:-}"; SAVE_ALIAS="$2"; shift 2 ;;
    --overwrite)             OVERWRITE=1; shift ;;
    --shuffle)               SHUFFLE=1; shift ;;
    --no-shuffle)            SHUFFLE=0; shift ;;
    --task-ids)              shift; TASK_IDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TASK_IDS+=("$1"); shift; done ;;
    --exclude-ids)           shift; EXCLUDE_IDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do EXCLUDE_IDS+=("$1"); shift; done ;;
    --shuffle-seed)          require_val "$1" "${2:-}"; SHUFFLE_SEED="$2"; shift 2 ;;
    --verbose)               VERBOSE=1; shift ;;
    --belief-graph-url)      require_val "$1" "${2:-}"; BELIEF_GRAPH_URL="$2"; shift 2 ;;
    --belief-graph-timeout)  require_val "$1" "${2:-}"; BELIEF_GRAPH_TIMEOUT="$2"; shift 2 ;;
    --no-belief-graph)       BELIEF_GRAPH_URL=""; shift ;;
    --belief-graph-mode)     require_val "$1" "${2:-}"; BELIEF_GRAPH_MODE="$2"; shift 2 ;;
    --graph-format)          require_val "$1" "${2:-}"; GRAPH_FORMAT="$2"; shift 2 ;;
    --belief-graph-placement) require_val "$1" "${2:-}"; BELIEF_GRAPH_PLACEMENT="$2"; shift 2 ;;
    --no-graph-relations)    NO_GRAPH_RELATIONS=1; shift ;;
    --vita-litellm-base-url) require_val "$1" "${2:-}"; VITA_LITELLM_BASE_URL="$2"; shift 2 ;;
    --vita-litellm-api-key)  require_val "$1" "${2:-}"; VITA_LITELLM_API_KEY="$2"; shift 2 ;;
    --vita-user-model)       require_val "$1" "${2:-}"; VITA_USER_MODEL="$2"; shift 2 ;;
    --vita-evaluator-model)  require_val "$1" "${2:-}"; VITA_EVALUATOR_MODEL="$2"; shift 2 ;;
    --vita-max-steps)        require_val "$1" "${2:-}"; VITA_MAX_STEPS="$2"; shift 2 ;;
    --vita-max-concurrency)  require_val "$1" "${2:-}"; VITA_MAX_CONCURRENCY="$2"; shift 2 ;;
    --vita-language)         require_val "$1" "${2:-}"; VITA_LANGUAGE="$2"; shift 2 ;;
    --vita-save-prefix)      require_val "$1" "${2:-}"; VITA_SAVE_PREFIX="$2"; shift 2 ;;
    --vita-conda-env)        require_val "$1" "${2:-}"; VITA_CONDA_ENV="$2"; shift 2 ;;
    -h|--help)               usage; exit 0 ;;
    --) shift; break ;;
    *)  die "unknown argument: $1 (run '$0 --help' for the flag list)" ;;
  esac
done

# Remaining args after -- are passed through to python directly.
PASSTHROUGH_ARGS=("$@")

[[ -n "$MODEL" ]] || die "--model-path is required"

IFS=',' read -r -a TASKS <<< "${BENCHMARKS_CSV}"
if [[ ${#TASKS[@]} -eq 1 && -z "${TASKS[0]}" ]]; then
  TASKS=(bamboogle hotpotqa 2wiki musique)
fi

# Partition requested tasks: vitabench domains go to `vita run`, everything
# else stays on the BeliefTracer `bcg.agent.rollout` pipeline. `vitabench` (bare) is
# shorthand for all three domains. `vitabench:<domain>` selects just that one.
VITA_DOMAINS=()
RLLM_TASKS=()
for t in "${TASKS[@]}"; do
  case "$t" in
    vitabench)
      VITA_DOMAINS+=(delivery ota instore)
      ;;
    vitabench:delivery|vitabench:ota|vitabench:instore)
      VITA_DOMAINS+=("${t#vitabench:}")
      ;;
    vitabench:*)
      die "unknown vitabench domain '$t' (expected vitabench:delivery, vitabench:ota, or vitabench:instore)"
      ;;
    *)
      RLLM_TASKS+=("$t")
      ;;
  esac
done

TOOLS_ARR=()
if [[ -n "${TOOLS_CSV}" && "${TOOLS_CSV,,}" != "none" ]]; then
  IFS=',' read -r -a TOOLS_ARR <<< "${TOOLS_CSV}"
fi

# Auto-select the averitec system prompt when averitec_search is used and no
# explicit --prompt was given. The choice follows --hyde/--no-hyde so the system
# prompt's query guidance matches the actual query format the tool expects:
#   --hyde    -> bcg/agent/prompts/averitec.txt        (teaches the "claim ||| passage" format)
#   --no-hyde -> bcg/agent/prompts/averitec_nohyde.txt  (teaches complete-sentence queries)
if [[ -z "$PROMPT_FILE" && " ${TOOLS_ARR[*]} " == *" averitec_search "* ]]; then
  if [[ "$HYDE" == "0" ]]; then
    PROMPT_FILE="bcg/agent/prompts/averitec_nohyde.txt"
  else
    PROMPT_FILE="bcg/agent/prompts/averitec.txt"
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROLLOUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROLLOUT_DIR"

export PYTHONPATH="${ROLLOUT_DIR}/rllm:${ROLLOUT_DIR}:${REPO_ROOT}:${REPO_ROOT}/rllm:${PYTHONPATH:-}"
export RETRIEVAL_SERVER_URL
export RLLM_RETRIEVAL_SUMMARIZE

# Activate conda env for the rollout python process.
ROLLOUT_CONDA_ENV="${ROLLOUT_CONDA_ENV:-${CONDA_ENV:-qwen3}}"
CONDA_SH_ROLLOUT=""
for cand in \
    "$HOME/app/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  [[ -f "$cand" ]] && CONDA_SH_ROLLOUT="$cand" && break
done
if [[ -n "$CONDA_SH_ROLLOUT" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH_ROLLOUT"
  set +eu
  conda activate "$ROLLOUT_CONDA_ENV"
  set -eu
fi

BACKEND_ARGS=(--backend "$BACKEND")

case "$BACKEND" in
  vllm)
    BACKEND_ARGS+=(
      --tensor-parallel-size "$VLLM_TP"
      --gpu-memory-utilization "${VLLM_MEM_FRACTION:-0.6}"
    )
    ;;
  ray_vllm)
    BACKEND_ARGS+=(
      --tensor-parallel-size "$VLLM_TP"
      --gpu-memory-utilization "${VLLM_MEM_FRACTION:-0.9}"
      --ray-num-replicas "$RAY_REPLICAS"
    )
    [[ -n "$RAY_ADDRESS" ]] && BACKEND_ARGS+=(--ray-address "$RAY_ADDRESS")
    ;;
  openai|api)
    BACKEND_ARGS+=(
      --base-url "${OPENAI_BASE_URL:-http://localhost:8001/v1}"
      --api-key "${OPENAI_API_KEY:-EMPTY}"
    )
    ;;
  *)
    die "unknown --backend '$BACKEND' (expected vllm, ray_vllm, openai, or api)"
    ;;
esac

if [[ "$BACKEND" == "vllm" || "$BACKEND" == "ray_vllm" ]]; then
  [[ -n "$MAX_MODEL_LEN" ]] && BACKEND_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  [[ -n "$VLLM_DTYPE" ]] && BACKEND_ARGS+=(--vllm-dtype "$VLLM_DTYPE")
  [[ "$VLLM_ENFORCE_EAGER" == "1" ]] && BACKEND_ARGS+=(--vllm-enforce-eager)
fi

if [[ "$SHUFFLE" == "1" ]]; then
  SHUFFLE_FLAG=(--shuffle)
else
  SHUFFLE_FLAG=(--no-shuffle)
fi

EXTRA_ARGS=()
# Benchmarks / rollout scale
[[ -n "$MAX_PROBLEMS" ]] && EXTRA_ARGS+=(--max-problems "$MAX_PROBLEMS")
[[ "$MIXED_ROLLOUTS" == "1" ]] && EXTRA_ARGS+=(--mixed-rollouts)

# Agent behavior
[[ "$ENABLE_THINKING" == "1" ]] && EXTRA_ARGS+=(--enable-thinking)
[[ -n "$PROMPT_FILE" ]] && EXTRA_ARGS+=(--prompt "$PROMPT_FILE")
[[ "$NO_SYSTEM_PROMPT" == "1" ]] && EXTRA_ARGS+=(--no-system-prompt)
[[ -n "$PARSER_NAME" ]] && EXTRA_ARGS+=(--parser-name "$PARSER_NAME")

# Sampling
[[ -n "$TEMPERATURE" ]] && EXTRA_ARGS+=(--temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && EXTRA_ARGS+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]] && EXTRA_ARGS+=(--top-k "$TOP_K")

# Output / logging
[[ -n "$SAVE_ALIAS" ]] && EXTRA_ARGS+=(--save-alias "$SAVE_ALIAS")
[[ "$OVERWRITE" == "1" ]] && EXTRA_ARGS+=(--overwrite)
[[ ${#TASK_IDS[@]} -gt 0 ]] && EXTRA_ARGS+=(--task-ids "${TASK_IDS[@]}")
[[ ${#EXCLUDE_IDS[@]} -gt 0 ]] && EXTRA_ARGS+=(--exclude-ids "${EXCLUDE_IDS[@]}")

# Belief Graph
[[ -n "$BELIEF_GRAPH_URL" ]] && EXTRA_ARGS+=(--belief-graph-url "$BELIEF_GRAPH_URL" --belief-graph-timeout "$BELIEF_GRAPH_TIMEOUT")
EXTRA_ARGS+=(--belief-graph-mode "$BELIEF_GRAPH_MODE")
EXTRA_ARGS+=(--graph-format "$GRAPH_FORMAT")
EXTRA_ARGS+=(--belief-graph-placement "$BELIEF_GRAPH_PLACEMENT")
[[ "${NO_GRAPH_RELATIONS:-0}" == "1" ]] && EXTRA_ARGS+=(--no-graph-relations)

# Four-stage retrieval (hero4) — harmless for other methods (python ignores them).
EXTRA_ARGS+=(
  --stage1-bm25-k "$STAGE1_BM25_K"
  --stage2-embed-k "$STAGE2_EMBED_K"
  --stage3-rerank-k "$STAGE3_RERANK_K"
  --rerank-url "$RERANK_URL"
  --rerank-model "$RERANK_MODEL"
  --judge-max-workers "$JUDGE_MAX_WORKERS"
  --judge-max-items "$JUDGE_MAX_ITEMS"
)
[[ "$ENABLE_JUDGE" == "0" ]] && EXTRA_ARGS+=(--no-judge)
[[ -n "$JUDGE_MODEL" ]] && EXTRA_ARGS+=(--judge-model "$JUDGE_MODEL")
[[ -n "$JUDGE_BASE_URL" ]] && EXTRA_ARGS+=(--judge-base-url "$JUDGE_BASE_URL")
[[ -n "$JUDGE_API_KEY" ]] && EXTRA_ARGS+=(--judge-api-key "$JUDGE_API_KEY")

# Sandboxed file-read tool + two-layer archive
[[ "$ENABLE_ARCHIVE" == "1" ]] && EXTRA_ARGS+=(--enable-archive)
[[ "$ENABLE_FILE_READ" == "1" ]] && EXTRA_ARGS+=(--enable-file-read)
[[ -n "$FILE_TOOL_ROOT" ]] && EXTRA_ARGS+=(--file-tool-root "$FILE_TOOL_ROOT")
[[ -n "$RECENT_TURNS" ]] && EXTRA_ARGS+=(--recent-turns "$RECENT_TURNS")

# Context-memory baselines
[[ -n "$CONTEXT_MEMORY_MODE" ]] && EXTRA_ARGS+=(--context-memory-mode "$CONTEXT_MEMORY_MODE")
[[ -n "$CONTEXT_MEMORY_SUMMARIZER" ]] && EXTRA_ARGS+=(--context-memory-summarizer "$CONTEXT_MEMORY_SUMMARIZER")
[[ -n "$CONTEXT_MEMORY_RECENT_OBSERVATIONS" ]] && EXTRA_ARGS+=(--context-memory-recent-observations "$CONTEXT_MEMORY_RECENT_OBSERVATIONS")
[[ -n "$CONTEXT_MEMORY_TAIL_TURNS" ]] && EXTRA_ARGS+=(--context-memory-tail-turns "$CONTEXT_MEMORY_TAIL_TURNS")
[[ -n "$CONTEXT_MEMORY_MAX_CHARS" ]] && EXTRA_ARGS+=(--context-memory-max-chars "$CONTEXT_MEMORY_MAX_CHARS")
[[ -n "$CONTEXT_MEMORY_TOOL_SUMMARY_CHARS" ]] && EXTRA_ARGS+=(--context-memory-tool-summary-chars "$CONTEXT_MEMORY_TOOL_SUMMARY_CHARS")
[[ -n "$CONTEXT_MEMORY_INTERVAL" ]] && EXTRA_ARGS+=(--context-memory-interval "$CONTEXT_MEMORY_INTERVAL")
[[ -n "$CONTEXT_MEMORY_SUMMARIZER_MAX_TOKENS" ]] && EXTRA_ARGS+=(--context-memory-summarizer-max-tokens "$CONTEXT_MEMORY_SUMMARIZER_MAX_TOKENS")
[[ -n "$CONTEXT_MEMORY_SUMMARIZER_TIMEOUT" ]] && EXTRA_ARGS+=(--context-memory-summarizer-timeout "$CONTEXT_MEMORY_SUMMARIZER_TIMEOUT")
[[ -n "$CONTEXT_MEMORY_SUMMARIZER_FAILURE_LIMIT" ]] && EXTRA_ARGS+=(--context-memory-summarizer-failure-limit "$CONTEXT_MEMORY_SUMMARIZER_FAILURE_LIMIT")
[[ -n "$CONTEXT_MEMORY_LOG_PREVIEW_CHARS" ]] && EXTRA_ARGS+=(--context-memory-log-preview-chars "$CONTEXT_MEMORY_LOG_PREVIEW_CHARS")

PY_CMD=(
  python -m bcg.agent run
  --model "$MODEL"
  "${BACKEND_ARGS[@]}"
  --tasks "${RLLM_TASKS[@]}"
  --retrieval-server-url "$RETRIEVAL_SERVER_URL"
  --retrieval-max-results "$RETRIEVAL_MAX_RESULTS"
  --retrieval-method "$RETRIEVAL_METHOD"
  --serper-endpoint "$SERPER_ENDPOINT"
  --serper-country "$SERPER_COUNTRY"
  --serper-language "$SERPER_LANGUAGE"
  --serper-timeout "$SERPER_TIMEOUT"
  --serper-max-output-chars "$SERPER_MAX_OUTPUT_CHARS"
  --browsecomp-grader-timeout "$BROWSECOMP_GRADER_TIMEOUT"
  --browsecomp-grader-max-tokens "$BROWSECOMP_GRADER_MAX_TOKENS"
  --browsecomp-grader-max-retries "$BROWSECOMP_GRADER_MAX_RETRIES"
  --hero-bm25-top-k "$HERO_BM25_TOP_K"
  --hero-embedding-model "$HERO_EMBEDDING_MODEL"
  --hero-embedding-url "$HERO_EMBEDDING_URL"
  --hero-embedding-device "$HERO_EMBEDDING_DEVICE"
  --hero-batch-size "$HERO_BATCH_SIZE"
  --num-samples "$NUM_SAMPLES"
  --passk "$PASSK"
  --max-steps "$MAX_STEPS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-response-length "$MAX_RESPONSE_LENGTH"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --n-parallel-tasks "$N_PARALLEL"
  --output-dir "$OUTPUT_DIR"
  "${SHUFFLE_FLAG[@]}"
  --shuffle-seed "$SHUFFLE_SEED"
  "${EXTRA_ARGS[@]}"
)

[[ -n "$BROWSECOMP_GRADER_MODEL" ]] && PY_CMD+=(--browsecomp-grader-model "$BROWSECOMP_GRADER_MODEL")
[[ -n "$BROWSECOMP_GRADER_BASE_URL" ]] && PY_CMD+=(--browsecomp-grader-base-url "$BROWSECOMP_GRADER_BASE_URL")

if [[ ${#TOOLS_ARR[@]} -gt 0 ]]; then
  PY_CMD+=(--tools "${TOOLS_ARR[@]}")
fi

if [[ "$HYDE" == "0" ]]; then
  PY_CMD+=(--no-hyde)
fi

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

if [[ ${#RLLM_TASKS[@]} -gt 0 ]]; then
  if [[ "$VERBOSE" == "1" ]]; then
    export LOGLEVEL="${LOGLEVEL:-INFO}"
    "${PY_CMD[@]}"
  else
    LOG_FILE="$(mktemp -t rollout.XXXXXX.log)"
    echo "[rollout] Running quietly; full log: $LOG_FILE" >&2
    set +e
    "${PY_CMD[@]}" >"$LOG_FILE" 2> >(tee -a "$LOG_FILE" >&2)
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[rollout] Run failed (exit $rc). Dumping log:" >&2
      cat "$LOG_FILE" >&2
      exit "$rc"
    fi
    awk '/Agent rollout summary/ {found=1} found {print}' "$LOG_FILE"
  fi
elif [[ ${#VITA_DOMAINS[@]} -eq 0 ]]; then
  die "no benchmarks to run (after stripping vitabench domains the task list was empty)"
fi

# ---------------------------------------------------------------------------
# Vitabench dispatch
# ---------------------------------------------------------------------------

if [[ ${#VITA_DOMAINS[@]} -gt 0 ]]; then
  VITA_REPO="$ROLLOUT_DIR/benchmarks/vitabench"
  [[ -d "$VITA_REPO" ]] || die "vitabench repo not found at $VITA_REPO (did you run pip install -e benchmarks/vitabench in the '$VITA_CONDA_ENV' env?)"

  case "$BACKEND" in
    openai)
      VITA_ASSISTANT_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
      VITA_ASSISTANT_API_KEY="${OPENAI_API_KEY:-EMPTY}"
      VITA_ASSISTANT_MODE="litellm_http"
      ;;
    vllm|ray_vllm)
      VITA_ASSISTANT_MODE="vllm_local"
      ;;
    *)
      die "unsupported --backend '$BACKEND' for vitabench (use vllm, ray_vllm, or openai)"
      ;;
  esac

  # Resolve save-prefix: explicit flag > derived from --save-alias > 'vita'.
  if [[ -n "$VITA_SAVE_PREFIX" ]]; then
    VITA_PREFIX="$VITA_SAVE_PREFIX"
  elif [[ -n "$SAVE_ALIAS" ]]; then
    VITA_PREFIX="$SAVE_ALIAS"
  else
    VITA_PREFIX="vita"
  fi

  # Vitabench requires a YAML file, so create a private, ephemeral adapter
  # from root .env values and remove it automatically when this script exits.
  VITA_CONFIG="$(mktemp "${TMPDIR:-/tmp}/bcg-vita-models.${VITA_PREFIX}.XXXXXX.yaml")"
  chmod 600 "$VITA_CONFIG"
  trap 'rm -f "${VITA_CONFIG:-}"' EXIT

  {
    echo "# Auto-generated by scripts/rollout.sh for --save-alias=${VITA_PREFIX}."
    echo "# Assistant: ${VITA_ASSISTANT_MODE} on ${MODEL}"
    echo "# User/Evaluator: ${VITA_USER_MODEL}/${VITA_EVALUATOR_MODEL} via ${VITA_LITELLM_BASE_URL}"
    echo ""
    echo "default:"
    echo "  backend: litellm_http"
    echo "  api_key: ${VITA_LITELLM_API_KEY}"
    echo "  base_url: ${VITA_LITELLM_BASE_URL}"
    echo "  temperature: 0.0"
    echo "  max_tokens: 4096"
    echo "  max_input_tokens: 32768"
    echo "  headers:"
    echo "    Authorization: \"Bearer \""
    echo "    Content-Type: \"application/json\""
    echo ""
    echo "models:"
    echo "  - name: assistant-agent"
    if [[ "$VITA_ASSISTANT_MODE" == "litellm_http" ]]; then
      echo "    backend: litellm_http"
      echo "    base_url: ${VITA_ASSISTANT_BASE_URL}"
      echo "    api_key: ${VITA_ASSISTANT_API_KEY}"
      echo "    model: ${MODEL}"
    else
      echo "    backend: vllm_local"
      echo "    model: ${MODEL}"
      echo "    vllm:"
      echo "      tensor_parallel_size: ${VLLM_TP}"
      echo "      gpu_memory_utilization: ${VLLM_MEM_FRACTION:-0.6}"
      [[ -n "$MAX_MODEL_LEN" ]] && echo "      max_model_len: ${MAX_MODEL_LEN}"
      [[ -n "$VLLM_DTYPE" ]] && echo "      dtype: ${VLLM_DTYPE}"
      [[ "$VLLM_ENFORCE_EAGER" == "1" ]] && echo "      enforce_eager: true"
    fi
    echo ""
    echo "  - name: user-agent"
    echo "    model: ${VITA_USER_MODEL}"
    echo ""
    echo "  - name: evaluator-agent"
    echo "    model: ${VITA_EVALUATOR_MODEL}"
  } > "$VITA_CONFIG"

  echo "[rollout] Wrote vitabench model config: $VITA_CONFIG" >&2

  # Activate the vita conda env without depending on the caller's shell state.
  CONDA_SH=""
  for cand in \
      "$HOME/app/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh" \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [[ -f "$cand" ]] && CONDA_SH="$cand" && break
  done
  [[ -n "$CONDA_SH" ]] || die "could not locate conda.sh; set CONDA_EXE or activate '$VITA_CONDA_ENV' before calling rollout.sh"

  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$VITA_CONDA_ENV" || die "failed to activate conda env '$VITA_CONDA_ENV'"
  command -v vita >/dev/null 2>&1 || die "'vita' CLI not found in env '$VITA_CONDA_ENV' (pip install -e benchmarks/vitabench)"

  # The litellm gateway is internal — avoid forcing outbound traffic through
  # whatever proxy the caller's shell had configured.
  unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY

  export VITA_MODEL_CONFIG_PATH="$VITA_CONFIG"

  # Allow `--overwrite` to force a fresh save file (vita prompts interactively
  # when the file already exists and otherwise requires a config-hash match).
  LOGDIR="$ROLLOUT_DIR/rollout_logs"
  mkdir -p "$LOGDIR"

  # Resolve VITA_MAX_CONCURRENCY into a per-domain lookup.
  # - Scalar (e.g. "32") applies to every domain.
  # - Map (e.g. "delivery=32,ota=16,instore=8") sets per-domain values; a
  #   "default=<N>" entry covers unlisted domains (falls back to 1 if absent).
  declare -A VITA_CONC_MAP=()
  VITA_CONC_DEFAULT=1
  if [[ "$VITA_MAX_CONCURRENCY" == *=* ]]; then
    IFS=',' read -r -a _vconc_pairs <<< "$VITA_MAX_CONCURRENCY"
    for _kv in "${_vconc_pairs[@]}"; do
      [[ -z "$_kv" ]] && continue
      _k="${_kv%%=*}"; _v="${_kv#*=}"
      [[ "$_v" =~ ^[0-9]+$ ]] || die "--vita-max-concurrency: non-integer value for '$_k' in '$VITA_MAX_CONCURRENCY'"
      if [[ "$_k" == "default" ]]; then
        VITA_CONC_DEFAULT="$_v"
      else
        VITA_CONC_MAP["$_k"]="$_v"
      fi
    done
  else
    [[ "$VITA_MAX_CONCURRENCY" =~ ^[0-9]+$ ]] || die "--vita-max-concurrency must be an int or key=val map, got '$VITA_MAX_CONCURRENCY'"
    VITA_CONC_DEFAULT="$VITA_MAX_CONCURRENCY"
  fi

  pushd "$VITA_REPO" >/dev/null

  VITA_STATUSES=()
  for d in "${VITA_DOMAINS[@]}"; do
    save_to="${VITA_PREFIX}_${d}"
    save_path="$VITA_REPO/data/simulations/${save_to}"
    if [[ "$OVERWRITE" == "1" && -f "$save_path" ]]; then
      rm -f "$save_path"
      echo "[rollout] --overwrite: removed existing $save_path" >&2
    fi
    log_file="$LOGDIR/${save_to}.log"
    conc="${VITA_CONC_MAP[$d]:-$VITA_CONC_DEFAULT}"
    echo "[rollout] vita run --domain=$d max-concurrency=$conc save-to=$save_to (log: $log_file)" >&2

    set +e
    yes | vita run \
      --domain "$d" \
      --user-llm user-agent \
      --agent-llm assistant-agent \
      --evaluator-llm evaluator-agent \
      --max-steps "$VITA_MAX_STEPS" \
      --max-concurrency "$conc" \
      --language "$VITA_LANGUAGE" \
      --save-to "$save_to" \
      --log-level INFO \
      >"$log_file" 2>&1
    rc=$?
    set -e
    VITA_STATUSES+=("$d=$rc")
    if [[ $rc -ne 0 ]]; then
      echo "[rollout] vita run for domain '$d' exited with status $rc (see $log_file)" >&2
    fi
  done

  popd >/dev/null

  # Summarise vitabench results: avg reward per domain.
  echo ""
  echo "=== Vitabench summary (${VITA_PREFIX}) ==="
  for d in "${VITA_DOMAINS[@]}"; do
    save_path="$VITA_REPO/data/simulations/${VITA_PREFIX}_${d}"
    if [[ -f "$save_path" ]]; then
      python - "$save_path" "$d" <<'PY'
import json, sys
path, domain = sys.argv[1], sys.argv[2]
with open(path) as fp:
    data = json.load(fp)
sims = data.get("simulations", [])
rewards = [s.get("reward_info", {}).get("reward") for s in sims]
rewards = [r for r in rewards if r is not None]
avg = sum(rewards) / len(rewards) if rewards else 0.0
print(f"{domain:>10s}: n={len(sims):3d} scored={len(rewards):3d} avg_reward={avg:.3f}")
PY
    else
      echo "${d:>10s}: (no output at $save_path)"
    fi
  done

  # Fail the whole run if any domain returned non-zero.
  for entry in "${VITA_STATUSES[@]}"; do
    rc="${entry##*=}"
    [[ "$rc" == "0" ]] || exit "$rc"
  done
fi
