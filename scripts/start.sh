#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Load the same root configuration used by the Python package. Export every
# assignment so child processes and rLLM can read the values as well.
ENV_FILE="${BCG_ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

command -v bcg >/dev/null 2>&1 || {
  echo "error: bcg command not found; install it with uv tool or activate .venv" >&2
  exit 127
}

MODEL_ID="${MODEL:-${OPENAI_MODEL:-}}"
if [[ -z "$MODEL_ID" ]]; then
  echo "error: set MODEL or OPENAI_MODEL in ${ENV_FILE}" >&2
  exit 2
fi

export AVERITEC_DATA_FILE="${AVERITEC_DATA_FILE:-dev_subset10.json}"
export LOGLEVEL="${LOGLEVEL:-INFO}"

BELIEF_GRAPH_URL="${BELIEF_GRAPH_URL:-http://127.0.0.1:8848}"
HERO_EMBEDDING_MODEL="${HERO_EMBEDDING_MODEL:-SFR-Embedding-2_R}"
HERO_EMBEDDING_URL="${HERO_EMBEDDING_URL:-}"
RERANK_URL="${RERANK_URL:-http://127.0.0.1:8010}"
RERANK_MODEL="${RERANK_MODEL:-Qwen3-Reranker-0.6B}"

BCG_ARGS=(
  agent run averitec
  --model "$MODEL_ID"
  --backend api
  --tools averitec_search
  --retrieval-method hero4
  --retrieval-max-results 10
  --hero-bm25-top-k 10
  --hero-embedding-model "$HERO_EMBEDDING_MODEL"
  --hero-embedding-url "$HERO_EMBEDDING_URL"
  --hero-embedding-device cpu
  --hero-batch-size 16
  --stage1-bm25-k 1000
  --stage2-embed-k 32
  --stage3-rerank-k 5
  --rerank-url "$RERANK_URL"
  --rerank-model "$RERANK_MODEL"
  --judge-max-workers 10
  --judge-max-items 10
  --enable-archive
  --recent-turns 2
  --belief-graph-url "$BELIEF_GRAPH_URL"
  --belief-graph-timeout 600
  --belief-graph-mode augment
  --graph-format deepseek_v4
  --belief-graph-placement system
  --file-tool-root ai_workspace
  --max-problems 10
  --n-parallel-tasks 1
  --enable-thinking
  --temperature 0.6
  --top-p 0.95
  --top-k 20
  --num-samples 1
  --passk 1
  --max-steps 12
  --max-new-tokens 32768
  --max-response-length 32768
  --max-prompt-length 32768
  --prompt bcg/agent/prompts/averitec_nohyde.txt
  --no-hyde
  --no-shuffle
  --shuffle-seed 0
  --output-dir output
  --save-alias augment_dsv4_system
  --overwrite
)

[[ -n "${OPENAI_BASE_URL:-}" ]] && BCG_ARGS+=(--base-url "$OPENAI_BASE_URL")
[[ -n "${JUDGE_MODEL:-}" ]] && BCG_ARGS+=(--judge-model "$JUDGE_MODEL")
[[ -n "${JUDGE_BASE_URL:-}" ]] && BCG_ARGS+=(--judge-base-url "$JUDGE_BASE_URL")

# Extra arguments are appended last, allowing one-off CLI overrides without
# editing this preset script.
bcg "${BCG_ARGS[@]}" "$@"

echo "Finished bcg agent run"
