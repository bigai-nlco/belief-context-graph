#!/usr/bin/env bash
# Start vLLM from the uv-managed project environment.
#
# Usage:
#   bash scripts/start_vllm.sh                          # default 65K context
#   bash scripts/start_vllm.sh --max-model-len 32768    # custom context length
#   bash scripts/start_vllm.sh --gpu 0                  # specify GPU
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

bcg_load_root_env

# Defaults
MODEL="${VLLM_MODEL:-}"
PORT="${VLLM_PORT:-8001}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
GPU="${VLLM_VISIBLE_GPUS:-0}"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2"; shift 2 ;;
    --port)            PORT="$2"; shift 2 ;;
    --host)            HOST="$2"; shift 2 ;;
    --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
    --gpu)             GPU="$2"; shift 2 ;;
    --vllm-bin)        VLLM_BIN="$2"; shift 2 ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--model PATH] [--port N] [--max-model-len N] [--gpu N] [--gpu-mem-util F] [--vllm-bin PATH] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

bcg_validate_port VLLM_PORT "$PORT"
bcg_require_env VLLM_MODEL

echo "[start_vllm] Model:         $MODEL"
echo "[start_vllm] Port:          $PORT"
echo "[start_vllm] Max model len: $MAX_MODEL_LEN"
echo "[start_vllm] GPU:           $GPU"
echo "[start_vllm] GPU mem util:  $GPU_MEM_UTIL"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU"

CMD=("$VLLM_BIN" serve "$MODEL" \
  --port "$PORT" \
  --host "$HOST" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL")

bcg_maybe_dry_run "${CMD[@]}"

[[ -x "$VLLM_BIN" ]] || {
  printf 'vLLM executable not found at %s. Run `uv sync --all-groups` then `uv pip install --python .venv/bin/python vllm`.\n' "$VLLM_BIN" >&2
  exit 127
}

exec "${CMD[@]}"
