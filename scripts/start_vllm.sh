#!/usr/bin/env bash
# Start vLLM serving Qwen3-8B on port 8001.
#
# Usage:
#   bash scripts/start_vllm.sh                          # default 65K context
#   bash scripts/start_vllm.sh --max-model-len 32768    # custom context length
#   bash scripts/start_vllm.sh --gpu 0                  # specify GPU
#
set -euo pipefail

# Defaults
MODEL="${MODEL:-/data/user/fukeshu/model/Qwen3-8B}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
GPU="${GPU:-0}"
CONDA_ENV="${CONDA_ENV:-qwen3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2"; shift 2 ;;
    --port)            PORT="$2"; shift 2 ;;
    --host)            HOST="$2"; shift 2 ;;
    --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
    --gpu)             GPU="$2"; shift 2 ;;
    --conda-env)       CONDA_ENV="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model PATH] [--port N] [--max-model-len N] [--gpu N] [--gpu-mem-util F]"
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "[start_vllm] Model:         $MODEL"
echo "[start_vllm] Port:          $PORT"
echo "[start_vllm] Max model len: $MAX_MODEL_LEN"
echo "[start_vllm] GPU:           $GPU"
echo "[start_vllm] GPU mem util:  $GPU_MEM_UTIL"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU"

exec conda run --no-capture-output -n "$CONDA_ENV" \
  vllm serve "$MODEL" \
  --port "$PORT" \
  --host "$HOST" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL"
