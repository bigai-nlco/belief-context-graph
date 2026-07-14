#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Run full AVeriTeC dev set (500 problems) + start UI

conda run --no-capture-output -n qwen3 \
  python -m bcg.agent run \
  --model /data/user/fukeshu/model/Qwen3-8B \
  --backend openai \
  --base-url http://127.0.0.1:8001/v1 \
  --api-key EMPTY \
  --tasks averitec \
  --prompt bcg/agent/prompts/averitec.txt \
  --tools averitec_search \
  --max-problems 500 \
  --num-samples 1 \
  --passk 1 \
  --n-parallel-tasks 32 \
  --max-new-tokens 32768 \
  --max-response-length 32768 \
  --max-steps 96 \
  --enable-thinking \
  --parser-name qwen \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --output-dir output \
  --save-alias dev_65k \
  --overwrite \
  --no-shuffle \
  --shuffle-seed 0 \
  --retrieval-server-url http://10.2.152.50:65432 \
  --retrieval-max-results 10

# Start UI after rollout completes
conda run --no-capture-output -n qwen3 ./scripts/start_ui.sh \
  --host 0.0.0.0 \
  --port 8002 \
  --artifacts-dir output
