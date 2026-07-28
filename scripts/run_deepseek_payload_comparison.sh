#!/usr/bin/env bash

# Resume the fixed 20-task BrowseComp/GAIA DeepSeek-V4 payload comparison.
# Secrets and API endpoints are loaded from the repository-root .env file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="${BCG_ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

BCG_PYTHON="${BCG_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL_ID="${BCG_COMPARE_MODEL:-deepseek-v4-pro-260425}"
GRAPH_URL="${BCG_COMPARE_GRAPH_URL:-http://127.0.0.1:8848}"
OUTPUT_ROOT="${BCG_COMPARE_OUTPUT_ROOT:-artifacts/deepseek_payload_compare}"

if [[ ! -x "$BCG_PYTHON" ]]; then
  printf 'Python executable not found: %s\n' "$BCG_PYTHON" >&2
  exit 127
fi

curl -fsS --max-time 10 "${GRAPH_URL%/}/health" >/dev/null

export GAIA_SPLIT=validation

browsecomp_ids=(
  browsecomp-0776 browsecomp-0108 browsecomp-0498 browsecomp-0099
  browsecomp-1018 browsecomp-0453 browsecomp-0257 browsecomp-0982
  browsecomp-0763 browsecomp-0435 browsecomp-0320 browsecomp-0665
  browsecomp-0316 browsecomp-0192 browsecomp-0050 browsecomp-0419
  browsecomp-0444 browsecomp-0287 browsecomp-1205 browsecomp-0123
)

gaia_ids=(
  edd4d4f2-1a58-45c4-b038-67337af4e029
  42d4198c-5895-4f0a-b0c0-424a66465d83
  853c8244-429e-46ca-89f2-addf40dfb2bd
  114d5fd0-e2ae-4b6d-a65a-870da2d19c08
  d1af70ea-a9a4-421a-b9cc-94b5e02f1788
  50f58759-7bd6-406f-9b0d-5692beb2a926
  a7feb290-76bb-4cb7-8800-7edaf7954f2f
  9d191bce-651d-4746-be2d-7ef8ecadb9c2
  54612da3-fd56-4941-80f4-5eb82330de25
  20194330-9976-4043-8632-f8485c6c71b2
  ded28325-3447-4c56-860f-e497d6fb3577
  6b078778-0b90-464d-83f6-59511c811b01
  72e110e7-464c-453c-a309-90a95aed6538
  0b260a57-3f3a-4405-9f29-6d7a1012dbfb
  65da0822-a48a-4a68-bbad-8ed1b835a834
  da52d699-e8d2-4dc5-9191-a2199e0b6a9b
  b9763138-c053-4832-9f55-86200cb1f99c
  08cae58d-4084-4616-b6dd-dd6534e4825b
  8b3379c0-0981-4f5b-8407-6444610cb212
  5a0c1adf-205e-4841-a666-7c3ef95def9d
)

common_args=(
  --model "$MODEL_ID"
  --backend api
  --benchmarks-dir "$REPO_ROOT/datasets"
  --no-shuffle
  --tools serper_search serper_scrape
  --enable-archive
  --recent-turns 2
  --max-steps 12
  --num-samples 1
  --n-parallel-tasks 4
  --no-auto-ui
)

run_graph() {
  local benchmark="$1"
  local payload_format="$2"
  local output_dir="$3"
  local start_mode="$4"
  shift 4
  local overwrite_args=()
  if [[ "$start_mode" == "fresh" ]]; then
    overwrite_args=(--overwrite)
  fi

  "$BCG_PYTHON" -m bcg.cli agent run \
    "${common_args[@]}" \
    --tasks "$benchmark" \
    --task-ids "$@" \
    --context-memory-mode belief_graph \
    --belief-graph-url "$GRAPH_URL" \
    --belief-graph-timeout 900 \
    --belief-graph-mode augment \
    --graph-format deepseek_v4 \
    --deepseek-v4-payload-format "$payload_format" \
    "${overwrite_args[@]}" \
    --output-dir "$output_dir"
}

run_agent() {
  local benchmark="$1"
  local output_dir="$2"
  local start_mode="$3"
  shift 3
  local overwrite_args=()
  if [[ "$start_mode" == "fresh" ]]; then
    overwrite_args=(--overwrite)
  fi

  "$BCG_PYTHON" -m bcg.cli agent run \
    "${common_args[@]}" \
    --tasks "$benchmark" \
    --task-ids "$@" \
    --context-memory-mode none \
    --belief-graph-mode none \
    --belief-graph-url "" \
    "${overwrite_args[@]}" \
    --output-dir "$output_dir"
}

# BrowseComp XML resumes the 13 valid trajectories from the interrupted run.
# GAIA XML is restarted because its first attempt ran while the graph service
# was unhealthy. The remaining output directories have not been run yet.
run_graph browsecomp xml "$OUTPUT_ROOT/browsecomp_xml" resume "${browsecomp_ids[@]}"
run_graph gaia xml "$OUTPUT_ROOT/gaia_xml" fresh "${gaia_ids[@]}"
run_graph browsecomp markdown "$OUTPUT_ROOT/browsecomp_markdown" fresh "${browsecomp_ids[@]}"
run_graph gaia markdown "$OUTPUT_ROOT/gaia_markdown" fresh "${gaia_ids[@]}"
run_agent browsecomp "$OUTPUT_ROOT/browsecomp_agent" fresh "${browsecomp_ids[@]}"
run_agent gaia "$OUTPUT_ROOT/gaia_agent" fresh "${gaia_ids[@]}"
