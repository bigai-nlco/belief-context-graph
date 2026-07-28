#!/usr/bin/env bash

# Compare pure Agent and DeepSeek-V4 Markdown belief payloads on 100
# BrowseComp tasks and all 100 text-only GAIA validation tasks.

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
OUTPUT_ROOT="${BCG_COMPARE_OUTPUT_ROOT:-artifacts/agent_vs_markdown_100}"

if [[ ! -x "$BCG_PYTHON" ]]; then
  printf 'Python executable not found: %s\n' "$BCG_PYTHON" >&2
  exit 127
fi

curl -fsS --max-time 10 "${GRAPH_URL%/}/health" >/dev/null
export GAIA_SPLIT=validation

# GAIA validation contains 165 tasks. Keep only tasks without attachments and
# whose official annotator tool list does not require image/video/audio/OCR/GIF
# processing. One additional task based on animated-video content is excluded
# explicitly because its annotation omitted the media tool from the tool list.
mapfile -t gaia_text_ids < <(
  "$BCG_PYTHON" - "$REPO_ROOT/datasets/gaia/2023/validation/metadata.jsonl" <<'PY'
import json
import re
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
media_tools = re.compile(
    r"video|youtube|image|audio|ocr|color recognition|gif|photo|visual|speech",
    re.IGNORECASE,
)
manual_media_tasks = {"d700d50d-c707-4dca-90dc-4528cddd0c80"}

task_ids = []
for line in metadata_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if str(row.get("file_name", "")).strip():
        continue
    task_id = str(row["task_id"])
    tools = str((row.get("Annotator Metadata") or {}).get("Tools", ""))
    if media_tools.search(tools) or task_id in manual_media_tasks:
        continue
    task_ids.append(task_id)

if len(task_ids) != 100 or len(set(task_ids)) != 100:
    raise SystemExit(
        f"Expected exactly 100 unique text-only GAIA tasks, got {len(task_ids)}"
    )
print("\n".join(task_ids))
PY
)

common_args=(
  --model "$MODEL_ID"
  --backend api
  --benchmarks-dir "$REPO_ROOT/datasets"
  --tools serper_search serper_scrape
  --enable-archive
  --recent-turns 2
  --max-steps 12
  --num-samples 1
  --n-parallel-tasks 4
  --no-auto-ui
)

run_agent() {
  local benchmark="$1"
  local selection_args=(--max-problems 100 --shuffle --shuffle-seed 42)
  if [[ "$benchmark" == "gaia" ]]; then
    selection_args=(--task-ids "${gaia_text_ids[@]}" --no-shuffle)
  fi
  "$BCG_PYTHON" -m bcg.cli agent run \
    "${common_args[@]}" \
    --tasks "$benchmark" \
    "${selection_args[@]}" \
    --context-memory-mode none \
    --belief-graph-mode none \
    --belief-graph-url "" \
    --output-dir "$OUTPUT_ROOT/${benchmark}_agent"
}

run_markdown() {
  local benchmark="$1"
  local selection_args=(--max-problems 100 --shuffle --shuffle-seed 42)
  if [[ "$benchmark" == "gaia" ]]; then
    selection_args=(--task-ids "${gaia_text_ids[@]}" --no-shuffle)
  fi
  "$BCG_PYTHON" -m bcg.cli agent run \
    "${common_args[@]}" \
    --tasks "$benchmark" \
    "${selection_args[@]}" \
    --context-memory-mode belief_graph \
    --belief-graph-url "$GRAPH_URL" \
    --belief-graph-timeout 900 \
    --belief-graph-mode augment \
    --graph-format deepseek_v4 \
    --deepseek-v4-payload-format markdown \
    --output-dir "$OUTPUT_ROOT/${benchmark}_markdown"
}

# Stable artifact paths plus omitted --overwrite make every group resumable.
# Suites can run in separate tmux sessions to reduce total wall-clock time.
run_agent_suite() {
  run_agent browsecomp
  run_agent gaia
}

run_markdown_suite() {
  run_markdown browsecomp
  run_markdown gaia
}

case "${1:-parallel}" in
  agent)
    run_agent_suite
    ;;
  markdown)
    run_markdown_suite
    ;;
  parallel)
    run_agent_suite &
    agent_pid=$!
    run_markdown_suite &
    markdown_pid=$!
    wait "$agent_pid"
    wait "$markdown_pid"
    ;;
  *)
    printf 'Usage: %s [agent|markdown|parallel]\n' "$0" >&2
    exit 2
    ;;
esac
