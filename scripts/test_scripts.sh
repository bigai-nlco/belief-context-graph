#!/usr/bin/env bash
# Dry-run golden tests for the BCG service scripts (step 13).
# Renders each script's command with defaults and with overrides, asserting
# the produced command lines. Run via `make check-scripts`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

failures=0

check() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  if [[ "$actual" == *"$expected"* ]]; then
    printf 'ok   - %s\n' "$description"
  else
    printf 'FAIL - %s\n      expected substring: %s\n      got: %s\n' "$description" "$expected" "$actual" >&2
    failures=$((failures + 1))
  fi
}

# --- vllm defaults ---
out="$(VLLM_MODEL=/models/qwen bash scripts/start_vllm.sh --dry-run 2>&1 || true)"
check "vllm: default port"        "--port 8001" "$out"
check "vllm: default max len"     "--max-model-len 65536" "$out"
check "vllm: default gpu util"    "--gpu-memory-utilization 0.88" "$out"
check "vllm: default model"       "serve /models/qwen" "$out"

# --- vllm overrides ---
out="$(VLLM_MODEL=/models/qwen bash scripts/start_vllm.sh --dry-run --port 9999 --max-model-len 32768 --gpu-mem-util 0.5 2>&1 || true)"
check "vllm: override port"       "--port 9999" "$out"
check "vllm: override max len"    "--max-model-len 32768" "$out"
check "vllm: override gpu util"   "--gpu-memory-utilization 0.5" "$out"

# --- vllm invalid port fails fast ---
out="$(VLLM_MODEL=/models/qwen VLLM_PORT=abc bash scripts/start_vllm.sh --dry-run 2>&1 || true)"
check "vllm: invalid port rejected" "Invalid VLLM_PORT: abc" "$out"

# --- sglang defaults and overrides ---
out="$(SGLANG_MODEL=/models/qwen bash scripts/start_sglang_server.sh --dry-run 2>&1 || true)"
check "sglang: default port"      "--port 8003" "$out"
check "sglang: served model name" "--served-model-name /models/qwen" "$out"

out="$(SGLANG_MODEL=/models/qwen bash scripts/start_sglang_server.sh --dry-run --port 7001 --served-model-name Qwen3-8B 2>&1 || true)"
check "sglang: override port"     "--port 7001" "$out"
check "sglang: override name"     "--served-model-name Qwen3-8B" "$out"

# --- tonggraph dry-run renders config injection ---
out="$(
  TONGGRAPH_ADMIN_TOKEN=t TONGGRAPH_AGENT_WRITER_TOKEN=t TONGGRAPH_AGENT_READER_TOKEN=t \
  TONGGRAPH_DATA_DIR=/tmp/tg-data TONGGRAPH_SERVER_BIN=/bin/echo \
  bash scripts/start_tonggraph_server.sh --dry-run 2>&1 || true
)"
check "tonggraph: runtime config consumed" "tonggraph-config.generated.yml" "$out"

# --- tonggraph missing tokens fail fast ---
out="$(bash scripts/start_tonggraph_server.sh --dry-run 2>&1 || true)"
check "tonggraph: missing token rejected" "Missing required TONGGRAPH_ADMIN_TOKEN" "$out"

if ((failures > 0)); then
  printf '%d script test(s) failed.\n' "$failures" >&2
  exit 1
fi
printf 'All script dry-run golden tests passed.\n'
