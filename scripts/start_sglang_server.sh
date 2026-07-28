#!/usr/bin/env bash
# Start an OpenAI-compatible SGLang server for BeliefTracer rollouts.
#
# Server parameters are configured through environment variables. The server exposes:
#   base_url: http://<host>:<port>/v1
#   api key:  none / EMPTY
#
# Usage:
#   scripts/start_sglang_server.sh
#   scripts/start_sglang_server.sh --background
#   scripts/start_sglang_server.sh --status
#   scripts/start_sglang_server.sh --stop
#   scripts/start_sglang_server.sh --gpus 0 --model /path/to/model --served-model-name Qwen3-8B
#   scripts/start_sglang_server.sh -- --chat-template /path/to/template.jinja

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

MODEL="${SGLANG_MODEL:-${VLLM_MODEL:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
HOST="${SGLANG_HOST:-${VLLM_HOST:-0.0.0.0}}"
PORT="${SGLANG_PORT:-${VLLM_PORT:-8003}}"
VISIBLE_GPUS="${SGLANG_VISIBLE_GPUS:-${VLLM_VISIBLE_GPUS:-${CUDA_VISIBLE_DEVICES:-1}}}"
TP="${SGLANG_TP:-${VLLM_TP:-1}}"
DP="${SGLANG_DP:-${SGLANG_DATA_PARALLEL_SIZE:-${VLLM_DP:-${VLLM_DATA_PARALLEL_SIZE:-1}}}}"
NNODES="${SGLANG_NNODES:-1}"
NODE_RANK="${SGLANG_NODE_RANK:-0}"
DIST_INIT_ADDR="${SGLANG_DIST_INIT_ADDR:-}"
MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-${SGLANG_MEM_FRACTION:-${VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_MEM_FRACTION:-0.90}}}}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-}"
ALLOW_LONGER_CONTEXT="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"
DTYPE="${SGLANG_DTYPE:-${VLLM_DTYPE:-auto}}"
TRUST_REMOTE_CODE="${SGLANG_TRUST_REMOTE_CODE:-${VLLM_TRUST_REMOTE_CODE:-1}}"
DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
LOG_FILE="${SGLANG_LOG:-/tmp/belief_tracer-sglang-${USER:-unknown}-${PORT}.log}"
PYTHON="${BCG_PYTHON:-$REPO_ROOT/.venv/bin/python}"
ACTION="start"
BACKGROUND="0"
RESTART="1"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [flags] [-- extra SGLang args]

Flags:
  --model PATH              HF model path/name (default: $MODEL)
  --served-model-name NAME  Model id exposed by /v1/models (default: same as --model)
  --host ADDR               Listen host (default: $HOST)
  --port N                  Listen port (default: $PORT)
  --gpus LIST               Set CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2,3
  --tp N                    Tensor parallel size (default: $TP)
  --dp N                    Data parallel size (default: $DP)
  --nnodes N                Number of nodes for distributed serving (default: $NNODES)
  --node-rank N             Node rank for distributed serving (default: $NODE_RANK)
  --dist-init-addr ADDR     Distributed init address, e.g. 10.0.0.1:20000
  --mem-fraction-static F   Fraction of GPU memory for weights/KV cache (default: $MEM_FRACTION_STATIC)
  --context-length N        Override max context length (default: SGLang auto)
  --allow-longer-context     Set SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  --dtype STR               auto|bfloat16|float16|... (default: $DTYPE)
  --trust-remote-code       Enable trust_remote_code (default: on)
  --no-trust-remote-code    Disable trust_remote_code
  --disable-cuda-graph      Disable SGLang CUDA graph backends
  --background              Run server under nohup and return after health check
  --no-restart              Fail if an SGLang server is already running on the port
  --log FILE                Background log file (default: $LOG_FILE)
  --status                  Check whether the server responds on /v1/models
  --stop                    Kill SGLang server process on this port and exit
  -h, --help                Show this help

Environment defaults:
  SGLANG_MODEL, VLLM_MODEL, SERVED_MODEL_NAME, SGLANG_HOST, SGLANG_PORT, SGLANG_VISIBLE_GPUS,
  CUDA_VISIBLE_DEVICES, SGLANG_TP, SGLANG_DP, SGLANG_NNODES,
  SGLANG_NODE_RANK, SGLANG_DIST_INIT_ADDR, SGLANG_MEM_FRACTION_STATIC,
  SGLANG_CONTEXT_LENGTH, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN,
  SGLANG_DTYPE, SGLANG_TRUST_REMOTE_CODE,
  SGLANG_DISABLE_CUDA_GRAPH, SGLANG_LOG

Compatibility envs still accepted:
  VLLM_HOST, VLLM_PORT, VLLM_VISIBLE_GPUS, VLLM_TP, VLLM_DP,
  VLLM_DATA_PARALLEL_SIZE, VLLM_GPU_MEMORY_UTILIZATION,
  VLLM_DTYPE, VLLM_TRUST_REMOTE_CODE

Examples:
  $0 --gpus 0 --tp 1 --dp 1 --background
  $0 --gpus 0,1 --tp 2 --served-model-name Qwen3-8B --background
  $0 --gpus 0,1 --tp 1 --dp 2 --background
  $0 --gpus 0,1,2,3 --tp 2 --dp 2 -- --max-running-requests 64
EOF
}

die() { echo "[sglang] ERROR: $*" >&2; exit 1; }
require_val() { [[ -n "${2:-}" ]] || die "flag '$1' requires a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-path)     require_val "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
    --served-model-name)      require_val "$1" "${2:-}"; SERVED_MODEL_NAME="$2"; shift 2 ;;
    --host)                   require_val "$1" "${2:-}"; HOST="$2"; shift 2 ;;
    --port)                   require_val "$1" "${2:-}"; PORT="$2"; shift 2 ;;
    --gpus|--cuda-visible-devices)
                              require_val "$1" "${2:-}"; VISIBLE_GPUS="$2"; shift 2 ;;
    --tp|--tp-size|--tensor-parallel-size)
                              require_val "$1" "${2:-}"; TP="$2"; shift 2 ;;
    --dp|--dp-size|--data-parallel-size)
                              require_val "$1" "${2:-}"; DP="$2"; shift 2 ;;
    --nnodes)                 require_val "$1" "${2:-}"; NNODES="$2"; shift 2 ;;
    --node-rank)              require_val "$1" "${2:-}"; NODE_RANK="$2"; shift 2 ;;
    --dist-init-addr|--nccl-init-addr)
                              require_val "$1" "${2:-}"; DIST_INIT_ADDR="$2"; shift 2 ;;
    --mem-fraction-static|--mem-fraction|--gpu-memory-utilization|--gpu-mem-fraction)
                              require_val "$1" "${2:-}"; MEM_FRACTION_STATIC="$2"; shift 2 ;;
    --context-length|--max-model-len)
                              require_val "$1" "${2:-}"; CONTEXT_LENGTH="$2"; shift 2 ;;
    --allow-longer-context)   ALLOW_LONGER_CONTEXT="1"; shift ;;
    --dtype)                  require_val "$1" "${2:-}"; DTYPE="$2"; shift 2 ;;
    --trust-remote-code)      TRUST_REMOTE_CODE="1"; shift ;;
    --no-trust-remote-code)   TRUST_REMOTE_CODE="0"; shift ;;
    --disable-cuda-graph|--enforce-eager)
                              DISABLE_CUDA_GRAPH="1"; shift ;;
    --background)             BACKGROUND="1"; shift ;;
    --no-restart)             RESTART="0"; shift ;;
    --log)                    require_val "$1" "${2:-}"; LOG_FILE="$2"; shift 2 ;;
    --status)                 ACTION="status"; shift ;;
    --stop)                   ACTION="stop"; shift ;;
    -h|--help)                usage; exit 0 ;;
    --)                       shift; EXTRA_ARGS+=("$@"); break ;;
    *)                        die "unknown argument: $1 (run '$0 --help')" ;;
  esac
done

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
URL_HOST="$HOST"
if [[ "$URL_HOST" == "0.0.0.0" || "$URL_HOST" == "::" ]]; then
  URL_HOST="127.0.0.1"
fi
BASE_URL="http://${URL_HOST}:${PORT}/v1"

list_pids() {
  "$PYTHON" - "$PORT" <<'PY_LIST_PIDS'
import re
import subprocess
import sys

port = sys.argv[1]
try:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
except Exception:
    output = ""
for line in output.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_text, _, args = line.partition(" ")
    if not pid_text.isdigit():
        continue
    normalized = " ".join(args.split())
    is_sglang = "sglang.launch_server" in normalized
    if not is_sglang:
        continue
    if re.search(rf"(?:--port(?:=|\s+)){re.escape(port)}(?:\s|$)", normalized):
        print(pid_text)
PY_LIST_PIDS
}

health_check() {
  "$PYTHON" - "$BASE_URL" <<'PY_HEALTH'
import json
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
deadline = time.time() + 120
last_error = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=3) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(body)
                    ids = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
                    if ids:
                        print("[sglang] Models:", ", ".join(ids))
                except Exception:
                    pass
                raise SystemExit(0)
    except Exception as exc:
        last_error = str(exc)
        time.sleep(2)
print(f"[sglang] Health check failed for {base_url}/models: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY_HEALTH
}

stop_server() {
  mapfile -t pids < <(list_pids)
  if ((${#pids[@]} == 0)); then
    echo "[sglang] No SGLang server found on port $PORT."
    return 0
  fi
  echo "[sglang] Stopping SGLang server on port $PORT: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  "$PYTHON" - "${pids[@]}" <<'PY_STOP'
import os
import signal
import sys
import time

pids = [int(x) for x in sys.argv[1:] if x.isdigit()]
deadline = time.time() + 20
while pids and time.time() < deadline:
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        alive.append(pid)
    pids = alive
    if pids:
        time.sleep(0.5)
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY_STOP
}

case "$ACTION" in
  status)
    if health_check; then
      echo "[sglang] Server is ready: $BASE_URL"
      exit 0
    fi
    exit 1
    ;;
  stop)
    stop_server
    exit 0
    ;;
esac

[[ -n "$MODEL" ]] || die "Set SGLANG_MODEL (or VLLM_MODEL) in the root .env, or pass --model PATH."

if ! "$PYTHON" -c 'import sglang' >/dev/null 2>&1; then
  die "SGLang is unavailable at $PYTHON. Run `uv sync --all-groups` then `uv pip install --python .venv/bin/python sglang`."
fi

mapfile -t existing < <(list_pids)
if ((${#existing[@]} > 0)); then
  if [[ "$RESTART" == "1" ]]; then
    stop_server
  else
    die "SGLang server already running on port $PORT: ${existing[*]}"
  fi
fi

if [[ -n "$VISIBLE_GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
fi
if [[ "$ALLOW_LONGER_CONTEXT" == "1" ]]; then
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
fi

CMD=(
  "$PYTHON" -m sglang.launch_server
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --data-parallel-size "$DP"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --dtype "$DTYPE"
)

[[ -n "$CONTEXT_LENGTH" ]] && CMD+=(--context-length "$CONTEXT_LENGTH")
[[ "$TRUST_REMOTE_CODE" == "1" ]] && CMD+=(--trust-remote-code)
if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
  CMD+=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
fi
if [[ "$NNODES" != "1" || "$NODE_RANK" != "0" ]]; then
  CMD+=(--nnodes "$NNODES" --node-rank "$NODE_RANK")
fi
[[ -n "$DIST_INIT_ADDR" ]] && CMD+=(--dist-init-addr "$DIST_INIT_ADDR")
((${#EXTRA_ARGS[@]} > 0)) && CMD+=("${EXTRA_ARGS[@]}")

mkdir -p "$(dirname "$LOG_FILE")"

echo "[sglang] Model: $MODEL"
echo "[sglang] Served model name: $SERVED_MODEL_NAME"
echo "[sglang] Base URL: $BASE_URL"
echo "[sglang] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "[sglang] Parallelism: TP=$TP DP=$DP NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "[sglang] Context length: ${CONTEXT_LENGTH:-<SGLang auto>}"
echo "[sglang] Command: ${CMD[*]}"

if [[ "$BACKGROUND" == "1" ]]; then
  echo "[sglang] Starting in background. Log: $LOG_FILE"
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  pid=$!
  echo "[sglang] PID: $pid"
  if health_check; then
    echo "[sglang] Ready: $BASE_URL"
    exit 0
  fi
  echo "[sglang] Startup did not pass health check. Last log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "[sglang] Starting in foreground. Press Ctrl+C to stop."
exec "${CMD[@]}"
