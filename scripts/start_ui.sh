#!/usr/bin/env bash
# Start or restart the BeliefTracer web UI in the uv-managed project environment.
#
# Usage:
#   scripts/start_ui.sh                            # defaults: 0.0.0.0:23456
#   scripts/start_ui.sh --host 0.0.0.0 --port 8080
#   scripts/start_ui.sh --no-restart               # fail if already running
#   scripts/start_ui.sh --stop                      # kill existing UI and exit
#   scripts/start_ui.sh --status                    # check whether UI is running

set -euo pipefail

# --- defaults -----------------------------------------------------------
HOST="${BT_UI_HOST:-0.0.0.0}"
PORT="${BT_UI_PORT:-8002}"
ARTIFACTS_DIR="${BT_ARTIFACTS_DIR:-output}"
LOG_FILE="${BT_UI_LOG:-/tmp/belief_tracer-ui-${USER:-unknown}.log}"
RESTART="1"
ACTION="start"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
PYTHON="${BCG_PYTHON:-$REPO_ROOT/.venv/bin/python}"

# --- helpers ------------------------------------------------------------
usage() {
  cat <<EOF
Usage:  $0 [flags]

Flags:
  --host ADDR          Listen address (default: 0.0.0.0)
  --port N             Listen port      (default: 23456)
  --artifacts-dir DIR  Artifact root    (default: artifacts/belief_tracer)
  --log FILE           Log file         (default: /tmp/belief_tracer-ui-$USER.log)
  --no-restart         Fail instead of restarting if a UI is already running
  --stop               Kill any running UI and exit
  --status             Print whether the UI is running, then exit
  -h, --help           Show this help
EOF
  exit 0
}

die() { echo "[start_ui] ERROR: $*" >&2; exit 1; }

# Find PIDs of running belief_tracer UI processes on the same port.
list_uids() {
  local pids=()
  while IFS= read -r line; do
    line="${line# }"
    [[ -z "$line" ]] && continue
    local pid="${line%% *}"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
      pids+=("$pid")
    fi
  done < <(ps -eo pid=,args= 2>/dev/null | grep -E "bcg.agent.*ui" | grep -v grep || true)
  local result=()
  for pid in "${pids[@]}"; do
    local args
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$args" == *"--port"*"$PORT"* ]] || [[ "$args" == *"--port="$PORT* ]]; then
      result+=("$pid")
    fi
  done
  ((${#result[@]})) && printf '%s\n' "${result[@]}"
}

kill_uids() {
  local pids
  mapfile -t pids < <(list_uids)
  ((${#pids[@]})) || { echo "[start_ui] No running BeliefTracer UI found on port $PORT."; return 0; }
  echo "[start_ui] Killing ${#pids[@]} existing UI process(es) on port $PORT..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Wait up to 5 s for graceful shutdown.
  local deadline=$(bc -l <<< "$(date +%s.%N) + 5")
  while (($(date +%s.%N | awk '{print int($1)}') < ${deadline%.*})) && ((${#pids[@]})); do
    local alive=()
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid") || true
    done
    pids=("${alive[@]}")
    ((${#pids[@]})) && sleep 0.3 || break
  done
  for pid in "${pids[@]}"; do
    echo "[start_ui] Force-killing PID $pid"
    kill -9 "$pid" 2>/dev/null || true
  done
}

health_check() {
  local proto host port
  proto="http"
  host="$1"
  port="$2"
  # Prefer python3; fall back to curl.
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import urllib.request, sys, time
deadline = time.time() + 10
while time.time() < deadline:
    try:
        r = urllib.request.urlopen('$proto://$host:$port/api/state', timeout=2)
        if r.status == 200:
            sys.exit(0)
    except Exception:
        time.sleep(0.5)
sys.exit(1)
" 2>/dev/null && return 0 || return 1
  elif command -v curl >/dev/null 2>&1; then
    local dl
    dl=$(bc -l <<< "$(date +%s.%N) + 10")
    while (($(date +%s.%N | awk '{print int($1)}') < ${dl%.*})); do
      if curl -sf --max-time 2 "$proto://$host:$port/api/state" >/dev/null 2>&1; then
        return 0
      fi
      sleep 0.5
    done
    return 1
  else
    sleep 2
    return 0  # best effort
  fi
}

# --- parse flags --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)           HOST="${2:?}"; shift 2 ;;
    --port)           PORT="${2:?}"; shift 2 ;;
    --artifacts-dir)  ARTIFACTS_DIR="${2:?}"; shift 2 ;;
    --log)            LOG_FILE="${2:?}"; shift 2 ;;
    --no-restart)     RESTART="0"; shift ;;
    --stop)           ACTION="stop"; shift ;;
    --status)         ACTION="status"; shift ;;
    -h|--help)        usage ;;
    --) shift; break ;;
    *) die "Unknown argument: $1 (run '$0 --help')" ;;
  esac
done

# --- actions -------------------------------------------------------------

# --status
if [[ "$ACTION" == "status" ]]; then
  PIDS=()
  mapfile -t PIDS < <(list_uids)
  if ((${#PIDS[@]})); then
    echo "BeliefTracer UI is running on http://$HOST:$PORT (PID: ${PIDS[*]})"
    exit 0
  else
    echo "BeliefTracer UI is NOT running on http://$HOST:$PORT"
    exit 1
  fi
fi

# --stop
if [[ "$ACTION" == "stop" ]]; then
  kill_uids
  exit 0
fi

# --start (default)
PIDS=()
mapfile -t PIDS < <(list_uids)
if ((${#PIDS[@]})); then
  if [[ "$RESTART" == "1" ]]; then
    kill_uids
  else
    echo "[start_ui] UI already running on http://$HOST:$PORT (PID: ${PIDS[*]})"
    echo "[start_ui] Use '$0 --stop' to kill it, or omit --no-restart to restart."
    exit 1
  fi
fi

mkdir -p "$(dirname "$LOG_FILE")"

[[ -x "$PYTHON" ]] || die "BCG Python not found at $PYTHON. Run `uv sync --all-groups` first."

echo "[start_ui] Starting BeliefTracer UI on http://$HOST:$PORT"
echo "[start_ui] Artifacts: $ARTIFACTS_DIR"
echo "[start_ui] Log: $LOG_FILE"

trap 'echo "[start_ui] Shutting down..."; exit 0' INT TERM

echo "[start_ui] UI is ready: http://$HOST:$PORT"
echo "[start_ui] Press Ctrl+C to stop."

"$PYTHON" -m bcg.agent ui \
  --host "$HOST" \
  --port "$PORT" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  2>&1 | tee "$LOG_FILE"
