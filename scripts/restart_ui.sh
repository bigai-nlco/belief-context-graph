#!/usr/bin/env sh
set -eu

CONTAINER="${BT_CONTAINER:-belief_tracer}"
HOST="${BT_UI_HOST:-172.25.10.2}"
PORT="${BT_UI_PORT:-23456}"
ARTIFACTS_DIR="${BT_ARTIFACTS_DIR:-artifacts/belief_tracer}"
WORKDIR="${BT_WORKDIR:-}"
LOG_FILE="${BT_UI_LOG:-/tmp/belief_tracer-ui-${USER:-unknown}.log}"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Container not found: $CONTAINER" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "Container is not running: $CONTAINER" >&2
  exit 1
fi

if [ -z "$WORKDIR" ]; then
  WORKDIR="$(docker inspect -f '{{.Config.WorkingDir}}' "$CONTAINER" 2>/dev/null || true)"
fi

if [ -z "$WORKDIR" ]; then
  WORKDIR="/workspace/belief_tracer"
fi

docker exec "$CONTAINER" sh -lc '
set -eu
workdir="$1"
host="$2"
port="$3"
artifacts_dir="$4"
log_file="$5"

cd "$workdir"
export PYTHONPATH="$workdir:$workdir/rllm:/workspace/rllm:${PYTHONPATH:-}"

python - "$port" <<'"'"'PY'"'"'
import os
import signal
import subprocess
import sys
import time

port = sys.argv[1]
protected = {os.getpid(), os.getppid()}

try:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
except Exception:
    output = ""

targets = []
for line in output.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_text, _, args = line.partition(" ")
    try:
        pid = int(pid_text)
    except ValueError:
        continue
    if pid in protected:
        continue
    normalized = " ".join(args.split())
    is_ui = (
        "bcg.agent.cli ui" in normalized
        or "bcg agent ui" in normalized
    )
    if is_ui and "--port" in normalized and port in normalized:
        targets.append(pid)

for pid in targets:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

deadline = time.time() + 5
while targets and time.time() < deadline:
    alive = []
    for pid in targets:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        alive.append(pid)
    targets = alive
    if targets:
        time.sleep(0.2)

for pid in targets:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY

nohup python -m bcg.agent ui \
  --host "$host" \
  --port "$port" \
  --artifacts-dir "$artifacts_dir" \
  > "$log_file" 2>&1 &
' sh "$WORKDIR" "$HOST" "$PORT" "$ARTIFACTS_DIR" "$LOG_FILE"

if command -v python3 >/dev/null 2>&1; then
  python3 - "$HOST" "$PORT" <<'PY'
import http.client
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 15

while time.time() < deadline:
    conn = http.client.HTTPConnection(host, port, timeout=1)
    try:
        conn.request("GET", "/api/state")
        if conn.getresponse().status == 200:
            print(f"BeliefTracer UI restarted: http://{host}:{port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(0.5)
    finally:
        conn.close()

print(f"BeliefTracer UI started in background: http://{host}:{port}")
print("Health check did not respond yet; inspect the log if it does not load.")
raise SystemExit(0)
PY
else
  echo "BeliefTracer UI restarted: http://$HOST:$PORT"
fi

echo "Log: $LOG_FILE"
