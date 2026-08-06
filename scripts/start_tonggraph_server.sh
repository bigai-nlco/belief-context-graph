#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

CONFIG="${TONGGRAPH_CONFIG:-$ROOT/deploy/tonggraph-server.yml}"
DATA_DIR="${TONGGRAPH_DATA_DIR:-$ROOT/var/tonggraph}"
BIN="${TONGGRAPH_SERVER_BIN:-tonggraph-server}"
RUNTIME_CONFIG="${TONGGRAPH_RUNTIME_CONFIG:-$ROOT/var/tonggraph-config.generated.yml}"

if ! command -v "$BIN" >/dev/null 2>&1; then
  printf 'tonggraph-server not found. Install tonggraph[server] or set TONGGRAPH_SERVER_BIN to an explicit path.\n' >&2
  exit 127
fi

for name in TONGGRAPH_ADMIN_TOKEN TONGGRAPH_AGENT_WRITER_TOKEN TONGGRAPH_AGENT_READER_TOKEN; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing %s. Add it to .env or export it before starting TongGraph.\n' "$name" >&2
    exit 2
  fi
done

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$RUNTIME_CONFIG")"

# Inject the effective data dir into a runtime copy of the config so
# TONGGRAPH_DATA_DIR is actually consumed by the server (the checked-in YAML
# stays a portable template with a relative default).
sed "s|^data_dir:.*|data_dir: $DATA_DIR|" "$CONFIG" > "$RUNTIME_CONFIG"

exec "$BIN" --config "$RUNTIME_CONFIG"
