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

if ! command -v "$BIN" >/dev/null 2>&1; then
  FALLBACK="/data/user/baijun/projects/TongGraph/.venv/bin/tonggraph-server"
  if [[ -x "$FALLBACK" ]]; then
    BIN="$FALLBACK"
  else
    printf 'tonggraph-server not found. Install tonggraph[server] or set TONGGRAPH_SERVER_BIN.\n' >&2
    exit 127
  fi
fi

for name in TONGGRAPH_ADMIN_TOKEN TONGGRAPH_AGENT_WRITER_TOKEN TONGGRAPH_AGENT_READER_TOKEN; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing %s. Add it to .env or export it before starting TongGraph.\n' "$name" >&2
    exit 2
  fi
done

mkdir -p "$DATA_DIR"
exec "$BIN" --config "$CONFIG"
