#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPO_ROOT="$ROOT"
# shellcheck source=lib/common.sh
. "$ROOT/scripts/lib/common.sh"

bcg_load_root_env

CONFIG="${TONGGRAPH_CONFIG:-$ROOT/deploy/tonggraph-server.yml}"
DATA_DIR="${TONGGRAPH_DATA_DIR:-$ROOT/var/tonggraph}"
BIN="${TONGGRAPH_SERVER_BIN:-tonggraph-server}"
RUNTIME_CONFIG="${TONGGRAPH_RUNTIME_CONFIG:-$ROOT/var/tonggraph-config.generated.yml}"
PORT="${TONGGRAPH_PORT:-8719}"
bcg_validate_port TONGGRAPH_PORT "$PORT"

bcg_require_env TONGGRAPH_ADMIN_TOKEN
bcg_require_env TONGGRAPH_AGENT_WRITER_TOKEN
bcg_require_env TONGGRAPH_AGENT_READER_TOKEN

if ! command -v "$BIN" >/dev/null 2>&1; then
  printf 'tonggraph-server not found. Install tonggraph[server] or set TONGGRAPH_SERVER_BIN to an explicit path.\n' >&2
  exit 127
fi

# --dry-run / -n: render the command without starting anything
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "Environment: TONGGRAPH_SERVER_BIN, TONGGRAPH_CONFIG, TONGGRAPH_DATA_DIR, TONGGRAPH_PORT, TONGGRAPH_*_TOKEN"
      exit 0 ;;
  esac
done

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$RUNTIME_CONFIG")"

# Inject the effective data dir into a runtime copy of the config so
# TONGGRAPH_DATA_DIR is actually consumed by the server (the checked-in YAML
# stays a portable template with a relative default).
sed "s|^data_dir:.*|data_dir: $DATA_DIR|" "$CONFIG" > "$RUNTIME_CONFIG"

bcg_maybe_dry_run "$BIN" --config "$RUNTIME_CONFIG"

exec "$BIN" --config "$RUNTIME_CONFIG"
