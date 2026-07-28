#!/usr/bin/env bash
# Thin compatibility entry point for the unified Python agent CLI.
#
# All rollout defaults are owned by AgentRolloutConfig. Use --preset for a
# maintained parameter bundle, or pass any `bcg agent run` option directly.
# The default executable is the project environment created by `uv sync`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BCG_BIN="${BCG_BIN:-${REPO_ROOT}/.venv/bin/bcg}"

if [[ "$BCG_BIN" == */* ]]; then
  [[ -x "$BCG_BIN" ]] || {
    printf 'BCG executable not found at %s. Run `uv sync --all-groups` first.\n' "$BCG_BIN" >&2
    exit 127
  }
elif ! command -v "$BCG_BIN" >/dev/null 2>&1; then
  printf 'BCG executable %s not found. Run `uv sync --all-groups` first.\n' "$BCG_BIN" >&2
  exit 127
fi

cd "$REPO_ROOT"
exec "$BCG_BIN" agent run "$@"
