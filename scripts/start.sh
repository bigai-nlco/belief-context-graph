#!/usr/bin/env bash
# Run the maintained AVeriTeC + HerO4 preset without duplicating its values.

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

MODEL_ID="${MODEL:-${OPENAI_MODEL:-}}"
if [[ -z "$MODEL_ID" ]]; then
  printf 'Set MODEL or OPENAI_MODEL in %s before starting a rollout.\n' "$ENV_FILE" >&2
  exit 2
fi

if [[ -z "${BCG_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/bcg" ]]; then
    BCG_BIN="${REPO_ROOT}/.venv/bin/bcg"
  elif command -v bcg >/dev/null 2>&1; then
    BCG_BIN="bcg"
  else
    printf 'BCG executable not found. Run `uv sync --all-groups` or `uv tool install .`.\n' >&2
    exit 127
  fi
fi
if [[ "$BCG_BIN" == */* ]]; then
  [[ -x "$BCG_BIN" ]] || {
    printf 'BCG executable not found at %s. Run `uv sync --all-groups` first.\n' "$BCG_BIN" >&2
    exit 127
  }
elif ! command -v "$BCG_BIN" >/dev/null 2>&1; then
  printf 'BCG executable %s not found. Run `uv sync --all-groups` first.\n' "$BCG_BIN" >&2
  exit 127
fi

exec "$BCG_BIN" agent run \
  --preset averitec-hero4 \
  --model "$MODEL_ID" \
  "$@"
