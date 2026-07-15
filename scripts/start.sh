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

exec "${BCG_BIN:-bcg}" agent run \
  --preset averitec-hero4 \
  --model "$MODEL_ID" \
  "$@"
