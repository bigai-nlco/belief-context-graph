#!/usr/bin/env bash
# Shared helpers for BCG service scripts (step 13).
# Source from scripts/start_*.sh:  . "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
# shellcheck disable=SC1091

# Load the repository .env once (idempotent, optional).
bcg_load_root_env() {
  local env_file="${1:-$REPO_ROOT/.env}"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

# Fail fast when a required variable is unset.
bcg_require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing required %s. Add it to .env or export it before running this script.\n' "$name" >&2
    exit 2
  fi
}

# Validate a port value; exits 2 on non-numeric or out-of-range.
bcg_validate_port() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || ((value < 1 || value > 65535)); then
    printf 'Invalid %s: %q (expected an integer port 1-65535).\n' "$name" "$value" >&2
    exit 2
  fi
}

# Fail fast when a port is already in use (listening).
bcg_check_port_free() {
  local name="$1"
  local port="$2"
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
    printf 'Port %s (%s) is already in use. Stop the conflicting process or set %s to another port.\n' "$port" "$name" "$name" >&2
    exit 2
  fi
}

# Render the effective command line for --dry-run mode.
bcg_maybe_dry_run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] Would run: %s\n' "$*" >&2
    exit 0
  fi
}
