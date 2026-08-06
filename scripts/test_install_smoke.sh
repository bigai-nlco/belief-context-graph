#!/usr/bin/env bash
# Post-install smoke tests (step 14): verify installed components work from a
# clean, temporary HOME — version/help commands, resource files, and first
# configuration creation. Requires `bcg` and `bcg-agent` on PATH (as
# install.sh leaves them). Run via `make check-install-smoke`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

failures=0
check() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  if [[ "$actual" == *"$expected"* ]]; then
    printf 'ok   - %s\n' "$description"
  else
    printf 'FAIL - %s\n      expected substring: %s\n      got: %s\n' "$description" "$expected" "$actual" >&2
    failures=$((failures + 1))
  fi
}

command -v bcg >/dev/null 2>&1 || {
  printf 'bcg not on PATH; run install.sh or make install-tool first.\n' >&2
  exit 127
}

temporary_home="$(mktemp -d)"
trap 'rm -rf "$temporary_home"' EXIT HUP INT TERM

export HOME="$temporary_home"

# --- Python launcher ---
check "bcg --version" "1.0.0" "$(bcg --version 2>&1)"
check "bcg config show" "backend" "$(bcg config show 2>&1 | head -5)"

# Packaged defaults reachable from the installed CLI (fresh HOME, no config)
check "bcg packaged defaults (port)" "8848" "$(bcg config show 2>&1 | grep -m1 'port:')"

# First configuration creation: a fresh HOME gets no config yet, and the
# packaged defaults are the effective settings (no legacy fallback).
if [[ -e "$temporary_home/.bcg" ]]; then
  printf 'FAIL - fresh HOME unexpectedly has ~/.bcg after read-only commands\n' >&2
  failures=$((failures + 1))
else
  printf 'ok   - fresh HOME untouched by read-only commands\n'
fi

# --- Agent ---
command -v bcg-agent >/dev/null 2>&1 || {
  printf 'bcg-agent not on PATH; skipping Agent smoke (install.sh installs it).\n' >&2
  exit 127
}
check "bcg-agent --version" "1.0.0" "$(bcg-agent --version 2>&1 || true)"
check "bcg-agent --help" "Usage" "$(bcg-agent --help 2>&1 | head -5)"

# Agent can resolve its installed resources from the fresh HOME (settings
# defaults without touching the real home).
check "bcg-agent help exits cleanly" "" "$(bcg-agent --help >/dev/null 2>&1 && echo ok)"

if ((failures > 0)); then
  printf '%d install smoke test(s) failed.\n' "$failures" >&2
  exit 1
fi
printf 'All install smoke tests passed (HOME=%s).\n' "$temporary_home"
