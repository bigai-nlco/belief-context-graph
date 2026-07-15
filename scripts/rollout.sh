#!/usr/bin/env bash
# Thin compatibility entry point for the unified Python agent CLI.
#
# All rollout defaults are owned by AgentRolloutConfig. Use --preset for a
# maintained parameter bundle, or pass any `bcg agent run` option directly.

set -euo pipefail

exec "${BCG_BIN:-bcg}" agent run "$@"
