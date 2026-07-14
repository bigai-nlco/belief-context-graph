#!/usr/bin/env bash
# Run the Memgraph unit-test suite against the running container.
#
# Usage:
#   scripts/run_memgraph_tests.sh           # uses bolt://localhost:7687
#   MEMGRAPH_URI=bolt://host:7687 scripts/run_memgraph_tests.sh
#
# The script auto-installs the `neo4j` Bolt driver if it isn't already present.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${MEMGRAPH_URI:=bolt://localhost:7687}"
: "${MEMGRAPH_USER:=}"
: "${MEMGRAPH_PASS:=}"
export MEMGRAPH_URI MEMGRAPH_USER MEMGRAPH_PASS

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import neo4j" >/dev/null 2>&1; then
    echo "[setup] installing neo4j Bolt driver..." >&2
    "$PYTHON" -m pip install --quiet neo4j
fi

echo "[run] MEMGRAPH_URI=$MEMGRAPH_URI" >&2
exec "$PYTHON" -m unittest -v scripts.test_memgraph
