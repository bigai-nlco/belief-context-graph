---
title: "Installation"
description: "Install the BCG SDK, command line, and optional reference Agent."
icon: "download"
---

BCG requires Python 3.11–3.13. The reference terminal Agent additionally requires Node.js 22.19 or newer.

## Global installation

```bash
curl -LsSf https://raw.githubusercontent.com/bigai-nlco/belief-context-graph/main/install.sh | sh
bcg --version
```

The installer requires `curl`, `tar`, npm, and Node.js. It installs `uv` when needed, installs the Python package and Node runtime, and removes the temporary checkout.

## Source installation

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph

uv sync
npm --prefix agent-cli ci
npm --prefix agent-cli run build

uv run bcg --version
```

Run commands from the checkout with `uv run`:

```bash
uv run bcg
uv run bcg construct --help
```

To expose this checkout globally:

```bash
uv tool install --editable .
npm install -g ./agent-cli
bcg --version
```

## Python SDK only

The Python package contains the SDK and graph construction services. The Node runtime is only needed for the bundled terminal Agent.

```bash
uv sync
uv run python -c "from bcg import BCG, BCGMemory, BCGRunner; print('ready')"
```

## Development dependencies

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

<Info>
Optional benchmark dependencies are declared under the `benchmarks` extra.
</Info>
