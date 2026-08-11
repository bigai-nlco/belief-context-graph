---
title: "CLI Reference"
description: "Top-level BCG commands and construction workflows."
icon: "terminal"
---

## Top-level

```bash
bcg
bcg agent
bcg setup
bcg construct
bcg benchmark
bcg --version
```

Running `bcg` without a subcommand opens the reference Agent.

## Construction

```bash
bcg construct run <backend> [options]
bcg construct server <backend> [options]
bcg construct replay <backend> [options]
bcg construct visualize <input> [--output path]
```

Backends:

- `unified`
- `hybrid`

When omitted from batch, server, or replay and the first remaining token is an option, the backend defaults to `unified`.

## Discover exact flags

```bash
bcg --help
bcg construct --help
bcg construct run unified --help
bcg construct run hybrid --help
bcg construct server unified --help
bcg construct replay unified --help
bcg benchmark run --help
```

## Visualize

```bash
bcg construct visualize result.json
bcg construct visualize final_graph.json --output graph.html
```
