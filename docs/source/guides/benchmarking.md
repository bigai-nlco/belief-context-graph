---
title: "Benchmarking"
description: "Compare the reference Agent in Default and BCG context modes."
icon: "gauge"
---

The benchmark adapter runs supported datasets through the bundled Agent and writes trajectories plus summary metrics.

## Basic command

```bash
bcg benchmark run hotpotqa \
  --data-root datasets \
  --modes default,bcg \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL"
```

## Supported benchmark names

The exact list is exposed by the CLI:

```bash
bcg benchmark run --help
```

The repository includes adapters for the benchmark set declared in `bcg.benchmark.loaders.BENCHMARKS`.

## Important controls

| Option | Purpose |
|---|---|
| `--modes` | Comma-separated `default,bcg` |
| `--max-problems` | Limit examples; `0` means all |
| `--task-ids` | Select exact IDs |
| `--shuffle/--no-shuffle` | Shuffle before truncation |
| `--workers` | Concurrent Agent processes |
| `--timeout` | Per-task timeout |
| `--graph-url` | Graph server for BCG mode |
| `--graph-max-turns` | Maximum graph messages |
| `--recent-turns` | Raw completed turns retained |
| `--graph-view` | `full` graph dialogue or low-token `compact` belief projection |
| `--allow-graph-fallback` | Score tasks that fell back to raw context |
| `--overwrite` | Rerun existing task artifacts |

## Fair comparison

Keep constant:

- agent model and endpoint
- tool availability
- benchmark sample selection
- timeout
- judge model where applicable

Change only context mode and the BCG policy parameters being evaluated.

## Interpret results carefully

Measure more than final accuracy:

- cumulative token cost over task horizon
- latency by turn
- graph construction cost
- graph fallback rate
- contradiction and merge behavior
- decision trace quality
- evidence grounding

BCG runs also write `graph-contexts/<task-id>.jsonl`. These traces contain the exact role-marked graph text supplied to each model request and can be aligned with the trajectory to verify whether later Agent actions use evicted information retained by the graph.
