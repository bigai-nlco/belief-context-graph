---
title: "Batch Construction"
description: "Build graphs from trajectory files and benchmark-style datasets."
icon: "boxes"
---

Use `bcg construct run` for saved inputs.

## Basic command

```bash
bcg construct run api_based \
  --input data.json \
  --config ~/.bcg/model_config.json \
  --output-dir outputs \
  --model-key graph-model \
  --embedding-key embedding
```

## Common options

| Flag | Default | Purpose |
|---|---|---|
| `--input`, `-i` | required | JSON or TXT input |
| `--config`, `-c` | `bcg/model_config.json` | Model configuration |
| `--output-dir`, `-o` | `outputs` | Root directory |
| `--model-key` | `gpt-5.5` | Chat model config entry |
| `--embedding-key` | `embedding` | Embedding config entry |
| `--item` | all | Process one item by ID or index |
| `--keep-order` | off | Preserve multi-session input order |

## Select one item

```bash
bcg construct run light \
  --input dataset.json \
  --item 3
```

## API-based controls

```bash
bcg construct run api_based \
  --input data.json \
  --evidence-mode excerpt \
  --incremental-merge-threshold 0.86 \
  --context-chars 100000
```

## Output layout

Each item receives a subdirectory:

```text
outputs/<item_id>/
  result.json
  final_graph.json
  token_usage.json
  events.jsonl
  logs/
  ...
```

See [Output artifacts](/reference/output-artifacts).
