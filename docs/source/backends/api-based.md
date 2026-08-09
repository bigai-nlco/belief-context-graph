---
title: "API-Based Backend"
description: "Graph construction with an OpenAI-compatible model and configurable embeddings."
icon: "server"
---

`api_based` is the default construction backend.

## Start a server

```bash
bcg construct server api_based \
  --config ~/.bcg/model_config.json \
  --model-key graph-model \
  --embedding-key embedding \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir ~/.bcg/graphs
```

## Batch construction

```bash
bcg construct run api_based \
  --input trajectory.json \
  --config ~/.bcg/model_config.json \
  --model-key graph-model \
  --embedding-key embedding
```

## Runtime options

| Option | Library default | Typical CLI default |
|---|---:|---:|
| `evidence_mode` | `sentence` | `sentence` |
| `incremental_merge` | `true` | `true` |
| `incremental_merge_threshold` | `0.86` | shared across SDK, batch, and server |
| `verify_merge` | `true` | shared across SDK, batch, and server |
| `context_chars` | `100000` | shared across SDK, batch, and server |
| `min_content_len` | `0` | `0` |

<Info>
Always confirm the effective command defaults with `bcg construct run api_based --help` or `bcg construct server api_based --help`. The public SDK and command layers intentionally have different defaults.
</Info>

## Evidence choices

```bash
# Whole-sentence evidence
bcg construct run api_based \
  --input trajectory.json \
  --evidence-mode sentence

# Free-span quoted evidence
bcg construct run api_based \
  --input trajectory.json \
  --evidence-mode excerpt
```

## Merge verification

Verification adds a model call for embedding-flagged candidate groups. The model can reject the merge or rewrite the canonical node to preserve all accepted meaning.

Use verification when semantic collisions are costly. Disable it when latency and cost matter more than conservative canonicalization.
