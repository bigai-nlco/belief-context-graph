---
title: "Unified Backend"
description: "Graph construction with an OpenAI-compatible model and configurable embeddings."
icon: "server"
---

`unified` is the default construction backend.

## Start a server

```bash
bcg construct server unified \
  --config ~/.bcg/model_config.json \
  --model-key graph-model \
  --embedding-key embedding \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir ~/.bcg/graphs
```

## Batch construction

```bash
bcg construct run unified \
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
| `max_previous_windows` | `4` | bounded relation-history search |
| `min_content_len` | `0` | `0` |

The chat-model entry also accepts `reasoning_effort`. When omitted,
`gpt-5.6-luna` uses `none`; other models retain the historical `medium`
default. For example:

```yaml
models:
  graph-model:
    model: gpt-5.6-luna
    reasoning_effort: none
```

<Info>
Always confirm the effective command defaults with `bcg construct run unified --help` or `bcg construct server unified --help`. The public SDK and command layers intentionally have different defaults.
</Info>

## Evidence choices

```bash
# Whole-sentence evidence
bcg construct run unified \
  --input trajectory.json \
  --evidence-mode sentence

# Free-span quoted evidence
bcg construct run unified \
  --input trajectory.json \
  --evidence-mode excerpt
```

## Merge verification

Verification adds a model call for embedding-flagged candidate groups. The model can reject the merge or rewrite the canonical node to preserve all accepted meaning.

Use verification when semantic collisions are costly. Disable it when latency and cost matter more than conservative canonicalization.
