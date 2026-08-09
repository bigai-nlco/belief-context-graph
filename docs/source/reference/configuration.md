---
title: "Configuration"
description: "Model routing, backend settings, confidence policy, and local component configuration."
icon: "sliders-horizontal"
---

Run the setup guide for a global installation:

```bash
bcg setup
```

It writes runtime choices to `~/.bcg/config.json`, model and construction
settings to `~/.bcg/config.yaml`, and secrets to `~/.bcg/.env`. For a
project-specific configuration, copy the YAML template:

```bash
cp bcg/config/config.example.yaml bcg.yaml
```

Do not store API keys in YAML or JSON. Each `api_key_env` names a variable in
`~/.bcg/.env` or the process environment.

## Model entry

```yaml
model_key: graph-model
models:
  graph-model:
    model: gpt-5.6-luna
    api_key_env: BCG_GRAPH_API_KEY
    base_url: https://your-openai-compatible-server/v1
    max_tokens: 100000
    temperature: 1
    top_p: 0.95
```

The matching secret belongs in `~/.bcg/.env`:

```dotenv
BCG_GRAPH_API_KEY=...
```

The reference Agent uses `OPENAI_API_KEY`, while its model and base URL are
selected by `bcg setup` and recorded in `~/.bcg/config.json`. The Agent and
Graph builder may use the same endpoint/key or independent ones.

## Embedding entry

Local provider example:

```yaml
embedding_key: embedding
models:
  embedding:
    provider: local
    model: /models/all-MiniLM-L6-v2
    device: cpu
    dtype: auto
    batch_size: 8
    max_length: 8192
    input_prefix: "Document: "
```

For a remote embedding endpoint, set `base_url` and `api_key_env` in this entry
and put that named key in `~/.bcg/.env`.

## `pipeline` sections

| Section | Used for |
|---|---|
| `extractor` | Light node generation |
| `stance` | Four-class local stance model |
| `edge_generation` | Light relation generation |
| `runtime` | Evidence mode and context budgets |
| `incremental_merge` | Candidate threshold and text policy |
| `entities` | NER method and fallbacks |
| `confidence` | Prior, evidence, and propagation policy |
| `chunking` | Semantic breakpoint behavior |

## Confidence policy

Key fields include:

- `source_weight`
- `stance_weight`
- source-specific reliability
- stance-specific quality
- relation weights
- `input_confidence_threshold`
- minimum propagation delta
- maximum propagation iterations

<Warning>
The light backend normalizers require complete sections. Copy the YAML template and modify values rather than deleting fields. Legacy `model_config.json` files can still be read during the compatibility window, but new setup runs write YAML.
</Warning>
