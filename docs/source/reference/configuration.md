---
title: "Configuration"
description: "Model routing, backend settings, confidence policy, and local component configuration."
icon: "sliders-horizontal"
---

Copy the template:

```bash
cp bcg/model_config.example.json bcg/model_config.json
```

Do not store API keys in this JSON. Use `api_key_env`.

## Model entry

```json
{
  "graph-model": {
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1",
    "max_tokens": 100000,
    "temperature": 1,
    "top_p": 0.95,
    "pricing": {
      "input_per_1k": 0.005,
      "output_per_1k": 0.03
    }
  }
}
```

## Embedding entry

Local provider example:

```json
{
  "embedding": {
    "provider": "local",
    "model": "/models/all-MiniLM-L6-v2",
    "device": "cpu",
    "dtype": "auto",
    "batch_size": 8,
    "max_length": 8192,
    "input_prefix": "Document: "
  }
}
```

## `belief_graph` sections

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
The light backend normalizers require complete sections. Copy the template and modify values rather than deleting fields.
</Warning>
