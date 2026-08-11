---
title: "Hybrid Backend"
description: "A locally controlled pipeline with small-model extraction, embeddings, stance, and NER."
icon: "feather"
---

The `hybrid` backend decomposes construction into local or separately served components.

## Components

- small generative extractor through an OpenAI-compatible endpoint
- small generative edge generator
- local sentence-transformers embeddings
- required four-class stance classifier
- local named-entity recognition
- semantic breakpoint chunking
- deterministic confidence and merge pipeline

## Start supporting services

The repository includes helper scripts:

```bash
./scripts/start_vllm.sh
./scripts/start_sglang_server.sh
```

For a normal installation, run `bcg setup`; BCG stores graph model routing in `~/.bcg/model_config.json`. The repository helper scripts remain available for source development.

## Run

```bash
bcg construct run hybrid \
  --input trajectory.json \
  --config ~/.bcg/model_config.json \
  --model-key graph-model \
  --embedding-key embedding
```

## Required configuration sections

Every key required by the normalizers must be present. Copy the complete template rather than authoring sections from scratch:

```bash
cp bcg/model_config.example.json bcg/model_config.json
```

The `belief_graph` block contains:

- `extractor`
- `stance`
- `edge_generation`
- `runtime`
- `incremental_merge`
- `entities`
- `confidence`
- `chunking`

## When to use

Use `hybrid` when:

- embeddings must remain local
- stance classification needs a dedicated model
- local NER is preferred
- you want independent control over extraction and edge generation
- you can operate the extra model and asset dependencies

<Warning>
The template contains placeholder local model paths. Replace them before starting the backend.
</Warning>
