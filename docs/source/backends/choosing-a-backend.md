---
title: "Choosing a Backend"
description: "Select `api_based` or `light` based on infrastructure and control requirements."
icon: "route"
---

Both backends produce the same conceptual BCG graph and support batch, replay, HTTP, and `BCGRunner` workflows.

| Choose | When |
|---|---|
| `api_based` | You have an OpenAI-compatible model endpoint and want the simplest deployment |
| `light` | You want local embeddings, local stance classification, local NER, and a smaller generative model |

## Comparison

| Capability | api_based | light |
|---|---|---|
| Default backend | Yes | No |
| Generative model | One compatible chat model | Small model via compatible endpoint |
| Embeddings | Configured local or API embedding provider | Local sentence-transformers |
| Stance | API pipeline | Required local classifier |
| NER | Extraction pipeline | Pattern, rules, spaCy ML, or Hugging Face |
| Semantic chunking | Sentence/excerpt modes | Configurable embedding breakpoint chunks |
| Configuration | Stream flags + confidence config | Full `belief_graph` config |
| Operational complexity | Lower | Higher |
| Local control | Moderate | Higher |

## Same public commands

```bash
bcg construct run api_based --input data.json
bcg construct run light --input data.json

bcg construct server api_based --port 8848
bcg construct server light --port 8848
```

If the backend positional argument is omitted and the first token is a flag, BCG selects `api_based` for compatibility.

## Recommendation

Start with `api_based`. Move to `light` when local model control, local classification, or infrastructure separation matters enough to justify the additional configuration.
