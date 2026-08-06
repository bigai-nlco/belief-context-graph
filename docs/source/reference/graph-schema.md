---
title: "Graph Schema"
description: "Typed node, edge, belief, evidence, and relation fields."
icon: "diagram-project"
---

## `BCG`

```json
{
  "nodes": [],
  "edges": [],
  "merges": [],
  "sessions": [],
  "evidence": [],
  "factors": [],
  "metadata": {}
}
```

## `BCGNode`

| Field | Type |
|---|---|
| `uuid` | string UUID |
| `name` | string |
| `kind` | belief, decision, evidence, factor, episode, or entity |
| `probability` | 0–1 |
| `belief` | `BeliefPayload` or null |
| `payload` | object |
| `created_at` / `updated_at` | datetime |
| `metadata` | object |

## `BeliefPayload`

Core fields:

```json
{
  "id": 0,
  "node_type": "belief",
  "belief": "The service is unavailable.",
  "stance": "asserted",
  "layer": "io",
  "source": {},
  "evidence_ids": [],
  "factor_ids": [],
  "entities": [],
  "merged_from": [],
  "confidence": 0.84,
  "initial_confidence": 0.76,
  "evidence_confidence": 0.12,
  "factor_confidence": 0.04,
  "confidence_history": []
}
```

## `BCGEdge`

| Field | Meaning |
|---|---|
| `source` | Source node UUID |
| `target` | Target node UUID |
| `weight` | Optional propagation weight |
| `relation` | Typed `RelationPayload` |
| `payload` | Extension object |
| timestamps | Creation and update time |

## `RelationPayload`

```json
{
  "from_id": 3,
  "to_id": 1,
  "type": "depends_on",
  "note": "Action depends on incident severity.",
  "weight": 0.7,
  "activated_condition": {
    "input_conf_threshold": 0.8
  }
}
```
