---
title: "BCG"
description: "Reference for the in-memory graph container."
icon: "circle-nodes"
---

```python
from bcg import BCG
```

`BCG` is a Pydantic model containing:

```python
BCG(
    nodes=[],
    edges=[],
    merges=[],
    sessions=[],
    evidence=[],
    factors=[],
    metadata={},
)
```

## Node operations

| Method | Purpose |
|---|---|
| `add_node(node)` | Async generic node insertion |
| `add_belief(belief)` | Create and append a belief node |
| `next_belief_id()` | Read the next monotonic ID |
| `allocate_belief_id()` | Reserve a stable ID |
| `node_for_belief_id(id)` | Find a node |
| `update_belief(payload)` | Replace payload while preserving UUID |
| `remove_belief(id)` | Remove node and optionally incident edges |
| `belief_nodes()` | Ordered belief nodes |
| `beliefs()` | Ordered typed payloads |
| `belief_dicts()` | JSON-compatible active beliefs |

## Relation operations

| Method | Purpose |
|---|---|
| `add_edge(edge)` | Async generic edge insertion |
| `add_relation(payload, source_uuid, target_uuid)` | Create a typed edge |
| `add_relation_by_ids(payload)` | Resolve endpoints by belief ID |
| `add_relations(dicts)` | Validate and add relation dictionaries |
| `relations(type=None)` | Return typed relations |
| `relation_dicts(type=None)` | Return JSON-compatible relations |
| `remap_relations(mapping)` | Rewire endpoints after merges |

## Serialization

```python
json_text = graph.model_dump_json(indent=2)
memory_shape = graph.to_memory_dict()
```

## Mutation note

The public graph is an in-memory container. Persist it explicitly or use `BCGRunner`, which writes run artifacts during finalization.
