---
title: "Merging"
description: "How BCG converts repeated observations into canonical beliefs."
icon: "code-merge"
---

Agents often restate the same proposition across turns. BCG merges duplicates before relation generation so the graph does not fragment into near-identical nodes.

## Merge order

```text
new candidates
    │
    ▼
embedding similarity
    │
    ▼
optional LLM verify + rewrite
    │
    ▼
select canonical node
    │
    ▼
combine evidence and lineage
    │
    ▼
rewire existing relations
    │
    ▼
generate new relations
```

## Incremental merge controls

For `unified`:

```bash
bcg construct run unified \
  --input trajectory.json \
  --incremental-merge \
  --incremental-merge-threshold 0.86 \
  --verify-merge
```

Disable candidate merging:

```bash
bcg construct run unified \
  --input trajectory.json \
  --no-incremental-merge
```

All construction entry points use the shared defaults: threshold `0.86` and `verify_merge=True`.

## Audit trail

Merge artifacts record:

- candidate groups
- similarity values
- optional model verification
- accepted and rejected merges
- canonical text rewrites
- absorbed belief IDs
- edge rewiring

Canonical nodes retain `merged_from` and may preserve `belief_original`.

## Design tradeoff

A lower threshold creates a smaller graph but risks collapsing distinct propositions. A higher threshold preserves nuance but may create redundant context. Evaluate both node quality and downstream relation quality.
