---
title: "Relations"
description: "Directional edges for dependency, supplementation, and contradiction."
icon: "share-nodes"
---

A relation connects belief IDs and may carry confidence propagation policy.

```python
from bcg.graph import RelationPayload

relation = RelationPayload(
    from_id=3,
    to_id=1,
    type="depends_on",
    note="The recommended action depends on the recurring outage claim.",
    weight=0.7,
)
```

## Active relation types

### `depends_on`

The source belief relies on the target belief.

```text
decision: contact Acme urgently
              │ depends_on
              ▼
belief: outages are recurring
```

Support propagates from `to_id` to `from_id`.

### `supplements`

The target adds useful context but is not a confidence factor. The current construction policy keeps `weight` and `activated_condition` as `null`.

### `contradicts`

The source conflicts with the target. When active, confidence can be reduced in the configured direction.

## Compatibility names

The public schema can still read:

| Legacy | Current semantic equivalent |
|---|---|
| `informs` | `depends_on` |
| `extends` | `supplements` |
| `confirms` | supportive legacy relation |

The live viewer may display legacy supportive edges as `supports`.

## Graph integrity

`BCG` validates relation endpoints against active belief IDs. During a merge, `remap_relations()` rewrites endpoints and drops invalid or duplicate edges.
