---
title: "Manual Beliefs"
description: "Insert a known belief without extraction, embeddings, or a model call."
icon: "brain"
---

Use `BCGMemory.observe()` when the input is already the proposition you want to store.

```python
from bcg import BCG, BCGMemory

memory = BCGMemory(graph=BCG())

observation = memory.observe(
    source_type="manual",
    content="Production deploys require two independent approvals.",
    actor={"id": "policy-bot", "type": "service"},
    observed_at="2026-08-04T19:00:00Z",
    metadata={
        "policy_id": "DEPLOY-02",
        "environment": "production",
    },
)

print(observation.belief.id)
print(observation.belief.confidence)
```

## Behavior

Manual observation:

- creates one asserted belief
- uses the complete `content` string
- records manual evidence
- assigns a source and provenance record
- adds the node to the current in-memory graph
- performs no LLM or embedding call

## Read it back

```python
print(memory.believe("two independent approvals"))

print(memory.context(
    task="Authorize a production deployment",
    focal_entities=["production"],
))
```

## When not to use

Do not use manual observation for raw conversations containing multiple propositions. Use `BCGRunner` so the backend can segment, extract, merge, and link the content.
