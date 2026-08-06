---
title: "BCGMemory"
description: "Reference for the application-facing memory facade."
icon: "brain"
---

```python
from bcg import BCGMemory
```

## Constructor

```python
BCGMemory(
    namespace="default",
    options={},
    graph=None,
    confidence_config=None,
)
```

A graph is created lazily when needed.

## `observe()`

```python
observation = memory.observe(
    source_type="manual",
    content="A known proposition.",
    actor=None,
    observed_at=None,
    metadata=None,
)
```

Returns `MemoryObservation` with:

- `belief`
- `graph`

The method performs no LLM extraction.

## `believe()`

```python
matches = memory.believe("substring")
```

Returns belief dictionaries whose text matches the target substring.

## `context()`

```python
text = memory.context(
    task="Plan the next action",
    focal_entities=["Acme"],
    max_variables=100,
    include_conflicts=True,
    include_missing_evidence=True,
)
```

Assembles a Markdown-like task context from the current graph.

## `search()`

```python
results = await memory.search(
    query="outage risk",
    max_results=10,
    include_conflicts=True,
    include_missing_evidence=True,
)
```

The current implementation searches belief text in the in-memory graph. It is not a vector database API.
