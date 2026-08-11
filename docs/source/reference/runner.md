---
title: "BCGRunner"
description: "Reference for graph-construction lifecycle orchestration."
icon: "play"
---

```python
from bcg import BCGRunner
```

## Constructor

```python
BCGRunner(
    memory=memory,
    llm=llm,
    output_root=".bcg/runs",
    backend="unified",
)
```

## Complete trajectory

```python
result = await runner.observe_trajectory(
    trajectory,
    run_id=None,
    model=None,
    max_tokens=None,
    scenario="research",
    item_id="trajectory",
    backend=None,
    evidence_mode="sentence",
    incremental_merge=True,
    incremental_merge_threshold=0.86,
    verify_merge=True,
    context_chars=100000,
    io_context_chars=6000,
    min_content_len=0,
    belief_graph_config=None,
    embedder=None,
    metadata=None,
    options=None,
)
```

## Explicit lifecycle

```python
runner.begin_belief_run(...)
runner.start_session(session_id, date=None)
snapshot = await runner.observe_turn(role, content, date=None, has_answer=None)
summary = await runner.end_session()
result = await runner.finalize()
```

## Result

`BeliefGraphRunResult` contains:

| Field | Meaning |
|---|---|
| `run_id` | Stable run identifier |
| `graph` | Final public `BCG` |
| `memory` | Serialized memory document |
| `output_paths` | Typed artifact paths |
| `token_usage` | Construction usage totals |
| `counts` | Node, relation, and related counts |
| `construct_result` | Native backend result |

## Errors

- calling turn operations without an active run raises `RuntimeError`
- beginning a second active run is invalid
- starting a session while one is active is invalid
- calling `finalize()` twice is invalid
