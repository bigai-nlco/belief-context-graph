---
title: "Custom Agent Integration"
description: "Add BCG to an existing agent without adopting the reference Agent."
icon: "plug"
---

BCG is agent-independent. Your integration needs three decisions:

1. how turns reach graph construction
2. how graph state is rendered for the model
3. when confidence or conflicts block action

## Integration patterns

<Tabs>

<Tab title="Python in-process">

Use `BCGRunner` and keep `BCGMemory` beside your agent state.

```python
snapshot = await runner.observe_turn(role, content)
context = memory.context(task=current_task)
```

</Tab>

<Tab title="HTTP service">

Push turns from any language:

```http
POST /turn
Content-Type: application/json

{
  "problem_id": "agent-session-id",
  "role": "tool",
  "content": "inventory=0"
}
```

</Tab>

<Tab title="Offline reconstruction">

Save the trajectory, then run `bcg construct run` after completion for audit and evaluation.

</Tab>

</Tabs>

## Recommended message mapping

| Agent event | BCG role |
|---|---|
| User input | `user` |
| Assistant reasoning or output | `assistant` |
| Tool result | `tool` |
| Function result | `function` |
| System policy | `system` when intentionally included |

## Context injection

A minimal policy:

```python
graph_context = memory.context(
    task=current_task,
    focal_entities=current_entities,
    include_conflicts=True,
    include_missing_evidence=True,
)

system_prompt = base_system_prompt + "\n\n" + graph_context
```

Production policies should filter by authorization, source role, confidence, recency, and task relevance.

## Action gating

BCG does not execute actions. Your agent can use graph state to require:

- minimum confidence
- no unresolved contradiction
- supporting evidence present
- specific source classes
- human confirmation for decision nodes

## Session cleanup

Finalize the graph when a task ends. When using the HTTP server, call `/release` after artifacts have been consumed and the in-memory session is no longer needed.
