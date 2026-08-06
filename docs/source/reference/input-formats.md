---
title: "Input Formats"
description: "Trajectory, message-list, multi-session, and streamed turn shapes."
icon: "file-input"
---

BCG loaders accept several practical forms.

## Trajectory object

```json
{
  "id": "case-1",
  "trajectory": [
    {"role": "user", "content": "The supplier is late."},
    {"role": "assistant", "content": "The launch date may need to move."}
  ]
}
```

## Messages object

```json
{
  "messages": [
    {"role": "user", "content": "The supplier is late."}
  ]
}
```

## Bare message list

```json
[
  {"role": "user", "content": "The supplier is late."},
  {"role": "assistant", "content": "We should revise the plan."}
]
```

## Streamed turn

```json
{
  "problem_id": "case-1",
  "role": "tool",
  "content": "eta=2026-08-08",
  "date": "2026-08-04",
  "has_answer": false,
  "is_message_end": true,
  "is_trajectory_end": false
}
```

## Fragmented message

Multiple objects with the same problem and role may be concatenated while `is_message_end=false`.

## Multi-session data

Dataset loaders can normalize multi-session QA-style items. Use `--keep-order` when source order must not be date-sorted and `--item` to select one item.

<Info>
Input adapters are intentionally permissive. Output graph schemas are more structured and should be treated as the stable audit format.
</Info>
