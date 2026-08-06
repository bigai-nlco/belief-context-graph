---
title: "Output Artifacts"
description: "Files written by SDK, batch, and streaming construction."
icon: "file-output"
---

BCG writes both public compatibility artifacts and backend-native audit files.

## SDK run directory

```text
.bcg/runs/<run_id>/
  graph.json
  memory.json
  token_usage.json
  events.jsonl
  result.json
  final_graph.json
  trajectory.json
  belief_graph.jsonl
  artifacts/
    segments.json
    io_beliefs.json
    reasoning_beliefs.json
    forward_relations.json
    backward_relations.json
    merges.json
```

## Batch or streaming item directory

```text
<output-root>/<item_or_problem_id>/
  result.json
  final_graph.json
  belief_graph_latest.json
  belief_graph.jsonl
  trajectory.json
  trajectory_stream.jsonl
  events.jsonl
  token_usage.json
  token_usage.txt
  logs/
```

Not every file is produced by every execution path.

## Key files

| File | Use |
|---|---|
| `result.json` | Complete result and audit metadata |
| `final_graph.json` | Final graph snapshot |
| `belief_graph.jsonl` | Per-turn snapshots |
| `trajectory_stream.jsonl` | Raw received stream |
| `events.jsonl` | New nodes, relations, merges, and timing |
| `token_usage.*` | Usage and estimated cost |
| `logs/prompts.jsonl` | Model prompts |
| `logs/embedding_calls.jsonl` | Embedding inputs and cache activity |
| `logs/merge_*` | Candidate and applied merge audit |
| `logs/timing.csv` | Per-turn and summary timing |

## Retention

Treat output directories as potentially sensitive. They may contain original user text, tool output, prompts, exact evidence excerpts, and model-generated reasoning summaries.
