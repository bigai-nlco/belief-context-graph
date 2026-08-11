---
title: "Replay Streams"
description: "Feed an NDJSON turn log through a construction backend."
icon: "clock-rotate-left"
---

Replay accepts one turn object per line.

```jsonl
{"problem_id":"p1","role":"user","content":"The supplier is late."}
{"problem_id":"p1","role":"assistant","content":"The release date may need to move.","is_trajectory_end":true}
```

## From a file

```bash
bcg construct replay unified \
  --input turns.jsonl \
  --config bcg/model_config.json \
  --output-dir outputs_stream
```

## From standard input

```bash
cat turns.jsonl | bcg construct replay hybrid
```

## End-of-file finalization

If a trajectory never sends `is_trajectory_end`, replay finalizes all remaining active problem IDs at end of file.

## Malformed lines

Blank lines are ignored. Malformed JSON and non-object lines produce warnings and are skipped rather than terminating the full replay.
