---
title: "Streaming HTTP Server"
description: "Maintain one incremental belief graph per problem ID."
icon: "server"
---

Start the server:

```bash
bcg construct server api_based \
  --config bcg/model_config.json \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir outputs_stream
```

## Push one turn

```bash
curl -s -X POST http://127.0.0.1:8848/turn \
  -H 'content-type: application/json' \
  -d '{
    "problem_id": "case-42",
    "role": "user",
    "content": "Which alloy resists seawater corrosion best?"
  }'
```

Finalize on the last turn:

```bash
curl -s -X POST http://127.0.0.1:8848/turn \
  -H 'content-type: application/json' \
  -d '{
    "problem_id": "case-42",
    "role": "assistant",
    "content": "Titanium grade 2 is the standard choice.",
    "is_trajectory_end": true
  }'
```

## Stream message fragments

Set `is_message_end=false` on partial fragments. They are buffered until a final fragment sets `is_message_end=true` or `is_trajectory_end=true`.

```json
{
  "problem_id": "case-42",
  "role": "assistant",
  "content": "Titanium grade ",
  "is_message_end": false
}
```

## Query the current graph

```bash
curl -s \
  'http://127.0.0.1:8848/graph?problem_id=case-42'
```

## Release memory

```bash
curl -s -X POST http://127.0.0.1:8848/release \
  -H 'content-type: application/json' \
  -d '{"problem_id":"case-42"}'
```

## Concurrency

A separate lock protects each `problem_id`. Same-problem turns stay ordered; independent problems can run concurrently.

<Warning>
Do not send the same problem through multiple server processes without sticky routing or external coordination.
</Warning>
