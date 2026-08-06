---
title: "HTTP API"
description: "Endpoints exposed by the streaming graph construction server."
icon: "server"
---

Base URL examples use `http://127.0.0.1:8848`.

## `GET /health`

```json
{
  "status": "ok",
  "active": ["p1"],
  "all": ["p1", "p2"]
}
```

## `POST /turn`

Body: one turn object.

```json
{
  "problem_id": "p1",
  "role": "user",
  "content": "The service is unavailable.",
  "is_message_end": true,
  "is_trajectory_end": false
}
```

Returns the latest graph snapshot.

## `POST /turns`

Body may be:

- JSON array of turn objects
- NDJSON turn objects

Returns pushed count, finalized IDs, and latest snapshots.

## `POST /input` or `/run`

Accepts any input shape supported by the backend loaders.

Query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `item` | none | Select one item |
| `keep_order` | false | Preserve multi-session input order |
| `finalize` | true | Finalize after ingest |

## `POST /finalize`

```json
{"problem_id":"p1"}
```

## `POST /release`

```json
{"problem_id":"p1"}
```

Returns whether an in-memory session was released.

## `GET /graph?problem_id=p1`

Returns the latest snapshot or `404`.

## Error responses

| Status | Meaning |
|---:|---|
| `400` | Invalid body or missing parameter |
| `404` | Unknown path or graph |
| `409` | Trajectory already closed |
| `500` | Unhandled backend failure |
