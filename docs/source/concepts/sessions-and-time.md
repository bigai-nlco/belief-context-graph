---
title: "Sessions & Time"
description: "Temporal boundaries and source metadata inside one belief run."
icon: "clock-rotate-left"
---

BCG provides run-based lifecycle and explicit session metadata.

## Run

A run has:

- stable `run_id`
- scenario and item ID
- one selected construction backend
- trajectory and session list
- graph metadata
- output paths and artifacts

## Session

Use explicit sessions when a trajectory spans meetings, days, users, or phases.

```python
runner.begin_belief_run(run_id="customer-history")
runner.start_session("support-call-1", "2026-06-12")

await runner.observe_turn("user", "The outage happened again.")
await runner.observe_turn("assistant", "I will escalate the reliability review.")

await runner.end_session()
result = await runner.finalize()
```

Each observed turn records:

- `session_id`
- `session_index`
- session date
- turn index
- optional `has_answer`

If you call `observe_turn()` without an active session, `BCGRunner` creates one automatically.

## Event time vs observation time

A belief can carry:

- source session and turn time
- `event_time` for when the described event occurred
- `time_text` preserving the original temporal expression
- node `created_at` and `updated_at`

These are different concepts. “The outage happened last Tuesday” is observed now but describes an earlier event.

## Current persistence model

Sessions are serialized in graph artifacts. Active HTTP sessions remain in process memory until finalization and optional release.
