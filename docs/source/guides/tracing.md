---
title: "Tracing"
description: "Enable optional Langfuse spans around BCG execution."
icon: "activity"
---

BCG tracing is optional and designed to be best-effort.

## Enable

Configure Langfuse through its standard environment variables, then enable BCG tracing in the runtime environment expected by `bcg.tracing`.

When tracing is not configured, the decorators and update helpers become no-ops.

## What the module provides

- `trace()` decorator
- `update_current_span()`
- `update_current_generation()`
- `flush_traces()`
- `is_tracing_enabled()`

## Local artifacts remain authoritative

Even with remote tracing enabled, retain BCG's run artifacts:

- prompts
- embedding calls
- merge logs
- events
- token usage
- timing
- final graph

These files provide graph-specific detail that a generic trace backend may not capture.
