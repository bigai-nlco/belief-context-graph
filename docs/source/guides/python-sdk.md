---
title: "Python SDK"
description: "Build, query, and stream BCG graphs inside a Python application."
icon: "python"
---

The public package exports:

```python
from bcg import BCG, BCGMemory, BCGRunner
```

`LLMClient` is available separately:

```python
from bcg.llm import LLMClient, LLMConfig
```

## Construct a complete trajectory

```python
import asyncio

from bcg import BCG, BCGMemory, BCGRunner
from bcg.llm import LLMClient


async def build() -> None:
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=LLMClient(),
        backend="unified",
        output_root=".bcg/runs",
    )

    result = await runner.observe_trajectory(
        [
            {"role": "user", "content": "The payment retry failed twice."},
            {"role": "tool", "content": "retry_count=2; status=declined"},
            {
                "role": "assistant",
                "content": "Do not retry automatically; request updated payment details.",
            },
        ],
        run_id="payment-review",
        scenario="support",
        item_id="ticket-1842",
    )

    print(result.counts)
    print(result.token_usage)


asyncio.run(build())
```

## Stream turns explicitly

```python
runner.begin_belief_run(
    run_id="long-running-investigation",
    backend="unified",
    evidence_mode="sentence",
    incremental_merge=True,
    incremental_merge_threshold=0.84,
)

runner.start_session("day-1", "2026-08-04")

await runner.observe_turn("user", "Supplier A missed the delivery window.")
await runner.observe_turn("tool", "shipment_status=delayed; eta=2026-08-08")
await runner.observe_turn(
    "assistant",
    "The launch plan depends on the revised supplier ETA.",
)

await runner.end_session()
result = await runner.finalize()
```

## Use the resulting memory

```python
matches = memory.believe("supplier")
context = memory.context(
    task="Assess whether launch should be delayed",
    focal_entities=["Supplier A"],
    include_conflicts=True,
    include_missing_evidence=True,
)
```

## Lifecycle rules

- one `BCGRunner` owns one active run at a time
- call `begin_belief_run()` before manual streaming
- `observe_turn()` creates an implicit session when needed
- `finalize()` may only be called once
- begin a new run after finalization
- snapshots keep `memory.graph` synchronized

## Custom model clients

`BCGRunner` accepts a client with either:

- async or sync `generate(messages, ...)`
- async or sync `generate_text(prompt, ...)`

The adapter exposes an OpenAI chat-completions-like shape to construction backends.
