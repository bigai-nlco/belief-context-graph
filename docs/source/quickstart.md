---
title: "Quick Start"
description: "Build and inspect a belief graph in a few minutes."
icon: "rocket"
---

This quick start uses the `unified` backend through the Python SDK.

## 1. Configure the model endpoint

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"
```

## 2. Build a graph

```python
import asyncio

from bcg import BCG, BCGMemory, BCGRunner
from bcg.llm import LLMClient


async def main():
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=LLMClient(),
        backend="unified",
    )

    result = await runner.observe_trajectory(
        [
            {
                "role": "user",
                "content": "Acme is threatening to churn after repeated outages.",
            },
            {
                "role": "assistant",
                "content": (
                    "We should prioritize a reliability review and contact Acme. "
                    "This is urgent because the outages are recurring."
                ),
            },
        ],
        run_id="acme-risk-review",
    )

    print("beliefs:", len(result.graph.beliefs()))
    print("relations:", len(result.graph.relations()))
    print("graph:", result.output_paths.graph)
    print("memory:", result.output_paths.memory)
    return memory, result


memory, result = asyncio.run(main())
```

## 3. Inspect beliefs and context

```python
for belief in memory.graph.beliefs():
    print(
        belief.id,
        belief.node_type,
        belief.confidence,
        belief.belief,
    )

print(memory.context(task="Decide the next customer-success action"))
```

## 4. Render an HTML graph

```bash
bcg construct visualize \
  .bcg/runs/acme-risk-review/result.json \
  --output acme-risk-review.html
```

<Note>
`observe_trajectory()` automatically begins the run, pushes every turn, closes the active session, finalizes the backend, synchronizes the public graph, and writes artifacts.
</Note>

## Next steps

<CardGroup cols={2}>
<Card title="Understand confidence" icon="gauge-high" href="/concepts/confidence" />
<Card title="Stream turns over HTTP" icon="server" href="/guides/streaming-server" />
<Card title="Use explicit sessions" icon="clock" href="/guides/python-sdk#stream-turns-explicitly" />
<Card title="Choose the hybrid backend" icon="feather" href="/backends/hybrid" />
</CardGroup>
