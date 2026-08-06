---
title: "Choose Your Path"
description: "Select the BCG interface that matches your application."
icon: "route"
---

BCG exposes several entry points over the same belief graph model.

| You need to… | Start with |
|---|---|
| Experience BCG interactively | [Reference Agent](/reference-agent) |
| Add graph construction to Python | [Python SDK](/guides/python-sdk) |
| Send turns from another language | [Streaming HTTP server](/guides/streaming-server) |
| Process saved trajectories or datasets | [Batch construction](/guides/batch-construction) |
| Reprocess an NDJSON stream | [Replay streams](/guides/replay-streams) |
| Inspect graph growth and evidence | [Live Stream Viewer](/guides/live-stream-viewer) |
| Compare BCG against raw context | [Benchmarking](/guides/benchmarking) |
| Insert known beliefs without an LLM | [Manual beliefs](/guides/manual-beliefs) |

## Decision guide

<AccordionGroup>

<Accordion title="I already have an agent framework">
Use the Python SDK when your runtime is Python. Otherwise run `bcg construct server` and push messages over `/turn` or `/turns`. Keep your own tools, UI, model routing, and agent loop.
</Accordion>

<Accordion title="I want BCG to manage a reference experience">
Run `bcg setup`, choose a graph backend, and then launch `bcg`. The included Agent demonstrates the BCG context policy but is not required.
</Accordion>

<Accordion title="I only need to inspect completed output">
Run `bcg construct visualize path/to/result.json`, or open the live viewer and select an output directory.
</Accordion>

<Accordion title="I already have structured beliefs">
Use `BCGMemory.observe()`. It inserts the full supplied content as one asserted belief and performs no model call.
</Accordion>

</AccordionGroup>
