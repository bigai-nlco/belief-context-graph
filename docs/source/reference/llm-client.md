---
title: "LLM Client"
description: "OpenAI-compatible model and embedding clients used by the public SDK."
icon: "bot"
---

```python
from bcg.llm import (
    EmbeddingClient,
    EmbeddingConfig,
    LLMClient,
    LLMConfig,
    TokenUsageTracker,
)
```

## `LLMClient`

Async methods:

- `generate(messages, ...)`
- `generate_text(prompt, ...)`
- `generate_json(prompt, ...)`
- `image(image_bytes, ...)`
- `close()`

The client uses the OpenAI Responses API shape, retries retryable failures, supports tool dispatch, and records usage.

## Example

```python
from bcg.llm import LLMClient, LLMConfig

client = LLMClient(
    LLMConfig(
        api_key="...",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
    )
)

text = await client.generate_text(
    "Return one sentence describing the incident.",
    label="incident_summary",
)
```

## `EmbeddingClient`

```python
vectors = embedder.embed(
    ["first belief", "second belief"],
    purpose="incremental_merge",
)
```

`EmbeddingConfig` makes provider, model, dimensions, batch size, and local settings explicit.

## Token usage

`TokenUsageTracker` aggregates records by label and total. BCG writes usage artifacts when the backend returns token counts; otherwise it may estimate counts from text.
