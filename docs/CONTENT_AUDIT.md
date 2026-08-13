# Content audit against the refactored repository

The documentation was checked against the following current source-of-truth
areas in the uploaded BCG repository:

- `README.md`
- `pyproject.toml`
- `Makefile`
- `bcg/__init__.py`
- `bcg/core/graph.py`
- `bcg/core/memory.py`
- `bcg/core/runner.py`
- `bcg/core/llm.py`
- `bcg/config/loader.py`
- `bcg/config/schema.py`
- `bcg/config/defaults.yaml`
- `bcg/config/config.example.yaml`
- `bcg/construct/`
- `bcg/construct/_shared/loaders.py`
- `bcg/construct/cli.py`
- `bcg/apps/cli.py`
- `bcg/apps/online_server.py`
- `bcg/apps/visualize_beliefs_graph.py`
- `contracts/http.schema.json`
- `dashboard/README.md`
- `dashboard/vite.config.ts`
- `dashboard/.env.example`

## Public root imports verified

```python
from bcg import (
    BCG,
    BCGMemory,
    BCGRunner,
    BCGSettings,
    LLMClient,
    load_settings,
)
```

## Runner lifecycle verified

```text
observe_trajectory
begin_belief_run
start_session
observe_turn
end_session
finalize
```

## Memory behavior verified

```text
observe  -> synchronous manual insertion, no LLM extraction
believe  -> case-insensitive substring match, typed payloads
context  -> structured dictionary, confidence-sorted
search   -> async interface, current in-memory substring implementation
```

## HTTP routes verified

```text
GET  /health
GET  /graph
POST /turn
POST /turns
POST /input
POST /run
POST /finalize
POST /release
```

## Known repository/documentation mismatch noted

The current top-level README contains a stale sentence saying `LLMClient` and
`LLMConfig` are available from `bcg.llm`. The implementation does not contain
that module. This documentation follows the implementation and package exports:
`LLMClient` from `bcg`, and `LLMConfig` from `bcg.core.llm`.


## Branded Fern revision

The second revision keeps the correctness audit above while changing
presentation and information architecture:

- 45 manually routed documentation/SDK pages;
- 8 OpenAPI HTTP routes;
- collapsed resource-oriented navigation;
- the original BCG color/typography system ported through custom CSS.
