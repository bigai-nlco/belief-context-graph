# Belief Context Graph

Belief Context Graph (`bcg`) is an Python SDK for a belief-native agent memory runtime.

The target system is a probabilistic, temporal, evidence-grounded memory substrate for autonomous agents. The current repository is still a scaffold: it defines package structure, graph model shells, memory interface stubs, an async LLM client, and Langfuse tracing utilities. It does not yet implement belief extraction, storage, graph mutation, retrieval, inference, decision gating, or calibration.

## Install

This project is managed with `uv`:

```bash
uv sync
```

The package requires Python 3.11+.

## Configuration

Copy `.env.example` when you need local API credentials:

```bash
cp .env.example .env
```

## Public Interfaces

The root package exports the graph model and memory interface:

```python
from bcg import BCG, BCGMemory

graph = BCG()
memory = BCGMemory(graph=graph)
```

`BCG` is a Pydantic graph container with `nodes` and `edges`. `BCGMemory` is a memory interface.

```python
memory.observe(
    source_type="message",
    content="Acme is threatening to churn after repeated outages.",
)
```

## LLM Client

`bcg.llm.LLMClient` is an async OpenAI Responses API client with retry handling, basic response parsing, image input support, and tool dispatch.

```python
from bcg.llm import LLMClient

client = LLMClient()
response = await client.generate(
    [{"role": "user", "content": "Summarize belief-native memory in one sentence."}]
)
print(response.content)
await client.close()
```

The client is disabled until either `OPENAI_API_KEY` or `OPENAI_BASE_URL` is set.

## Tracing

`bcg.tracing` wraps Langfuse v4 instrumentation behind a project-local `trace` decorator.

Tracing is active only when:

- `BCG_TRACING_ENABLED` is not set to `false`, `0`, `no`, or `off`
- `LANGFUSE_PUBLIC_KEY` is set
- `LANGFUSE_SECRET_KEY` is set

`LLMClient` currently traces:

- `bcg.llm.generate` as a chain
- `bcg.llm.image` as a chain
- `bcg.openai.responses.create` as a generation
- `bcg.llm.tool_call` as a tool span

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. It covers branch policy, Conventional Commits, local checks, pre-commit hooks, Codex review expectations, AI-assisted PR transparency, and code style.

## Development

Run the local quality checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q bcg tests
uv run python -m unittest discover -s tests
```

Format and auto-fix lint where possible:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run all pre-commit hooks manually:

```bash
uv run pre-commit run --all-files
```

## Dashboard

The `dashboard/` directory contains a minimal Vite scaffold for the future frontend. It is intentionally not implemented yet.

```bash
cd dashboard
npm install
npm run dev
```

## Package Layout

```text
bcg/
  __init__.py      # public exports
  api/
    __init__.py    # API namespace placeholder
  graph.py         # Pydantic graph, node, and edge shells
  llm.py           # async OpenAI Responses API client
  memory.py        # BCGMemory interface skeleton
  py.typed         # PEP 561 typed package marker
  tracing.py       # Langfuse tracing helpers
  utils.py         # shared utility functions
dashboard/
  index.html       # Vite entry HTML
  package.json     # frontend package scripts and dev dependencies
  src/
    main.ts        # placeholder app bootstrap
    style.css      # placeholder styles
tests/
  ...              # Unit module tests
```
