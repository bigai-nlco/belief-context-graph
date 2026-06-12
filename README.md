# Belief Context Graph

Belief Context Graph (`bcg`) is an Python SDK for a belief-native agent memory runtime.

The target system is a probabilistic, temporal, evidence-grounded memory substrate for autonomous agents. The current library implements trajectory segmentation, belief extraction, relation linking, confidence updates, graph assembly, local run artifacts, and a lightweight memory facade. Retrieval and inference APIs are intentionally still minimal.

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

The root package exports the graph model, memory interface, and run orchestrator:

```python
from bcg import BCG, BCGMemory, BCGRunner

graph = BCG()
memory = BCGMemory(graph=graph)
```

`BCG` is a Pydantic graph container with `nodes` and `edges`. `BCGMemory`
stores and searches memory. `BCGRunner` owns belief-run lifecycle, run ids,
sessions, and local output paths.

```python
observation = memory.observe(
    source_type="message",
    content="Acme is threatening to churn after repeated outages.",
)
```

For multi-turn trajectories, use the async belief-graph runner:

```python
from bcg.llm import LLMClient

memory = BCGMemory(graph=BCG())
runner = BCGRunner(memory=memory, llm=LLMClient())
result = await runner.observe_trajectory(
    [{"role": "user", "content": "Acme is threatening to churn."}],
)
print(result.output_paths.memory)
```

Run outputs are written under `.bcg/runs/<run_id>/`:

```text
.bcg/runs/<run_id>/
  graph.json
  memory.json
  token_usage.json
  events.jsonl
  artifacts/
    segments.json
    io_beliefs.json
    reasoning_beliefs.json
    forward_relations.json
    backward_relations.json
    merges.json
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

LLM API configuration uses:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_TIMEOUT`
- `OPENAI_MAX_RETRIES`

Optional embedding calls for semantic split/merge use:

- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_API_KEY`


## Incremental Belief Memory

`BCGMemory` is the stable memory-facing protocol. `BCGRunner` owns online
construction: start a run, ingest turns as they arrive, end sessions when
appropriate, and finalize when you want a local snapshot:

```python
memory = BCGMemory(graph=BCG())
runner = BCGRunner(memory=memory, llm=LLMClient())
runner.begin_belief_run(run_id="demo")
runner.start_session("session-1", "2026-06-12")
await runner.observe_turn("user", "Alice likes green tea.")
await runner.end_session()
result = await runner.finalize()
```

Pipeline behavior such as semantic splitting, merge strategy, context budgets,
run id, and output root is configured with Python arguments on `BCGRunner` /
`BeliefGraphPipeline`, not hidden environment defaults.

Belief confidence is deterministic and auditable. The stored confidence value is
computed as the average of four dimensions: `source_reliability`,
`evidence_directness`, `claim_specificity`, and `linguistic_certainty`.
LLMs do not generate the final confidence number.

## Tracing

`bcg.tracing` wraps Langfuse v4 instrumentation behind a project-local `trace` decorator.

Tracing is active only when:

- `BCG_TRACING_ENABLED` is not set to `false`, `0`, `no`, or `off`
- `LANGFUSE_PUBLIC_KEY` is set
- `LANGFUSE_SECRET_KEY` is set

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. It covers branch policy, Conventional Commits, local checks, pre-commit hooks, Codex review expectations, AI-assisted PR transparency, and code style.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. It covers branch policy, Conventional Commits, local checks, pre-commit hooks, Codex review expectations, AI-assisted PR transparency, and code style.

## Development

Run the local quality checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q bcg scripts tests
uv run pytest tests/test_belief_graph.py
uv run python -m unittest discover -s tests -p 'test_tracing.py'
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

The `dashboard/` directory contains the Vite graph-memory dashboard merged from `main`.

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
  graph.py         # Pydantic graph, typed belief payloads, and relation edges
  llm.py           # async OpenAI Responses API client
  memory.py        # BCGMemory storage and search facade
  runner.py        # BCGRunner lifecycle and output orchestration
  belief_graph/
    constants.py   # shared non-prompt defaults and validation constants
    confidence.py  # dimension-based confidence assessment and updates
    evidence.py    # exact-offset evidence provenance
    extraction.py  # per-segment belief extraction
    linking.py     # forward/backward relation validation and linking
    merge.py       # optional duplicate belief merge pass
    pipeline.py    # end-to-end graph construction pipeline
    prompts.py     # prompt registry
    segment.py     # trajectory segmentation
    split.py       # optional sentence split and semantic clustering
    utils.py       # shared JSON output, run-id, span, and counting helpers
  py.typed         # PEP 561 typed package marker
  tracing.py       # Langfuse tracing helpers
  utils.py         # shared utility functions
scripts/
  visualize_belief_graph.py  # temporary HTML visualizer for BCG run JSON
dashboard/
  index.html       # Vite entry HTML
  package.json     # frontend package scripts and dev dependencies
  src/
    main.ts        # placeholder app bootstrap
    style.css      # placeholder styles
tests/
  ...              # Unit module tests
```
