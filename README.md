# Belief Context Graph

This repository provides one Python package, `bcg`, with two integrated
applications:

- `bcg.construct` builds, serves, replays, and visualizes belief graphs.
- `bcg.agent` runs BeliefTracer benchmarks, evaluation workflows, graph sync,
  and the trajectory UI.

The shared construction engine extracts belief and decision nodes, links typed
relations, updates confidence, merges duplicates, and writes auditable local
artifacts. The SDK, batch CLI, HTTP service, and Agent integration all use this
same implementation.

## Install

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

### Project environment (recommended)

Clone this repository first, then create the project environment from the lock
file:

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph
uv sync --all-groups
```

This creates `.venv` inside the repository and is the recommended method for
development and testing.

### Optional user-level tool

If you only need the SDK and Construct commands, you can optionally install
`bcg` as a user-level command instead of activating `.venv`:

```bash
uv tool install .
bcg --version
```

This installation is isolated from the repository `.venv`. Agent support also
needs rLLM in the same tool environment; follow the optional Agent installation
below. The package always installs only one executable, `bcg`. Agent and
construction features are selected with subcommands rather than separate
system commands.

## Configuration

Create the root credential file and the non-secret model routing file:

```bash
cp .env.example .env
cp bcg/model_config.example.json bcg/model_config.json
```

Use the files for different purposes:

- `.env` is the only place for API keys and other credentials. It is ignored
  by Git.
- `bcg/model_config.json` contains non-secret model URLs and generation
  settings. Its `api_key_env` values refer to variable names in `.env`.

Importing `bcg` loads missing values from `.env` without overriding variables
already exported by the process.

With the optional `uv tool` installation, run `bcg` from the cloned repository
root so it finds `.env`, or export `BCG_ENV_FILE="$(pwd)/.env"` before invoking
it elsewhere.

## Command-line usage

Inspect the two command families with:

```bash
bcg --help
bcg agent --help
bcg construct --help
```

### Construct belief graphs

```bash
# Process a complete trajectory or dataset.
bcg construct run \
    --input data.json \
    --config bcg/model_config.json \
    --output-dir outputs

# Start the incremental HTTP service.
bcg construct server \
    --config bcg/model_config.json \
    --host 127.0.0.1 \
    --port 8848

# Replay recorded JSONL turns through the streaming constructor.
bcg construct replay \
    --input stream.jsonl \
    --config bcg/model_config.json

# Render a result as a self-contained HTML graph.
bcg construct visualize outputs/item/result.json --output graph.html
```

See [bcg/README.md](bcg/README.md) for input formats, HTTP endpoints, model
configuration, confidence behavior, and construction options.

### Run Agent workflows

BeliefTracer uses the rLLM workflow, tool, trajectory, and reward interfaces.
From the cloned `belief-context-graph/` directory, clone rLLM into the parent
directory so the two repositories are siblings:

```bash
cd ..
git clone https://github.com/rllm-org/rllm.git rllm
cd belief-context-graph
```

The directories should look like this:

```text
workspace/
  belief-context-graph/ # this repository/package
  rllm/                 # rllm-org/rllm checkout
    rllm-model-gateway/
```

For the project `.venv` created by `uv sync`, install both local rLLM packages
with uv:

```bash
uv pip install \
    --python .venv/bin/python \
    --editable ../rllm/rllm-model-gateway \
    --editable ../rllm

.venv/bin/python -c "import rllm; print(rllm.__file__)"
.venv/bin/bcg agent tasks
```

This installation is local to `.venv` and does not add the sibling checkout
to `pyproject.toml`. Activate that environment before running Agent commands:

```bash
source .venv/bin/activate

bcg agent run gpqa_diamond \
    --model <MODEL_PATH_OR_ID> \
    --belief-graph-url http://127.0.0.1:8848
bcg agent ui --artifacts-dir artifacts/belief_tracer
```

For the repository's preset AVeriTeC + HerO4 + Belief Graph rollout, configure
the Agent values in the root `.env` and run:

```bash
bash scripts/start.sh
```

The script loads `.env` and calls `bcg agent run` directly. It does not activate
Conda or invoke `scripts/rollout.sh`; additional arguments override its preset,
for example `bash scripts/start.sh --max-problems 2`.

#### Optional: install Agent as a user-level tool

If you prefer to use `bcg` without activating `.venv`, install the project and
the two local rLLM packages into one isolated uv tool environment:

```bash
uv tool install --force --refresh-package bcg . \
    --with-editable ../rllm/rllm-model-gateway \
    --with-editable ../rllm

bcg agent tasks
```

Use `--force --refresh-package bcg` again after changing or updating the BCG
source so uv rebuilds the local wheel instead of reusing a cached copy. The
rLLM packages use editable installation, so their local checkout must not be
moved or deleted.

Do not rely on `PYTHONPATH` or a Conda environment after using either uv
installation method above.

Local GPU backends such as `vllm`, `sglang`, `ray`, and a hardware-compatible
`torch` build are installed separately for the target machine. Remote
OpenAI-compatible backends do not require those local inference packages.

See [bcg/agent/README.md](bcg/agent/README.md) for benchmark data, retrieval,
evaluation, and rollout options.

The equivalent module entry points are:

```bash
python -m bcg.agent --help
python -m bcg.construct --help
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
  result.json
  final_graph.json
  belief_graph.jsonl
  belief_graph_latest.json
  trajectory.json
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

`bcg.llm.LLMClient` is an async OpenAI Responses API client with retry
handling, response parsing, image input support, and tool dispatch.

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

Belief confidence is deterministic and auditable. The construct engine assigns
a role/stance prior, then recomputes the posterior from additional evidence and
relation-activated factor contributions. Every change is recorded in
`confidence_history`; the LLM does not directly choose the final score.

## Tracing

`bcg.tracing` wraps Langfuse v4 instrumentation behind a project-local
`trace` decorator.

Tracing is active only when:

- `BCG_TRACING_ENABLED` is not set to `false`, `0`, `no`, or `off`
- `LANGFUSE_PUBLIC_KEY` is set
- `LANGFUSE_SECRET_KEY` is set

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. It
covers branch policy, Conventional Commits, local checks, pre-commit hooks,
AI-assisted PR transparency, and code style.

## Development

Run the local quality checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q bcg scripts tests
uv run pytest
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

The optional `dashboard/` directory contains the Vite graph-memory frontend.
It requires Node.js and is developed separately from the Python CLI:

```bash
cd dashboard
npm install
npm run dev
```

## Package Layout

```text
bcg/
  __init__.py       # public SDK exports
  cli.py            # the single `bcg` executable and top-level dispatcher
  env.py            # root .env discovery and loading
  graph.py          # public Pydantic graph model
  llm.py            # async OpenAI Responses API client
  memory.py         # BCGMemory facade
  runner.py         # BCGRunner lifecycle and artifact orchestration
  agent/
    __main__.py     # `python -m bcg.agent`
    cli.py          # run, ui, tasks, thinking check, and TongGraph sync
    prompts/        # packaged Agent prompts
    tools/          # search, retrieval, archive, and file tools
    evaluators/     # benchmark evaluation helpers
  construct/
    __main__.py     # `python -m bcg.construct`
    cli.py          # run, server, replay, and visualize dispatch
    pipeline.py     # batch and async SDK entry points
    stream.py       # canonical per-turn construction engine
    online.py       # concurrent trajectory/session driver
    graph.py        # internal belief graph representation
    confidence.py   # prior/evidence/factor confidence policy
    extract.py      # node and relation extraction
    merge.py        # incremental and final deduplication
  model_config.example.json
  py.typed          # PEP 561 typed package marker
dashboard/
  package.json      # optional Vite frontend
scripts/            # operational rollout, UI, and service launch helpers
tests/
  ...               # SDK, Agent, Construct, env, and CLI tests
```
