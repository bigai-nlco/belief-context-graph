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

- `.env` holds API keys plus machine-specific Agent and local-service settings.
  It is ignored by Git.
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
to `pyproject.toml`. Run the entry point from the uv-managed environment
directly; activation is not required:

```bash
.venv/bin/bcg agent run gpqa_diamond \
    --model <MODEL_PATH_OR_ID> \
    --belief-graph-url http://127.0.0.1:8848
.venv/bin/bcg agent ui --artifacts-dir artifacts/belief_tracer
```

For the repository's preset AVeriTeC + HerO4 + Belief Graph rollout, configure
the Agent values in the root `.env` and run:

```bash
bash scripts/start.sh
```

The script loads `.env`, prefers `.venv/bin/bcg`, and falls back to a `bcg`
installed on `PATH` with `uv tool install`. Additional arguments override the
named preset, for example `bash scripts/start.sh --max-problems 2`.

#### Run BrowseComp and GAIA

The following examples use an OpenAI-compatible DeepSeek endpoint, live web
search through Serper, and the Belief Graph service. Configure these values in
the root `.env` first:

- `OPENAI_BASE_URL` and `OPENAI_API_KEY`
- `SERPER_API_KEY`
- `BELIEF_GRAPH_URL`
- optionally `BROWSECOMP_GRADER_*` to use a separate BrowseComp judge; otherwise
  the judge reuses the rollout model and endpoint

Set the machine-local values used by the commands. Sourcing `.env` exports
`BELIEF_GRAPH_URL` for the explicit CLI argument; `bcg` also loads the root
`.env` itself.

```bash
set -a
source .env
set +a

# Defaults to a portable repository-local data directory. Override it in .env.
BENCHMARKS_DIR="${BENCHMARKS_DIR:-$PWD/datasets}"

# API-side model ID comes from the same root configuration as other Agent runs.
MODEL_ID="${MODEL:-${OPENAI_MODEL:-}}"
test -n "$MODEL_ID" || { echo "Set MODEL or OPENAI_MODEL in .env" >&2; exit 2; }
```

Prepare the public BrowseComp and GAIA datasets in that directory before the
first run:

```bash
uv run --frozen python scripts/prepare_web_benchmarks.py \
    --data-root "$BENCHMARKS_DIR"
```

BrowseComp-Plus uses a separate dense corpus index. Configure an embedding
endpoint, choose a local output directory, and build it with:

```bash
BCP_INDEX_DIR="${BCP_INDEX_DIR:-$PWD/datasets/browsecomp_plus/indexes/hero}"
uv run --frozen python scripts/build_bcp_dense_index.py \
    --output-dir "$BCP_INDEX_DIR" \
    --embedding-url "$HERO_EMBEDDING_URL" \
    --model-name "$HERO_EMBEDDING_MODEL"
```

Run one verified, attachment-free GAIA validation task. It is a text/web-search
question and does not require a multimodal model:

```bash
GAIA_SPLIT=validation uv run --frozen bcg agent run gaia \
    --model "$MODEL_ID" \
    --backend api \
    --benchmarks-dir "$BENCHMARKS_DIR" \
    --task-ids 8e867cd7-cff9-4e6c-867a-ff5ddc2550be \
    --tools serper_search serper_scrape \
    --context-memory-mode belief_graph \
    --belief-graph-url "$BELIEF_GRAPH_URL" \
    --belief-graph-interval 3 \
    --max-steps 12 \
    --num-samples 1 \
    --n-parallel-tasks 1 \
    --no-auto-ui \
    --output-dir artifacts/gaia_deepseek_graph_smoke
```

Run one deterministic BrowseComp smoke-test sample. `browsecomp` and
`browse_comp` are aliases for the same benchmark; use `browsecomp` consistently
in new commands:

```bash
uv run --frozen bcg agent run browsecomp \
    --model "$MODEL_ID" \
    --backend api \
    --benchmarks-dir "$BENCHMARKS_DIR" \
    --max-problems 1 \
    --shuffle-seed 0 \
    --tools serper_search serper_scrape \
    --context-memory-mode belief_graph \
    --belief-graph-url "$BELIEF_GRAPH_URL" \
    --belief-graph-interval 3 \
    --max-steps 12 \
    --num-samples 1 \
    --n-parallel-tasks 1 \
    --no-auto-ui \
    --output-dir artifacts/browsecomp_deepseek_graph_smoke
```

Operational notes:

- Remove `--task-ids` or `--max-problems` to run the complete selected split.
  Full BrowseComp contains 1,266 tasks; GAIA validation contains 165 tasks.
- `GAIA_SPLIT` accepts `validation` or `test`, and `GAIA_LEVEL` accepts `all`,
  `1`, `2`, or `3`. The public validation split has reference answers; the test
  split does not.
- Some GAIA tasks contain local attachments. For text-compatible files, add
  `--enable-file-read --file-tool-root "$BENCHMARKS_DIR"`. Image, audio, video,
  or other multimodal attachments still require a capable model/tool, so use
  `--task-ids` to curate text-only tasks for a text-only DeepSeek endpoint.
- `--backend api` uses BCG's compatibility layer, which fills the standard
  `tool_calls.type`, `id`, and `function` fields required by stricter API
  providers. Prefer it over `--backend openai` for the tested DeepSeek endpoint.
- `--belief-graph-interval 1` rebuilds graph context after every model turn and
  can make long BrowseComp runs extremely slow. Start with `3`; lower it only
  when every-turn graph updates are required by the experiment.
- Serper pages and snippets are untrusted evidence. BrowseComp in particular can
  attract SEO-spam results, so inspect trajectories before treating a scored
  answer or a high-confidence graph belief as reliable.

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

Do not rely on `PYTHONPATH`; use one of the uv-managed environments above.

Local GPU backends such as `vllm`, `sglang`, `ray`, and a hardware-compatible
`torch` build are installed separately into the project `.venv` for the target
machine, for example `uv pip install --python .venv/bin/python vllm` or
`uv pip install --python .venv/bin/python sglang`. Remote OpenAI-compatible
backends do not require those local inference packages.

The GPU launch scripts load `.env` and use `VLLM_*` or `SGLANG_*` variables.
They intentionally do not reuse `MODEL`, which identifies the Agent's remote
API model.

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
