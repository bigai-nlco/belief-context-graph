
<div align="center">

# Belief Context Graph


**Probabilistic · Temporal · Explainable · Stateful**

[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/uv-managed-6E4BF9.svg?style=flat-square)](https://docs.astral.sh/uv/)
[![Documentation](https://img.shields.io/badge/documentation-online-0F766E.svg?style=flat-square&logo=readthedocs&logoColor=white)](https://belief-context-graph.docs.buildwithfern.com/)

</div>

<p align="center">
  <img src="assert/benchmark_browsecomp.svg" width="49%" alt="BrowseComp full-dataset dual-axis comparison of accuracy and mean token cost per task">
  <img src="assert/benchmark_browsecomp_zh.svg" width="49%" alt="BrowseComp-ZH full-dataset dual-axis comparison of accuracy and mean token cost per task">
</p>

## **Why Belief Context Graph**
**Current agent memory systems fall short.** Conversation memory preserves history. Vector memory retrieves similar fragments. GraphRAG extracts entities and relations. Trace memory records tool calls. Temporal KGs track facts over time.

> These systems answer **retrieval questions**: *which text is relevant? which entities are related? what happened in the past?*

But agents executing real tasks also need to answer **belief questions**:

> - Should I actually believe this fact?
> - Is it still valid, or has it expired?
> - Did it come from a reliable source?
> - Does it conflict with other evidence?
> - Is it certain enough to act on?
> - Did the outcome prove my prior belief wrong?


Belief Context Graph (`BCG`) upgrades agent memory from **retrieval memory** to **belief context graph**. It is a probabilistic, temporal, evidence-grounded and computational memory substrate that helps agents continuously maintain: what to believe, at what confidence, from which evidence, and whether uncertainty should block action. The result is agent memory you can query, audit, and trust.

---

<p align="center">
  <strong><a href="#live-demo">Live Demo</a></strong> &nbsp;·&nbsp;
  <strong><a href="#core-capabilities">Core Capabilities</a></strong> &nbsp;·&nbsp;
  <strong><a href="#quick-start">Quick Start</a></strong> &nbsp;·&nbsp;
  <strong><a href="#architecture">Architecture</a></strong> &nbsp;·&nbsp;
  <strong><a href="#core-concepts">Core Concepts</a></strong> &nbsp;·&nbsp;
  <strong><a href="#benchmarking-results">Benchmark Results</a></strong> &nbsp;·&nbsp;
  <strong><a href="#comparison-with-existing-memory-solutions">Comparison</a></strong> &nbsp;·&nbsp;
  <strong><a href="#contributing">Contributing</a></strong>
</p>

---
## Live Demo

<div align="center">

</div>

## Token cost across the task horizon

<p align="center">
  <a href="assert/readme_token_cost.png">
    <img src="assert/readme_token_cost.png" width="100%" alt="Cumulative token cost">
  </a>
</p>

- **Model calls.** The number of model calls (assistant turns) a task has made so far. Moving right means progressing deeper into the task horizon. The full-horizon figure shows the overall trend, while the accompanying zoomed-in figure focuses on the first 20 calls to make the early-phase behavior and crossover easier to see.
- **Mean cumulative tokens per task.** Tokens summed from the first call up to and including call *k*, then averaged over all 1,200 trajectories in that mode.

## **Core capabilities:**

- **Belief Extraction:** Segment trajectories, extract structured beliefs that counts for agent reasoning and link them into a connected graph
- **Deterministic Confidence:** Auditable posterior confidence computed from `initial_confidence`, `evidence_confidence`, and relation-derived `factor_confidence`; source reliability and stance quality set the prior, while relation weights and activation thresholds propagate support or contradiction deterministically
- **Evidence Provenance:** Every belief carries exact-offset source references back to the originating conversation turn
- **Temporal Awareness:** Run-based lifecycle with sessions and timestamps — know when each belief was formed and how it evolved
- **Relation Linking:** Forward and backward relationship edges between beliefs, forming a casual decision graph/trace.


## Quick Start

Core BCG uses Python/`uv` for the SDK and Graph Construction. The optional reference terminal Agent uses an isolated Node.js 22.19+ runtime.

### 1. Install BCG

Two supported paths, kept in lockstep with `install.sh` and `Makefile`
(see ADR-0001 for the release versioning policy and the release manifest):

| Path | Audience | Locked deps | Commands |
|---|---|---|---|
| `install.sh` (release) | end users | `uv.lock` + `package-lock.json` | `curl .../install.sh \| sh` |
| `make install` (source) | developers | `uv.lock` + both `package-lock.json` | `make install` |

#### Option A: install globally with curl (release)

```bash
curl -LsSf https://raw.githubusercontent.com/bigai-nlco/belief-context-graph/main/install.sh | sh
bcg --version
```

The installer requires `curl`, `tar`, npm, and Node.js 22.19 or newer. It
installs `uv` when necessary, downloads the repository into a temporary
directory, installs the Python and Node runtimes from their lockfiles, and
then removes the temporary source. Download failures abort before any
install step; partial installs are detected and PATH guidance is printed.

#### Option B: clone and run from source (development)

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph
make install          # uv sync --locked --all-groups + agent/dashboard npm ci + agent build

uv run bcg --version
```

Run the source checkout with `uv run bcg`. Optionally expose the current
checkout as the global `bcg` command:

```bash
make install-tool     # uv tool install . + agent build + npm install -g ./agent-cli
bcg --version
```

The Node package provides the internal `bcg-agent` executable launched by
`bcg`; users normally do not invoke it directly. The Dashboard is a
separately deployed release artifact (not part of these installs); see
`dashboard/README.md`.

### 2. Start the reference BCG Agent

This step is optional. It launches the bundled reference Agent so you can test BCG immediately; integrating BCG into another Agent does not require using this CLI.

For a curl or global installation:

```bash
bcg
```

From a source checkout:

```bash
uv run bcg
```

On the first launch, the setup guide asks for:

1. Agent authentication: an OpenAI-compatible API key and base URL, or the interactive `/login` flow.
2. The Agent model.
3. Optional Serper web search credentials, required for BrowseComp and used by the `web_search` tool.
4. The default context mode: **BCG** or **Default**.
5. Whether BCG should manage a local Graph Construction server or connect to an existing one.
6. For a managed server, the Graph backend: **unified** or **hybrid**.

The setup is saved under `~/.bcg` and works from every directory:

```text
~/.bcg/config.json        # Agent, context, and Graph runtime choices
~/.bcg/.env               # API keys and credentials; mode 0600
~/.bcg/config.yaml        # Unified YAML settings: models, pipeline, runner (see bcg/config/config.example.yaml)
~/.bcg/agent/             # Agent settings, authentication, and sessions
```

Run `bcg setup` at any time to change these settings.

When a managed Graph backend is selected, `bcg` automatically:

1. Checks `http://127.0.0.1:8848/health`.
2. Reuses a healthy Graph Construction server or starts one with the selected backend and `~/.bcg/config.yaml`.
3. Writes its log to `~/.bcg/logs/graph-server.log` and graph artifacts to `~/.bcg/graphs/`.
4. Opens the terminal Agent after the Graph server is ready.

`bcg agent` is an explicit alias for the same terminal interface.

The terminal exposes the following commands:

| Command | Purpose |
| --- | --- |
| `/help` | Show commands and essential keyboard controls |
| `/model` | Select the inference model |
| `/mode` | Choose Default or BCG context before the session's first message |
| `/login` / `/logout` | Configure or remove the model API key |
| `/new` / `/resume` | Start or restore a BCG session |
| `/graph` | Check Graph connectivity and context policy |
| `/exit` | Exit BCG |

Context mode is fixed after the first user message. Use `/new` to start a session with another mode:

- **BCG** permanently retains the initial user input and keeps the latest two completed turns as raw messages. On the first request, the system prompt and initial user input seed the Graph. Messages leaving the two-turn raw window are then added incrementally. The current Graph is encoded with a dialogue context template (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, and `<｜Assistant｜>` markers with Markdown belief payloads) and appended to the system prompt. Traditional compaction is disabled.
- **Default** uses the full normal Agent conversation with automatic compaction. Graph context is not injected.

Use `/mode` for the selector, or `/mode bcg` and `/mode default` directly. `BCG_RECENT_TURNS` can override the default raw window of `2`.

### 3. Configure and Start Graph Construction

BCG supports two construction backends. In normal Agent use, you choose one during `bcg setup` and `bcg` starts or reuses the Graph Construction HTTP server automatically. The commands below are also provided for independent deployment and debugging.

#### Option A: `unified`

`unified` uses one OpenAI-compatible model for graph node and relation generation. During setup, it can reuse the Agent model endpoint and API key or use a separate endpoint:

```bash
bcg setup
# Graph server: Start and manage a local Graph server automatically
# Graph backend: Unified

bcg
```

No vLLM process is required when the configured API endpoint is already available.

The equivalent manual Graph server command is:

```bash
bcg construct server unified \
  --config ~/.bcg/config.yaml \
  --model-key graph-model \
  --embedding-key embedding \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir ~/.bcg/graphs
```

If that server is already healthy, a later `bcg` invocation reuses it.

The `unified` backend provides two switchable construction modes under `pipeline.runtime.construction_mode` in `~/.bcg/config.yaml`:

```yaml
pipeline:
  runtime:
    construction_mode: llm  # canonical/default implementation
  token_efficient:
    max_search_results: 10
    max_snippet_chars: 240
    semantic_tool_results: true
    max_facts: 3
    max_semantic_calls: 12
```

Set the mode to `token_efficient` to parse canonical Agent tool calls and tool results in code, distill a bounded number of non-empty tool results with a short current-query-only prompt, create provenance edges deterministically, and use embedding-only merge without LLM verification. Raw tool output remains attached as evidence. After `max_semantic_calls`, later tool results automatically use the zero-LLM rule path; set `semantic_tool_results: false` to use that path from the start. Set `construction_mode` back to `llm` and restart the Graph server to restore the canonical model-extraction and model-linking implementation.

The Agent graph presentation is independently switchable. `full` preserves the complete belief-and-relation dialogue view. `compact` keeps the same chat markers and selects whole beliefs under a fixed character budget. It omits the duplicated initial question but never rewrites belief text, synthesizes query/result mappings, or introduces renderer-only node concepts:

```bash
bcg benchmark run browsecomp \
  --modes bcg \
  --graph-view compact \
  --recent-turns 2
```

For interactive use, set `BCG_GRAPH_VIEW=compact`; unset it or use `full` to restore the complete graph rendering.

#### Option B: `hybrid`

`hybrid` uses:

- a small generative model served through an OpenAI-compatible vLLM endpoint;
- a local sentence-transformers embedding model;
- a local/Hugging Face stance classifier;
- spaCy for local language processing and entity extraction.

The BCG Python installation contains the Graph-side dependencies, but it does not install or manage the separate vLLM GPU service.

First create a dedicated vLLM environment:

```bash
uv venv ~/.bcg/vllm --python 3.11
uv pip install --python ~/.bcg/vllm/bin/python vllm
```

Start vLLM in another terminal or a persistent service such as tmux:

```bash
~/.bcg/vllm/bin/vllm serve <MODEL_OR_LOCAL_PATH> \
  --served-model-name <MODEL_NAME> \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 10000 \
  --max-num-seqs 8
```

Then configure BCG with:

```text
Graph server: Start and manage a local Graph server automatically
Graph backend: Hybrid
vLLM base URL: http://127.0.0.1:8001/v1
Model served by vLLM: <MODEL_NAME>
vLLM API key: EMPTY
```

`<MODEL_NAME>` must match the value passed to `--served-model-name`. After vLLM is ready, launch the Agent:

```bash
bcg
```

BCG starts the hybrid Graph Construction server at `127.0.0.1:8848`; it does not start the vLLM process. To start Graph Construction manually instead:

```bash
bcg construct server hybrid \
  --config ~/.bcg/config.yaml \
  --model-key graph-model \
  --embedding-key embedding \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir ~/.bcg/graphs
```

For source development, `scripts/start_vllm.sh` is also available, but it reads `VLLM_*` values from the checkout's root `.env` or explicit command-line arguments; it does not read `~/.bcg/config.yaml`.

#### Connect to an existing Graph server

If Graph Construction is already hosted elsewhere, run `bcg setup`, choose **Connect to an existing Graph server**, and enter its URL. In this mode BCG checks and reuses that endpoint but does not start or manage it.

### 4. Run Graph Construction Without the Agent

The same two backends can process saved trajectories:

```bash
bcg construct run unified \
  --input data.json \
  --config ~/.bcg/config.yaml \
  --model-key graph-model

bcg construct replay hybrid \
  --input stream.jsonl \
  --config ~/.bcg/config.yaml \
  --model-key graph-model
```

See [bcg/README.md](bcg/README.md) for input formats, HTTP endpoints, output artifacts, and Python APIs.

## Python SDK

The Python SDK exposes the graph data model, memory operations, construction lifecycle, and model client as regular Python objects. Use it when BCG needs to run inside another Python application without going through the terminal Agent or the Graph Server HTTP API.

### Core Classes

| Class | Purpose |
|---|---|
| `BCG` | The in-memory Belief Context Graph. It stores typed belief nodes, relation edges, evidence, merge history, sessions, and graph metadata. Use it to inspect, serialize, or modify a graph directly. |
| `BCGMemory` | The application-facing memory wrapper around a `BCG` graph. It supports manual belief insertion with `observe()`, substring lookup with `believe()`/`search()`, and task-context assembly with `context()`. Manual `observe()` treats the supplied content as one asserted belief; it does not call an LLM. |
| `BCGRunner` | The graph-construction orchestrator. It sends complete trajectories or individual turns through the `unified` or `hybrid` construction backend, synchronizes the resulting graph into `BCGMemory`, tracks sessions, and writes run artifacts. |
| `LLMClient` | The asynchronous OpenAI-compatible model client used by `BCGRunner`. It reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from the environment or BCG configuration, and records model token usage. |

`BCG`, `BCGMemory`, and `BCGRunner` are exported directly from `bcg`. `LLMClient` and its explicit `LLMConfig` are available from `bcg.llm`.

### Example: Manual Belief Storage

Use `BCGMemory.observe()` when the input is already a belief and does not need model-based extraction:

```python
from bcg import BCG, BCGMemory

graph = BCG()
memory = BCGMemory(graph=graph)

observation = memory.observe(
    source_type="message",
    content="Acme is threatening to churn after repeated outages.",
)

print(observation.belief.id)
print(observation.belief.belief)
print(memory.context(task="Review customer churn risk"))
```

This example runs locally and does not require an API key. The complete `content` string becomes one asserted belief node.

### Example: Build a Graph from a Conversation

Use `BCGRunner` when raw messages need to be segmented and converted into beliefs, decisions, evidence, confidence values, and relations by a construction model.

Configure the OpenAI-compatible endpoint first:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"
```

Then run the following as a normal Python program:

```python
import asyncio

from bcg import BCG, BCGMemory, BCGRunner
from bcg.core.llm import LLMClient


async def main() -> None:
    memory = BCGMemory(graph=BCG())
    runner = BCGRunner(
        memory=memory,
        llm=LLMClient(),
        backend="unified",  # use "hybrid" for the hybrid construction backend
    )

    result = await runner.observe_trajectory(
        [
            {
                "role": "user",
                "content": "Acme is threatening to churn after repeated outages.",
            },
            {
                "role": "assistant",
                "content": "We should prioritize a reliability review and contact Acme.",
            },
        ],
        run_id="acme-risk-review",
    )

    print(f"beliefs: {len(result.graph.beliefs())}")
    print(f"relations: {len(result.graph.relations())}")
    print(f"memory artifact: {result.output_paths.memory}")
    print(f"token usage: {result.token_usage}")


asyncio.run(main())
```

`observe_trajectory()` starts and finalizes one construction run automatically. Its result contains the final graph, a memory snapshot, output paths, token usage, and node/relation counts.

### Example: Control Sessions and Turns

For a streaming application, manage the lifecycle explicitly and push turns as they arrive:

```python
import asyncio

from bcg import BCG, BCGMemory, BCGRunner
from bcg.core.llm import LLMClient


async def main() -> None:
    runner = BCGRunner(
        memory=BCGMemory(graph=BCG()),
        llm=LLMClient(),
    )

    runner.begin_belief_run(run_id="preference-demo")
    runner.start_session("session-1", "2026-06-12")
    await runner.observe_turn("user", "Alice likes green tea.")
    await runner.observe_turn(
        "assistant",
        "Noted. I'll remember that preference.",
    )
    await runner.end_session()
    result = await runner.finalize()

    print(result.graph.model_dump_json(indent=2))


asyncio.run(main())
```

### Run Output Artifacts

Each run produces a structured directory under `.bcg/runs/<run_id>/`:

```text
.bcg/runs/<run_id>/
  graph.json                # full belief graph with nodes and edges
  memory.json               # memory facade snapshot
  token_usage.json          # LLM token consumption
  events.jsonl              # timestamped event log
  artifacts/
    segments.json           # trajectory segmentation
    io_beliefs.json          # extracted input/output beliefs
    reasoning_beliefs.json   # extracted reasoning beliefs
    forward_relations.json  # forward relation edges
    backward_relations.json # backward relation edges
    merges.json             # duplicate belief merge decisions
```

---

## Architecture

The belief graph construction pipeline processes each conversation turn incrementally. Merge runs before relation linking so edges are created against the surviving canonical nodes:

```text
Turn input ──▶ Split / chunk ──▶ Extract nodes + initialize confidence ──▶ Merge ──▶ Link relations ──▶ Propagate relation confidence ──▶ BCG graph
```

| Stage | Actual implementation | Description |
|---|---|---|
| **Segmentation** | `bcg.construct.unified.split.split_sentences` / `bcg.construct.hybrid.split.semantic_chunks_isolating_tool_calls` | Splits a turn into sentence evidence (`unified`) or optional semantic chunks with isolated tool calls (`hybrid`) |
| **Extraction** | `bcg.construct.unified.extract.extract_nodes` / `bcg.construct.hybrid.extractor.QwenChunkExtractor.extract_turn` | Extracts belief and decision nodes from the current turn |
| **Confidence** | `bcg.construct.unified.confidence.init_belief_confidence` / `bcg.construct.hybrid.confidence.init_belief_confidence` | Initializes `initial_confidence` from source role and stance; later merged evidence updates `evidence_confidence`, and active relations update `factor_confidence` |
| **Merge** | `bcg.construct.unified.merge.run_merge_pass` / `bcg.construct.hybrid.merge.run_merge_pass` | Deduplicates belief nodes before relation generation and rewires existing relation endpoints |
| **Linking** | `bcg.construct.unified.extract.extract_relations` / `bcg.construct.hybrid.edge_generation.QwenEdgeGenerator.generate_window` | Generates, validates, and adds typed relations between surviving nodes; confidence-carrying edges include `weight` and `activated_condition`, while `supplements` keeps both fields as `null` |

`BCGRunner` is the public orchestration layer. It delegates each run to the selected backend's `StreamingTrajectorySession`, whose `StreamingBeliefBuilder` executes the stages above. `BCGMemory` is the user-facing memory facade for manually observing already-formed beliefs and reading or searching the resulting graph; it does not implement the construction stages itself. Context budgets, merge strategy, run IDs, and output paths are configured explicitly through `BCGRunner` and backend options.

---

## Core Concepts

### Belief

A belief is the fundamental unit of knowledge in BCG. Each belief carries:

- **Typed payload** — structured data describing what the agent believes (facts, tool call response, reasoning steps, etc.)
- **Confidence score** — deterministic posterior computed from the node prior, merged evidence contribution, and relation-propagated factor contribution
- **Evidence provenance** — exact character-offset references back to the source turn
- **Temporal metadata** — session ID, turn index, and timestamp of formation

### Confidence Assessment

Belief confidence is **deterministic and auditable**. The current design recomputes confidence from explicit graph fields:

```text
confidence = sigmoid(
  logit(initial_confidence)
  + evidence_confidence
  + factor_confidence
)
```

| Component | What It Measures |
|---|---|
| `initial_confidence` | The node prior derived from source reliability and stance quality |
| `evidence_confidence` | Additional evidence contribution accumulated when duplicate evidence is merged into a canonical node |
| `factor_confidence` | Relation-propagated support or contradiction from active `depends_on` and `contradicts` edges |

`depends_on` relations contribute positive factor confidence, `contradicts` relations contribute negative factor confidence, and `supplements` relations are semantic-only and do not propagate confidence. Confidence propagation is controlled by relation `weight`, `activated_condition.input_conf_threshold`, `propagation_min_confidence_delta`, and `max_propagation_iterations` in the model config. No model-generated confidence score is accepted directly into the graph.

### Evidence

Every belief links to its source via **exact character-offset provenance** (`bcg.belief_graph.evidence`). You can trace any belief back to the precise span of text in the original conversation that produced it.

### Relations

Beliefs connect through typed, directed edges:

- **Forward relations** — a belief implies, causes, or supports another
- **Backward relations** — a belief is implied by, caused by, or supported by another

Relations are validated during linking to ensure graph consistency.

### Run Lifecycle

A **run** (`BCGRunner`) is the top-level lifecycle orchestrator. Each run contains one or more **sessions**, and each session contains a sequence of **turns** (user/assistant messages). The runner tracks run IDs, session boundaries, and output paths — giving you full temporal and structural context for every belief.

---

## Benchmark Adapter

The reference Agent can evaluate **BrowseComp**, **GAIA**, **HotpotQA**, and **MMLU-Pro** in Default mode, BCG mode, or both. The same Agent model, prompt, tool policy, task selection, and scorer are used for both modes; only context management changes.

Prepare the data under one root:

```text
datasets/
├── browse_comp/data.json
├── gaia/2023/validation/metadata.jsonl
├── hotpotqa/data.json
└── mmlu_pro/data.json
```

JSON, JSONL, and CSV are supported without extra packages. Parquet input requires:

```bash
uv sync --extra benchmarks
```

Each path may instead be supplied explicitly with a repeatable `--data-file BENCHMARK=PATH` option. The loader accepts the native field names used by the official datasets, including GAIA attachments, HotpotQA `_id`, and MMLU-Pro's A–J options.

Configure the Agent and web search in `~/.bcg/.env` or the current environment:

```bash
OPENAI_BASE_URL=https://your-openai-compatible-server/v1
OPENAI_API_KEY=...
OPENAI_MODEL=your-agent-model
SERPER_API_KEY=...  # BrowseComp, HotpotQA, and online GAIA research
SERPER_MAX_CALLS=20 # Hard web_search budget per Agent session
```

Then run the same selected examples in both context modes:

```bash
bcg benchmark run browsecomp gaia hotpotqa mmlu_pro \
  --modes default,bcg \
  --thinking off \
  --max-problems 100 \
  --workers 8 \
  --gaia-split validation \
  --gaia-text-only \
  --output-dir results/four-benchmark-comparison
```

Use `--graph-view compact` in BCG mode to inject the low-token belief projection while retaining the dialogue-style chat markers. The default is `full`, so existing runs keep the complete graph rendering unless this option is selected explicitly.

Use `--thinking medium` (or another supported level) to set the Agent's reasoning effort for the run. This setting is independent of the Graph Construction model's `pipeline.extractor.enable_thinking` and `pipeline.edge_generation.enable_thinking` settings.

When BCG mode is requested, the command reuses a healthy Graph Construction server or starts the configured local server in the same way as `bcg`. A BCG request that falls back to raw context is marked `graph_fallback` and excluded from accuracy by default. Use `--allow-graph-fallback` only when that behavior is intentional.

Scoring follows each benchmark's answer protocol:

| Benchmark | Scoring |
| --- | --- |
| BrowseComp | Official-style binary LLM judge |
| GAIA validation | Normalized exact match; test is unscored because references are private |
| HotpotQA | Answer exact match and token F1 |
| MMLU-Pro | Exact A–J option accuracy |

BrowseComp uses the Agent model and endpoint as the judge by default. Override it with `--judge-model`, `--judge-base-url`, and `--judge-api-key-env`.

Every task runs in an isolated working directory. Artifacts are resumable and contain:

```text
<output-dir>/
├── run.json
├── summary.json
└── <benchmark>/<mode>/
    ├── tasks/<task-id>.json
    ├── trajectories/<task-id>.jsonl
    └── graph-contexts/<task-id>.jsonl  # BCG mode: exact graph text injected per request
```

Each BCG task record links to its graph-context trace. The JSONL entries include the renderer (`full` or `compact`), graph size, character count, stream position, and exact role-marked text, making it possible to align graph injection with the following Agent action during trajectory audits.

Dataset files and benchmark results are ignored by Git because task artifacts contain plaintext questions, model responses, and reference answers. Do not publish them unless the dataset's license and benchmark policy explicitly allow it.

`summary.json` separates Agent input, output, cache-read, cache-write, reasoning, and total tokens. Judge input/output tokens are reported separately when the judge endpoint returns usage. It also records wall time, tool/search calls, model-reported Agent cost, status counts, and benchmark-specific metrics. Timeout, max-token, Agent error, and judge failure attempts count as incorrect in the primary accuracy; Graph fallbacks and splits without public references are excluded. A separate completed-only accuracy is retained for diagnosis. See every option with:

```bash
bcg benchmark run --help
```

## Benchmarking Results

The following experiments compare the built-in Agent's normal context management (`Default`) with graph-backed context (`BCG`).

### Evaluation setup

- **Agent model:** GPT-5.6-luna with `thinking=low`.
- **Datasets:** Full BrowseComp (1,266 tasks) and full BrowseComp-ZH (289 tasks); both modes evaluate the same tasks.
- **BCG setup:** Compact Graph Context injected into the system prompt, two recent completed turns retained verbatim, GPT-5.6-luna Graph Construction with reasoning disabled, token-efficient construction, and local `all-MiniLM-L6-v2` embeddings.

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Mode</th>
      <th>Evaluated</th>
      <th>Accuracy</th>
      <th>Mean Agent tokens / task</th>
      <th>Mean Graph tokens / task</th>
      <th>Mean total tokens / task</th>
      <th>Token change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>BrowseComp</strong></td>
      <td>Default</td>
      <td>1,266</td>
      <td>33.33%</td>
      <td>35.39K</td>
      <td>—</td>
      <td>35.39K</td>
      <td>Baseline</td>
    </tr>
    <tr>
      <td>BCG</td>
      <td>1,266</td>
      <td><strong>37.12%</strong></td>
      <td>22.55K</td>
      <td>7.10K</td>
      <td><strong>29.64K</strong></td>
      <td><strong>−16.23%</strong></td>
    </tr>
    <tr>
      <td rowspan="2"><strong>BrowseComp-ZH</strong></td>
      <td>Default</td>
      <td>289</td>
      <td>49.48%</td>
      <td>30.84K</td>
      <td>—</td>
      <td>30.84K</td>
      <td>Baseline</td>
    </tr>
    <tr>
      <td>BCG</td>
      <td>289</td>
      <td><strong>59.17%</strong></td>
      <td>21.55K</td>
      <td>6.35K</td>
      <td><strong>27.90K</strong></td>
      <td><strong>−9.53%</strong></td>
    </tr>
  </tbody>
</table>


## Comparison with Existing Memory Solutions

Agent memory systems serve different purposes. Below is a feature-level comparison of BCG against the most widely-used memory and knowledge graph solutions in the LLM agent ecosystem.

| | Mem0 | Zep | Letta (MemGPT) | LangChain Memory | LlamaIndex | TrustGraph | Semantica | **BCG** |
|---|---|---|---|---|---|---|---|---|
| **Belief-native extraction** |⚡ | ⚡|❌ |⚡ |⚡ |⚡ |⚡ |✅ |
| **Deterministic confidence** | ❌| ❌|❌ |❌ |❌ |❌ |⚡ |✅ |
| **Evidence provenance** | ⚡|✅ | ⚡| ⚡| ⚡| ✅|✅ | ✅|
| **Temporal lifecycle** | ⚡| ✅| ⚡| ⚡| ❌|⚡| ✅| ⚡|
| **Relation linking** |⚡ | ✅| ❌|❌ | ⚡| ✅| ✅| ✅|
| **Local-first artifacts** |⚡ | ❌| ✅| ⚡| ✅|✅ | ✅|✅ |
| **Graph queryability** | ❌| ✅|❌ |❌ | ⚡| ✅|✅ |⚡ |
| **Merge / dedup** |⚡ |✅ | ⚡|⚡ |⚡ |⚡ |✅ | ✅|
| **Conflict detection** |⚡ | ✅|❌ | ⚡| ❌|❌ |✅ | ✅|
| **Run without an independent DB** | ✅|✅ | ✅|✅ | ✅| ❌|✅ |✅ |

> ✅ Full support &nbsp;&nbsp; ⚡ Partial / optional &nbsp;&nbsp; ❌ Not supported

---

<!-- ## Agent Framework Integrations

BCG is a Python library with a minimal dependency footprint. It integrates with agent frameworks through direct API usage, MCP servers, or HTTP bridges. -->


<!-- ### Supported Frameworks -->
<!--
| Framework | Integration Method | Notes |
|---|---|---|
| **Claude Code** | MCP server or Python tool | Expose BCG as an MCP tool for belief extraction and query during agent sessions |
| **LangChain** | Custom `BaseMemory` / Tool | Wrap `BCGMemory` as a LangChain memory backend; register belief extraction as a tool |
| **LlamaIndex** | Custom `BaseMemory` / Tool | Integrate via LlamaIndex's memory abstraction or as an ingestion pipeline tool |
| **OpenAI Agents SDK** | Python function tool | Register `BCGMemory.observe()` and `BCGMemory.search()` as callable tools |
| **CrewAI** | Python tool | Add BCG as a crew tool for belief tracking across multi-agent workflows |
| **AutoGen** | Python tool / agent | Use BCG as a shared memory backend for AutoGen agent groups |
| **Agno** | Python tool | Integrate as a tool or memory backend in Agno agent pipelines |
| **Any MCP-compatible agent** | MCP server | BCG's HTTP API can be fronted by an MCP server exposing belief extraction and query tools |
| **Any Python agent loop** | `BCGRunner` / `BCGMemory` API | Direct import and use in custom agent loops — see the Quick Start examples above |
-->

<!-- ### MCP Server -->
<!--
A built-in MCP server is on the roadmap, exposing:

| Tool | Purpose |
|---|---|
| `bcg_observe_turn` | Feed a conversation turn into the belief graph |
| `bcg_search_beliefs` | Search beliefs by text, confidence range, or entity |
| `bcg_query_graph` | Traverse the belief graph by relation type or temporal range |
| `bcg_get_context` | Assemble belief-aware context for a given task |
| `bcg_finalize_run` | Finalize the current run and return the complete graph |

Until the native MCP server ships, BCG can be used with any MCP-compatible agent via its HTTP API (see `online_server.py`) or direct Python integration.

--- -->

## Configuration

### Environment Variables

Normal users configure these values through `bcg setup`; they are persisted in `~/.bcg/config.json` and `~/.bcg/.env`. `.env.example` is a development and automation reference:

```bash
cp .env.example .env
```

#### LLM API

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for OpenAI-compatible endpoints |
| `OPENAI_BASE_URL` | Base URL for the API endpoint |
| `OPENAI_MODEL` | Model name to use for generation |
| `OPENAI_TIMEOUT` (Optional) | Request timeout in seconds |
| `OPENAI_MAX_RETRIES`  (Optional)| Maximum retry attempts on failure |

#### Embeddings

Used by the semantic split, extract and merge passes:

| Variable | Description |
|---|---|
| `EMBEDDING_PROVIDER` | Embedding service provider |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_BASE_URL` | Base URL for the embedding endpoint |
| `EMBEDDING_API_KEY` | API key for the embedding service |

### Pipeline Configuration

Pipeline behavior — semantic splitting, merge strategy, context budgets, run IDs, and output roots — is configured through Python constructor arguments on `BCGRunner` and `BCGMemory`. There are no hidden environment defaults for pipeline parameters.

---



## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

MIT — see the [LICENSE](LICENSE) file for details.

---
