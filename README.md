
<div align="center">

# Belief Context Graph

### A Belief-Native Graph Memory for LLM Agents

**Probabilistic · Temporal · Explainable · Stateful**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/uv-managed-6E4BF9.svg?style=flat-square)](https://docs.astral.sh/uv/)

</div>

---
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


Belief Context Graph (`BCG`) upgrades agent memory from **retrieval memory** to **belief computation memory**. It is a probabilistic, temporal, evidence-grounded memory substrate that helps agents continuously maintain: what to believe, at what confidence, from which evidence, and whether uncertainty should block action. The result is agent memory you can query, audit, and trust.

## **Core capabilities:**

- **Belief Extraction:** Segment trajectories, extract structured beliefs that counts for agent reasoning and link them into a connected graph
- **Deterministic Confidence:** Four-dimensional confidence assessment (source reliability, evidence directness, claim specificity, linguistic certainty) — LLMs inform the dimensions, but the final score is deterministic and auditable
- **Evidence Provenance:** Every belief carries exact-offset source references back to the originating conversation turn
- **Temporal Awareness:** Run-based lifecycle with sessions and timestamps — know when each belief was formed and how it evolved
- **Relation Linking:** Forward and backward relationship edges between beliefs, forming a casual decision graph/trace.

---

**[Live Demo](#live-demo)** &nbsp;·&nbsp;**[Quick Start](#quick-start)** &nbsp;·&nbsp; **[Architecture](#architecture)** &nbsp;·&nbsp; **[Core Concepts](#core-concepts)** &nbsp;·&nbsp; **[Benchmarking](#benchmarking-results)** &nbsp;·&nbsp; **[Comparison](#comparison-with-existing-memory-solutions)** &nbsp;·&nbsp; **[Integrations](#agent-framework-integrations)** &nbsp;·&nbsp; **[Configuration](#configuration)** &nbsp;·&nbsp;  **[Contributing](#contributing)**

---
## Live Demo

## Quick Start

BCG uses two isolated runtimes: Python/`uv` for the SDK and Graph Construction,
and Node.js 22.19+ for the interactive terminal Agent. RLLM is not used.

### 1. Install BCG

Choose one of the following installation methods.

#### Option A: install globally with curl

Use this method when you want the `bcg` command without keeping a source
checkout:

```bash
curl -LsSf https://raw.githubusercontent.com/bigai-nlco/belief-context-graph/main/install.sh | sh
bcg --version
```

The installer requires `curl`, `tar`, npm, and Node.js 22.19 or newer. It
installs `uv` when necessary, downloads the repository into a temporary
directory, installs the Python and Node runtimes, and then removes the temporary
source.

#### Option B: clone and run from source

Use this method for development or when you want to inspect and modify the
code:

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph

uv sync
npm --prefix agent-cli ci
npm --prefix agent-cli run build

uv run bcg --version
```

Run the source checkout with:

```bash
uv run bcg
```

Optionally expose the current checkout as the global `bcg` command:

```bash
uv tool install --editable .
npm install -g ./agent-cli
bcg --version
```

The editable Python install follows changes in the checkout. The Node package
provides the internal `bcg-agent` executable launched by `bcg`; users normally
do not invoke it directly.

### 2. Start the BCG Agent

For a curl or global installation:

```bash
bcg
```

From a source checkout:

```bash
uv run bcg
```

On the first launch, the setup guide asks for:

1. Agent authentication: an OpenAI-compatible API key and base URL, or the
   interactive `/login` flow.
2. The Agent model.
3. The default context mode: **BCG** or **Default**.
4. Whether BCG should manage a local Graph Construction server or connect to an
   existing one.
5. For a managed server, the Graph backend: **api_based** or **light**.

The setup is saved under `~/.bcg` and works from every directory:

```text
~/.bcg/config.json        # Agent, context, and Graph runtime choices
~/.bcg/.env               # API keys and credentials; mode 0600
~/.bcg/model_config.json  # Graph model routing; no inline secrets
~/.bcg/agent/             # Agent settings, authentication, and sessions
```

Run `bcg setup` at any time to change these settings.

When a managed Graph backend is selected, `bcg` automatically:

1. Checks `http://127.0.0.1:8848/health`.
2. Reuses a healthy Graph Construction server or starts one with the selected
   backend and `~/.bcg/model_config.json`.
3. Writes its log to `~/.bcg/logs/graph-server.log` and graph artifacts to
   `~/.bcg/graphs/`.
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

Context mode is fixed after the first user message. Use `/new` to start a
session with another mode:

- **BCG** permanently retains the initial user input and keeps the latest two
  completed turns as raw messages. On the first request, the system prompt and
  initial user input seed the Graph. Messages leaving the two-turn raw window
  are then added incrementally. The current Markdown Graph is wrapped in
  `<belief_graph format="markdown">...</belief_graph>` and appended to the
  system prompt. Traditional compaction is disabled.
- **Default** uses the full normal Agent conversation with automatic
  compaction. Graph context is not injected.

Use `/mode` for the selector, or `/mode bcg` and `/mode default` directly.
`BCG_RECENT_TURNS` can override the default raw window of `2`.

### 3. Configure and Start Graph Construction

BCG supports two construction backends. In normal Agent use, you choose one
during `bcg setup` and `bcg` starts or reuses the Graph Construction HTTP server
automatically. The commands below are also provided for independent deployment
and debugging.

#### Option A: `api_based`

`api_based` uses one OpenAI-compatible model for graph node and relation
generation. During setup, it can reuse the Agent model endpoint and API key or
use a separate endpoint:

```bash
bcg setup
# Graph server: Start and manage a local Graph server automatically
# Graph backend: API based

bcg
```

No vLLM process is required when the configured API endpoint is already
available.

The equivalent manual Graph server command is:

```bash
bcg construct server api_based \
  --config ~/.bcg/model_config.json \
  --model-key graph-model \
  --embedding-key embedding \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir ~/.bcg/graphs
```

If that server is already healthy, a later `bcg` invocation reuses it.

#### Option B: `light`

`light` uses:

- a small generative model served through an OpenAI-compatible vLLM endpoint;
- a local sentence-transformers embedding model;
- a local/Hugging Face stance classifier;
- spaCy for local language processing and entity extraction.

The BCG Python installation contains the Graph-side dependencies, but it does
not install or manage the separate vLLM GPU service.

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
Graph backend: Light
vLLM base URL: http://127.0.0.1:8001/v1
Model served by vLLM: <MODEL_NAME>
vLLM API key: EMPTY
```

`<MODEL_NAME>` must match the value passed to `--served-model-name`. After vLLM
is ready, launch the Agent:

```bash
bcg
```

BCG starts the light Graph Construction server at `127.0.0.1:8848`; it does not
start the vLLM process. To start Graph Construction manually instead:

```bash
bcg construct server light \
  --config ~/.bcg/model_config.json \
  --model-key graph-model \
  --embedding-key embedding \
  --host 127.0.0.1 \
  --port 8848 \
  --output-dir ~/.bcg/graphs
```

For source development, `scripts/start_vllm.sh` is also available, but it reads
`VLLM_*` values from the checkout's root `.env` or explicit command-line
arguments; it does not read `~/.bcg/model_config.json`.

#### Connect to an existing Graph server

If Graph Construction is already hosted elsewhere, run `bcg setup`, choose
**Connect to an existing Graph server**, and enter its URL. In this mode BCG
checks and reuses that endpoint but does not start or manage it.

### 4. Run Graph Construction Without the Agent

The same two backends can process saved trajectories:

```bash
bcg construct run api_based \
  --input data.json \
  --config ~/.bcg/model_config.json \
  --model-key graph-model

bcg construct replay light \
  --input stream.jsonl \
  --config ~/.bcg/model_config.json \
  --model-key graph-model
```

See [bcg/README.md](bcg/README.md) for input formats, HTTP endpoints, output
artifacts, and Python APIs.

## Python SDK

### Minimal Example

```python
from bcg import BCG, BCGMemory

graph = BCG()
memory = BCGMemory(graph=graph)

observation = memory.observe(
    source_type="message",
    content="Acme is threatening to churn after repeated outages.",
)
```

### Multi-Turn Belief Construction

For full trajectory processing, use `BCGRunner` with an LLM client:

```python
from bcg import BCG, BCGMemory, BCGRunner
from bcg.llm import LLMClient

memory = BCGMemory(graph=BCG())
runner = BCGRunner(memory=memory, llm=LLMClient())

# Process a conversation and build the belief graph
result = await runner.observe_trajectory(
    [{"role": "user", "content": "Acme is threatening to churn."}],
)
print(result.output_paths.memory)
```

### Run Lifecycle (Step by Step)

For fine-grained control over sessions and turns:

```python
memory = BCGMemory(graph=BCG())
runner = BCGRunner(memory=memory, llm=LLMClient())

runner.begin_belief_run(run_id="demo")
runner.start_session("session-1", "2026-06-12")
await runner.observe_turn("user", "Alice likes green tea.")
await runner.observe_turn("assistant", "Noted. I'll remember that preference.")
await runner.end_session()
result = await runner.finalize()
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
    beliefs.json            # extracted beliefs
    forward_relations.json  # forward relation edges
    merges.json             # duplicate belief merge decisions
```

---

## Architecture

The belief graph construction gpipeline transforms raw conversation trajectories into a structured, queryable belief graph in five stages:

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Trajectory  │───▶│  Segment     │───▶│  Extract     │───▶│  Extract     │───▶│  Merge &     │
│  Input       │    │  & Split     │    │  Beliefs     │    │  Relations   │    │  Assemble    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
  Raw messages      Semantic split      Belief extraction      Forward/backward
  or turns          into segments       & confidence scoring   belief linking    Graph assembly
```

| Stage | Module | Description |
|---|---|---|
| **Segmentation** | `BCGMemory.segment` | Splits multi-turn trajectories into coherent segments; optional semantic clustering |
| **Extraction** | `BCGMemory.belief_extraction` | Extracts structured beliefs for reasoning from each segment |
| **Confidence** | `BCGMemory.belief_confidence` | Computes deterministic confidence across four dimensions — no LLM-generated confidence numbers |
| **Linking** | `BCGMemory.link_extraction` | Validates and creates forward/backward relation edges between beliefs |
| **Merge** | `BCGMemory.graph_merge` | Optional deduplication pass that merges semantically equivalent beliefs |

The pipeline is orchestrated by `BCGMemory` that are configurable via Python constructor arguments — context budgets, merge strategy, run IDs, and output paths are all explicit, not hidden behind environment variables.

---

## Core Concepts

### Belief

A belief is the fundamental unit of knowledge in BCG. Each belief carries:

- **Typed payload** — structured data describing what the agent believes (facts, tool call response, reasoning steps, etc.)
- **Confidence score** — deterministic average of four assessment dimensions
- **Evidence provenance** — exact character-offset references back to the source turn
- **Temporal metadata** — session ID, turn index, and timestamp of formation

### Confidence Assessment

Belief confidence is **deterministic and auditable**. The stored value is the average of four independently assessed dimensions:

| Dimension | What It Measures |
|---|---|
| `source_reliability` | Trustworthiness of the information source |
| `evidence_directness` | How directly the evidence supports the claim |
| `claim_specificity` | Granularity and concreteness of the belief |
| `linguistic_certainty` | Certainty signals in the language used |

LLMs contribute to assessing each dimension, but the final confidence number is computed mathematically — no model-generated confidence scores enter the graph.

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

## Benchmarking Results

> **Methodology.** Each dataset was run with Claude Opus 4.8 as the extraction LLM, sentence-mode evidence, and embedding-based merge (threshold 0.86). Confidence dimensions were assessed per-belief by the LLM; final scores are deterministic averages of the four dimensions. All runs were local, single-process, with a 100k-char context budget.


| Benchmark | Task Perf. (w/o BCG) | Task Perf. (w/ BCG) | Time Cost  | Token Effiency | API Cost  |
|---|---|---|---|---|---|
| **GAIA** | | | | | |
| **WebArena**| | | | | |
| **SWE-bench** | | | | | |
| **HotpotQA**| | | | | |
| **ALFWorld** | | | | | |
| **Mind2Web** | | | | | |
| **AgentBench** | | | | | |


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

## Agent Framework Integrations

BCG is a Python library with a minimal dependency footprint. It integrates with agent frameworks through direct API usage, MCP servers, or HTTP bridges.


### Supported Frameworks
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

### MCP Server
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

Normal users configure these values through `bcg setup`; they are persisted in
`~/.bcg/config.json` and `~/.bcg/.env`. `.env.example` is a development and
automation reference:

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
