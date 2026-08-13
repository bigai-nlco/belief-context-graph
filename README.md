
<div align="center">

# Belief Context Graph


**Probabilistic · Temporal · Explainable · Stateful**

[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/uv-managed-6E4BF9.svg?style=flat-square)](https://docs.astral.sh/uv/)
[![Documentation](https://img.shields.io/badge/documentation-online-0F766E.svg?style=flat-square&logo=readthedocs&logoColor=white)](https://belief-context-graph.docs.buildwithfern.com/)

</div>

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

<p align="center">
  <img src="assert/benchmark_browsecomp.svg" width="49%" alt="BrowseComp full-dataset dual-axis comparison of accuracy and mean token cost per task">
  <img src="assert/benchmark_browsecomp_zh.svg" width="49%" alt="BrowseComp-ZH full-dataset dual-axis comparison of accuracy and mean token cost per task">
</p>

<p align="right">
  <sub><sub>Evaluation setup: GPT-5.6-luna Agent with <code>thinking=low</code> · BCG uses compact Graph Context in the system prompt, two recent completed turns, GPT-5.6-luna Graph Construction with reasoning disabled, and local <code>all-MiniLM-L6-v2</code> embeddings.</sub></sub>
</p>

---

<p align="center">
  <strong><a href="#live-demo">Live Demo</a></strong> &nbsp;·&nbsp;
  <strong><a href="#core-capabilities">Core Capabilities</a></strong> &nbsp;·&nbsp;
  <strong><a href="#quick-start">Quick Start</a></strong> &nbsp;·&nbsp;
  <strong><a href="#architecture">Architecture</a></strong> &nbsp;·&nbsp;
  <strong><a href="#core-concepts">Core Concepts</a></strong> &nbsp;·&nbsp;
  <strong><a href="#comparison-with-existing-memory-solutions">Comparison</a></strong> &nbsp;·&nbsp;
  <strong><a href="#contributing">Contributing</a></strong>
</p>

---
## Live Demo

<div align="center">

<img
  src="assert/live_demo.gif"
  alt="Belief Context Graph live demo"
  width="900"
/>

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

Core BCG uses Python/`uv` for the SDK and Graph Construction. The optional reference terminal Agent uses Node.js 22.19+.

### 1. Install BCG

Choose one install path.

#### Option A: install globally with curl (release)

```bash
curl -LsSf https://raw.githubusercontent.com/bigai-nlco/belief-context-graph/main/install.sh | sh
bcg --version
```

Requires `curl`, `tar`, npm, and Node.js 22.19+. The installer uses the repository lockfiles and installs `uv` when needed.

#### Option B: clone and run from source (development)

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph
make install
uv run bcg --version
```

Optional: expose the checkout as a global command.

```bash
make install-tool
bcg --version
```

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
