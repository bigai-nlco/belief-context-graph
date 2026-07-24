
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

### Install

This project uses `uv` for dependency management. Python 3.11+ required.

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph
uv sync
```

This creates a repository-local `.venv`. You may alternatively install the
single `bcg` executable in an isolated user tool environment:

```bash
uv tool install .
bcg --version
```

### Configuration

Keep credentials in the ignored root `.env`, and model routing in the
non-secret JSON config:

```bash
cp .env.example .env
cp bcg/model_config.example.json bcg/model_config.json
```

Every `api_key_env` value in `model_config.json` names a variable from
`.env`; do not put API keys directly in JSON.

### Construct Commands

The constructor provides two backends: `api_based` uses one large
OpenAI-compatible model, while `light` uses local embeddings, spaCy, and
smaller extractor/relation models. Put the backend after `run`, `server`,
or `replay`. Omitting it remains compatible and defaults to `api_based`.

```bash
bcg construct run api_based --input data.json --config bcg/model_config.json
bcg construct server light --config bcg/model_config.json --host 127.0.0.1 --port 8848
bcg construct replay api_based --input stream.jsonl --config bcg/model_config.json
```

See [bcg/README.md](bcg/README.md) for backend configuration, input formats,
HTTP endpoints, and Python APIs.

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
| **Belief-native extraction** | | | | | | | | |
| **Deterministic confidence** | | | | | | | | |
| **Evidence provenance** | | | | | | | | |
| **Temporal lifecycle** | | | | | | | | |
| **Relation linking** | | | | | | | | |
| **Local-first artifacts** | | | | | | | | |
| **Graph queryability** | | | | | | | | |
| **Merge / dedup** | | | | | | | | |
| **Conflict detection** | | | | | | | | |
| **Multi-agent shared context** | | | | | | | | |
| **External DB required** | | | | | | | |

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

Copy `.env.example` when you need local API credentials:

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
