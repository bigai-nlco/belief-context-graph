
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


## **Core capabilities:**

- **Belief Extraction:** Segment trajectories, extract structured beliefs that counts for agent reasoning and link them into a connected graph
- **Deterministic Confidence:** Auditable posterior confidence computed from `initial_confidence`, `evidence_confidence`, and relation-derived `factor_confidence`; source reliability and stance quality set the prior, while relation weights and activation thresholds propagate support or contradiction deterministically
- **Evidence Provenance:** Every belief carries exact-offset source references back to the originating conversation turn
- **Temporal Awareness:** Run-based lifecycle with sessions and timestamps — know when each belief was formed and how it evolved
- **Relation Linking:** Forward and backward relationship edges between beliefs, forming a casual decision graph/trace.

  

<p align="center">
  <a href="assert/benchmark_overview.svg">
    <img src="assert/benchmark_overview.svg" width="100%" alt="Full-dataset Default, BCG, and Summary accuracy and mean token cost for BrowseComp and BrowseComp-ZH, followed by the unchanged Default-versus-BCG token-horizon comparison">
  </a>
</p>

<p align="left"><sub><sub>Agent: gpt-5.6-luna with low-effort thinking · Graph/Summary: gpt-5.6-luna with thinking disabled · Harness: <a href="agent-cli/packages/agent-core/src/harness">repository harness</a> · BCG: compact Graph Context with two recent completed turns</sub></sub></p>


## Live Demo

https://github.com/user-attachments/assets/fa10c247-7f6f-4567-9a1f-648c79b1df44

---

<p align="center">
  <strong><a href="#quick-start">Quick Start</a></strong> &nbsp;·&nbsp;
  <strong><a href="#architecture">Architecture</a></strong> &nbsp;·&nbsp;
  <strong><a href="#an-evidence-of-belief-context-graph-usage">Case Study</a></strong> &nbsp;·&nbsp;
  <strong><a href="#comparison-with-existing-memory-solutions">Comparison</a></strong> &nbsp;·&nbsp;
  <strong><a href="#contributing">Contributing</a></strong>
</p>

---


## Quick Start

Requires Python 3.11–3.13. The optional reference terminal Agent additionally requires Node.js 22.19+.

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph
make install
uv run bcg --version
```

Launch the bundled reference Agent to try BCG immediately — integrating BCG into your own Agent does not require this CLI:

```bash
uv run bcg
```

The first run walks you through model credentials, context mode, and Graph Construction backend, and saves your choices under `~/.bcg`. See the [documentation](https://belief-context-graph.docs.buildwithfern.com/) for everything else.


---

## Architecture

BCG is an optional context layer between an Agent and its model. In BCG mode, the initial user input and recent completed turns remain in the raw context, while older completed turns stream into Graph Construction; the resulting belief snapshot is then injected into the system prompt. Both the HTTP service and the Python SDK use the same backend registry, construction pipeline, confidence semantics, and graph artifacts.

<p align="center">
  <a href="assert/architecture.svg">
    <img src="assert/architecture.svg" width="100%" alt="BCG architecture: Agent context management and Python SDK feed unified or hybrid Graph Construction backends, which share an incremental belief graph pipeline and return Graph snapshots to the Agent system prompt">
  </a>
</p>

---

## Benchmark Adapter

The reference Agent can be evaluated head-to-head in Default, BCG, and Summary modes against **BrowseComp** and **BrowseComp (ZH)**. All modes use the same Agent model, prompt, and scorer; only context management changes.

```bash
bcg benchmark run browsecomp browsecomp_zh --modes default,bcg,summary \
    --thinking off \
    --summary-model gpt-4.1-mini \
    --summary-thinking off \
    --max-problems 100 \
    --workers 8 \
    --output-dir results/browsecomp-comparison
```

See [Evaluate with benchmarks](https://belief-context-graph.docs.buildwithfern.com/operate/benchmarking) for dataset setup, scoring, output artifacts, and every `bcg benchmark run` option.

---

## An Evidence of Belief Context Graph Usage

This successful BrowseComp case (`browsecomp-0836`) shows Kimi K3 using belief identities and confidence from BCG instead of repeating an already completed search.

<p align="center">
  <img src="assert/case_study.svg" alt="BrowseComp task, injected graph beliefs, and selected Kimi K3 thinking passages with graph references highlighted in red" width="100%">
</p>

---

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

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

MIT — see the [LICENSE](LICENSE) file for details.

---
