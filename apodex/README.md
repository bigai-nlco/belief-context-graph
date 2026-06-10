<div align="center">
  <picture>
    <img src="./assets/apodex_logo.png" width="30%" alt="Apodex-1.0">
  </picture>
</div>
<hr>
<div align="center" style="line-height:1">
<a href="https://www.apodex.ai" target="_blank"><img alt="Research" src="https://img.shields.io/badge/🤖%20Online Service-Apodex 1.0-ff6b6b?color=1783ff&logoColor=white"/></a>
<a href="https://www.apodex.com/" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Apodex AI-white?logoColor=white"/></a>
<a href="https://platform.apodex.ai" target="_blank"><img alt="API" src="https://img.shields.io/badge/API-Apodex 1.0-1783ff?color=1783ff&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
<a href="https://huggingface.co/collections/apodex" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Apodex AI-ffc107?color=ffc107&logoColor=white"/></a>
<a href="https://github.com/ApodexAI" target="_blank"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Apodex AI-white?logo=github&logoColor=white"/></a>
<a href="https://discord.gg/TDJA59TCng" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Apodex AI-white?logo=discord&logoColor=white"/></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue?color=blue"/></a>
</div>
<p align="center">
<b>📰<a href="https://www.apodex.com/blog/apodex-1.0">Tech Blog</a></b> |  <b>📄<a href="http://www.apodex.com/pdf/20260608">Tech Report</a></b>
</p>

---

# AgentHarness

**Evaluation harness for [Apodex-1.0](https://huggingface.co/apodex/Apodex) on public deep-research benchmarks.**

AgentHarness is the open-source evaluation harness used to reproduce the public benchmark results for **Apodex-1.0** in a standard **ReAct setup**. Apodex-1.0 is a verification-centric model for deep research developed by the Apodex team. This repository focuses on the public, standard ReAct evaluation setup reported in the paper.

<p align="center">
  <img src="./assets/apodex1.0_bench.png" alt="Apodex-1.0 results across deep-research benchmarks" width="800"/>
</p>

---

## 📊 Performance

Open-source Apodex-1.0 variants on the four-benchmark deep-research suite:

| Model               | BrowseComp | BrowseComp-ZH | HLE-Text | DeepSearchQA |
| ------------------- | ---------- | ------------- | -------- | ------------ |
| Apodex-1.0-mini     | 71.5       | 80.6          | 46.8     | 82.2         |
| Apodex-1.0-4B-SFT   | 48.8       | 63.5          | 32.9     | 69.9         |
| Apodex-1.0-2B-SFT   | 27.9       | 35.0          | 18.2     | 49.9         |
| Apodex-1.0-0.8B-SFT | 13.9       | 10.7          | 11.2     | 25.8         |

---

## ⚡ Quick Start on the harness

### 1. Install dependencies

```bash
uv sync --python 3.12
```

### 2. Serve the model (SGLang)

```bash
python3 -m sglang.launch_server \
  --model-path apodex/Apodex-1.0-35B-A3B \
  --tp 8 \
  --host 0.0.0.0 \
  --port 1234 \
  --context-length 262144 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
```

For smaller variants, other serving options, see the [Hugging Face model card](https://huggingface.co/collections/apodex/apodex-1).

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in the required keys in `.env` — `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` point at the agent model (your SGLang endpoint from step 2 or any OpenAI-compatible service); `SERPER_API_KEY` / `JINA_API_KEY` / `E2B_API_KEY` enable web search, web fetch, and the code sandbox respectively.

### 4. Download the benchmark datasets

```bash
wget https://huggingface.co/datasets/apodex/Deep-Research-Benchmarks/resolve/main/deep_research_benchmarks_260607.zip
unzip -P 'apodex*()_2026' deep_research_benchmarks_260607.zip
rm deep_research_benchmarks_260607.zip
```

The single quotes around the password are required — it contains shell-meta characters (`*`, `(`, `)`).

> **HLE is not included.** Its license forbids redistributing the answers. To run `hle_text`, accept the license on [`cais/hle`](https://huggingface.co/datasets/cais/hle) and place the JSONL at `benchmarks/datasets/HLE-text/standardized_data.jsonl`.

### 5. Run a smoke test

```bash
uv run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp \
  --pipeline react_base \
  --profile default \
  --limit 1 \
  --concurrency 1 \
  --out ./tmp/smoke
```

### 6. Run a full benchmark

```bash
uv run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp \
  --pipeline react_base \
  --profile default \
  --runs 5 \
  --concurrency 30 \
  --out ./bc-runs
```

### 7. Check progress and aggregate accuracy

```bash
uv run python -m benchmarks.runner.check_progress ./bc-runs
```

Each question runs in its own subprocess, which makes runs easier to reproduce and debug:

* isolated execution per question
* no asyncio saturation
* individual hangs can be `SIGKILL`'d
* failed samples can be rerun independently

---

## ✅ Supported Benchmarks

BrowseComp, BrowseComp-ZH, xbench-DeepResearch, Humanity's Last Exam (text-only), SuperChem, FrontierScience-Research, FrontierScience-Olympiad, DeepSearchQA, WideSearch

See [`benchmarks/README.md`](benchmarks/README.md) for dataset layout, judge configuration, and how to add a new benchmark.

---

## ⭐ Star History

<a href="https://star-history.com/#ApodexAI/AgentHarness&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ApodexAI/AgentHarness&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ApodexAI/AgentHarness&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ApodexAI/AgentHarness&type=Date" />
  </picture>
</a>

---

## 📚 Citation

```bibtex
@techreport{apodex2026,
  title  = {Apodex-1.0: A Verification-Centric Agent Team for Discoverative Intelligence},
  author = {Apodex Team},
  year   = {2026}
}
```

---

## 📄 License

Apache 2.0 — see [LICENSE](./LICENSE).
