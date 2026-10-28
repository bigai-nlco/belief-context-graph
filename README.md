# BCG Deep Research Agent

A Deep Research Agent with Belief Context Graph (BCG) memory. This branch provides a single-agent ReAct workflow and a multi-agent research team. Both preserve user inputs and recent complete steps, move older steps into a belief graph, and inject selected beliefs and relations into the system context.

This guide is for the `feat/bcg-deep-research-agent` branch. The executable is still named **`frontier-agent`**. The **BCG Graph server runs separately**; starting the Agent does not start the server or configure its model.

## 1. Install the Agent

Requirements: Git, Python 3.12, and `uv`. You also need a tool-capable model endpoint, a compatible BCG Graph server, and search credentials for web research. The native terminal workflow does not require Node.js, Docker, or a local Agent model.

```bash
git clone --branch feat/bcg-deep-research-agent --single-branch \
  https://github.com/bigai-nlco/belief-context-graph.git \
  bcg-deep-research-agent
cd bcg-deep-research-agent

uv sync --python 3.12 --extra document-readers
cp .env.example .env
chmod 600 .env
```

If you already have this checkout and a configured `.env`, keep that file instead of copying over it.

## 2. Configure the Agent and search tools

Edit `.env` in the **Agent checkout**:

```dotenv
# Agent model: used by ReAct, or by the coordinator and specialists in Agent Team.
OPENAI_API_KEY=your-agent-api-key
OPENAI_BASE_URL=https://your-agent-endpoint.example/v1
OPENAI_MODEL=your-agent-model-id

# Web search.
WEB_SEARCH_PROVIDER=serper
SERPER_API_KEY=your-serper-api-key
SERPER_BASE_URL=https://google.serper.dev

# Optional Jina Reader credentials for fetching web pages.
JINA_API_KEY=
JINA_BASE_URL=https://r.jina.ai

# Connection to the separately running BCG Graph server.
BCG_GRAPH_URL=http://127.0.0.1:8850
BCG_RECENT_TURNS=2
BCG_GRAPH_CHARS=6600
BCG_TEAM_CROSS_GRAPH_CHARS=2000
BCG_GRAPH_TIMEOUT=120
BCG_GRAPH_API_KEY=
```

Use an API base URL, not a full `/chat/completions` URL. The model ID must match the model exposed by that endpoint. For a Serper-compatible search service, change both `SERPER_BASE_URL` and `SERPER_API_KEY`.

The Agent and Graph model can use different providers, models, and keys. **`BCG_GRAPH_API_KEY` is only an optional bearer token for a protected Graph HTTP endpoint**, not the Graph model's key. Leave it empty for the standard local server. This Agent does not automatically load model credentials from `~/.bcg/.env`.

Run the Agent commands below from this checkout so its `.env` is loaded before switching to `--cwd`. Exported environment variables take precedence over `.env`; an old exported key, model, or URL can therefore override your edits. See [`.env.example`](.env.example) for optional page-summary and document-reading endpoints.

### Model-specific settings

The terminal uses these workflow profiles:

| Workflow | Profile |
|---|---|
| Single Agent | [`workflows/stateful_react_agent/profiles/tui.yaml`](workflows/stateful_react_agent/profiles/tui.yaml) |
| Agent Team | [`workflows/agent_team/profiles/tui.yaml`](workflows/agent_team/profiles/tui.yaml) |

The bundled profiles include Qwen-style `top_k` and `chat_template_kwargs` parameters in `llm.extra_body`, and use `agent.thinking_format: tag`. When using another model/provider, adjust these profiles to the endpoint's supported sampling and reasoning format. Remove unsupported fields rather than assuming every OpenAI-compatible endpoint accepts them. Model reasoning is configured in the profile; changing `.env` does not automatically select a reasoning effort.

## 3. Start the BCG Graph server

If you already have a compatible server, reuse it and set `BCG_GRAPH_URL` above to its address. The integration requires `POST /turns`, `GET /graph`, and `POST /context-selection` with the `focused` strategy, plus snapshots containing `n_turns_ingested`.

Otherwise, install the Graph service from the BCG repository's **`main` branch in a separate directory**. Do not switch this Agent checkout to `main`: the two branches contain different applications.

In another terminal, from the parent directory of your Agent checkout:

```bash
git clone --branch main --single-branch \
  https://github.com/bigai-nlco/belief-context-graph.git \
  bcg-graph-server
cd bcg-graph-server
uv sync --python 3.12
```

Create a `.env` in the **Graph server checkout** with the Graph model's key:

```dotenv
GRAPH_API_KEY=your-graph-model-api-key
```

Create `bcg.yaml` in that same directory. Replace the model and endpoint placeholders with your Graph model's settings:

```yaml
schema_version: 1
backend: unified
model_key: graph-model
embedding_key: embedding

models:
  graph-model:
    model: your-graph-model-id
    base_url: https://your-graph-endpoint.example/v1
    api_key_env: GRAPH_API_KEY
    reasoning_effort: none
    max_tokens: 8192
  embedding:
    provider: local
    model: sentence-transformers/all-MiniLM-L6-v2
    device: cpu
```

This example uses the `unified` backend with a hosted Graph model and local CPU embeddings. No local vLLM server is needed for this configuration. The embedding weights may download on first use; use a local model path if the server is offline. Adjust the Graph model's reasoning and output-token settings to what its endpoint supports. The full configuration reference is in the Graph checkout's `bcg/config/config.example.yaml`.

Start the server and leave this terminal running:

```bash
chmod 600 .env
uv run --env-file .env bcg construct server unified \
  --config bcg.yaml \
  --host 127.0.0.1 \
  --port 8850 \
  --output-dir ./graphs
```

From another terminal, check that it is reachable:

```bash
curl --fail --silent --show-error http://127.0.0.1:8850/health
```

The response should contain `"status": "ok"`. This checks service reachability, not whether the Graph model's credentials or first embedding download will succeed. If the server is on another machine, use its reachable address in `BCG_GRAPH_URL`. Only send conversation and tool contents to a server you trust.

## 4. Launch the research Agent

Return to the **Agent checkout** and create a working directory:

```bash
mkdir -p research-workspace
```

### Single Agent

```bash
uv run frontier-agent --native --mode react --cwd ./research-workspace
```

Enter a research question in the terminal interface. One Agent searches, reads sources, checks evidence, and produces the answer or report.

### Agent Team

```bash
uv run frontier-agent --native --mode agent_team --cwd ./research-workspace
```

A coordinator breaks the question into assignments, dispatches specialists, collects their findings, and synthesizes the final report. The coordinator and each dispatched job have independent graphs, connected through delegation and report references. They do not share one global graph.

You can also pass the initial task directly:

```bash
uv run frontier-agent --native --mode agent_team --cwd ./research-workspace \
  "Compare current approaches to long-term Agent memory. Check primary sources, discuss trade-offs, and write a report with citations."
```

Both workflows already use BCG. There is no extra `--bcg` switch. Agent Team can make several model calls concurrently and usually needs a larger budget than a single Agent.

`--native` runs tools under your current operating-system account with the terminal's approval checks; it is not an OS-level sandbox. Keep the default approval prompts. For stronger isolation, see [sandboxing](docs/framework.md#sandboxing); an isolated runtime must also be able to reach the Graph server.

### One-shot runs, files, and session recovery

```bash
# Run one task, print the answer, and exit.
uv run frontier-agent --native --mode react --cwd ./research-workspace \
  -p "Research a topic of your choice and summarize the evidence with citations."

# Use line mode instead of the full-screen interface.
uv run frontier-agent --native --mode react --cwd ./research-workspace --no-tui

# List saved sessions in this workspace.
uv run frontier-agent --native --cwd ./research-workspace --resume

# Resume a specific session; replace SESSION_ID with an actual ID.
uv run frontier-agent --native --cwd ./research-workspace --resume SESSION_ID
```

Use `--input /absolute/path/to/document.pdf` to attach a read-only document, and repeat `--input` for additional files. In the interface, `/config` shows redacted model configuration, `/log` shows the trace location, and `/exit` closes the session. `--max-turns` bounds the main Agent's turns; Agent Team specialist budgets are configured separately in its profile.

## BCG settings and records

| Setting | Default | Meaning |
|---|---:|---|
| `BCG_RECENT_TURNS` | `2` | Complete recent assistant/tool steps kept verbatim, in addition to user inputs. Parallel tool results remain in the same step. |
| `BCG_GRAPH_CHARS` | `6600` | Maximum rendered Graph content in system memory, excluding fixed guidance and original system instructions. |
| `BCG_TEAM_CROSS_GRAPH_CHARS` | `2000` | Part of the Team's total Graph budget reserved for linked remote beliefs, not an additional budget. |
| `BCG_GRAPH_TIMEOUT` | `120` | Timeout in seconds for Graph HTTP requests. |

Older steps leave the model context only after the Graph server acknowledges them. If Graph writing or selection fails, the Agent preserves the original context and records a fallback. Seeing an answer alone does not prove Graph memory was used; inspect the BCG request records when checking an integration.

Interactive runs save records under the workspace selected by `--cwd`:

```text
research-workspace/.apodex/runs/<session-id>/
  engine.log
  outputs/                       generated deliverables
  bcg/                           single-agent memory records
    state.json
    requests.jsonl
  bcg_team/                      multi-agent memory records
    index.json
    g_<id>/state.json
    g_<id>/requests.jsonl
```

Only the active workflow's memory directory is created. The run also retains session checkpoints and full trajectories. `requests.jsonl` records selected beliefs/relations, injected memory, retained transcript indices, Graph usage, and fallback reasons. Full trajectories are not the same as the projected requests sent to the model.

Server-side construction records are separate: with the command above, they are saved under `bcg-graph-server/graphs/<problem_id>/`. Use the problem ID in the Agent's memory state to find the corresponding Graph. Conversation, tool results, and local files can contain private information; do not publish these records or `.env` files.

## Annotate case studies

Case Study Studio provides a shared run dashboard and annotation workbench. Start approved tests, view live per-task progress, and open completed results by run number without importing files. Reviewers identify themselves by nickname and save independent annotations on the server.

The shared dashboard starts and displays DeepResearch Bench II only (`drb2`, task IDs such as `2,5,30`). Historical DRBench files and annotations remain stored; the command-line benchmark adapters are unchanged.

```bash
.venv/bin/python tools/case_study_studio/serve.py --shared --host 0.0.0.0 --port 8876
```

Open **http://127.0.0.1:8876** on the host, or `http://<server-LAN-IP>:8876` on the trusted LAN. Sign in with a nickname to start or stop tests, view results, and annotate; no administrator code is required. Existing experiments are cataloged automatically. Each new launch gets an independent `results/studio/RUN-<number>/` directory. Profiles and API keys are server-side only. See the [Case Study Studio guide](tools/case_study_studio/README.md) for prerequisites, identity recovery, and storage details. Do not expose this plain-HTTP service to the public Internet.

The original browser-local mode remains available by starting the server without `--shared`. Local drafts are retained; the shared dashboard has no local-draft import entry. No model, Graph server, or API key is needed for viewing and manually annotating existing evidence. Export JSON workspaces, Markdown case studies, or CSV summaries as before. AI annotation is a separate, explicitly started model call; see the [AI Review guide](tools/ai_review/README.md).

BCG workflows now automatically write `case-study.json.gz` in the task/session artifact directory. The report and subprocess benchmark runners also write `case-studies.json.gz` at the batch output root when the run finishes. Import one batch bundle or several task bundles together; each Case retains its own evidence and annotations. No manual extraction is needed. Graph updates received by the Agent are saved locally alongside injection records, so a remote Graph server does not require a second manual download. Reimporting identical evidence preserves existing annotations.

For existing records, run `python -m workflows._shared.case_study /path/to/task-or-batch`; add `--graph-dir /path/to/graphs` to include older service-side histories. This packages existing files only and makes no model calls.

## Run Deep Research benchmarks

The report runner supports **DeepResearch Bench II** (`drb2`) and the local-file **DRBench L1 variant** (`drbench`) with the same BCG workflows. Run these commands from the Agent checkout after configuring and starting the Graph server:

```bash
uv sync --extra eval --extra sandbox --extra document-readers
uv run python -m benchmarks.public.runner.run_reports prepare drb2 drbench

# Inspect a two-task configuration without calling models.
uv run python -m benchmarks.public.runner.run_reports run drb2 \
  --pipeline agent_team --language zh --limit 2 --seed 42 \
  --workers 1 --out results/drb2-bcg-team-smoke --dry-run
```

Remove `--dry-run` to run it. Use `--pipeline stateful-react-agent` for a single Agent. `--workers` controls concurrent questions, not specialist count; start with `1`. Local benchmark execution requires a usable sandbox such as Linux bubblewrap, separately from the native TUI setup above. See the [report benchmark guide](benchmarks/public/reports/README.md) for data preparation, DRBench commands, tool restrictions, result paths, and resuming.

DRB2 defaults to 20 search queries and 30 URL fetches per task, shared across agents. Use `--search-calls 0 --fetch-calls 0` to disable both count limits, or pass positive values to set larger budgets. Each query or URL in a batched tool call counts separately. This does not disable turn, timeout, source-access, or response-size limits.

These adapters save full reports, diagnostic metrics, and separate Agent/Graph token usage. Research runs leave `official_score` as `null`. For DeepResearch Bench II, grade the saved reports separately using `uv run python -m benchmarks.public.runner.run_reports judge --out <run-output>` with the same dataset arguments as the run. This applies the official rubric protocol without rerunning the Agent; see the [grading configuration and score interpretation](benchmarks/public/reports/README.md#drb2-rubric-judging). DRBench insight judging is not implemented. Diagnostic metrics and AI behavior annotations are not official benchmark scores.

## Troubleshooting

| Symptom | Check |
|---|---|
| `frontier-agent` is not found | Run `uv sync` and use `uv run frontier-agent` from the Agent checkout. Installing the separate `bcg` package does not install this Agent. |
| Model authentication or model-not-found error | Check the Agent checkout's `OPENAI_*` settings, the exact model ID, and any exported variables overriding `.env`. |
| Unsupported model parameter | Update the active workflow profile's `llm` settings; Qwen-style parameters are not accepted by every provider. |
| Graph connection failure or full-context fallback | Check `BCG_GRAPH_URL`, `/health`, Graph server logs, model credentials, and support for `focused` context selection. |
| Search fails | Check `SERPER_API_KEY`, `SERPER_BASE_URL`, and `WEB_SEARCH_PROVIDER` in the Agent environment. |
| A private page cannot be fetched | Check the configured fetch provider and its credentials; research tools do not bypass website authentication. |
| TUI does not open | Run in a normal interactive terminal, or use `--no-tui` for line mode. |
| Benchmark sandbox initialization fails | Install/configure a supported sandbox and check host user-namespace support. Do not treat native TUI mode as benchmark sandbox isolation. |

## Further reading and attribution

- [Single-agent BCG memory](workflows/stateful_react_agent/README.md#bcg-memory)
- [Agent Team graph ownership and cross-graph links](workflows/agent_team/README.md#bcg-multi-agent-memory)
- [Deep Research report benchmarks](benchmarks/public/reports/README.md)
- [Terminal user guide](docs/tui-user-guide.md)

This branch adapts the [FrontierAgent](https://github.com/ApodexAI/FrontierAgent) runtime for BCG-backed research. Its retained runtime code is licensed under [Apache 2.0](LICENSE). Migrated report benchmark adapters retain their [upstream license](benchmarks/public/reports/UPSTREAM_LICENSE); the separate BCG Graph service retains its own license.
