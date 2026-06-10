# AgentHarness Contributor Notes

## Project Shape

- `agent_harness/` contains the reusable runtime kernel, scheduling layer,
  model adapters, observers, and the FastAPI surface in `agent_harness/api/`.
- `workflows/react_base/` is the default single-agent ReAct workflow. It uses
  `ReactStepTracker` to collect tool-call trace steps and returns
  `final_answer`, `final_content`, and `react_steps`.
- `plugins/tools/` contains the built-in tools exposed to the ReAct solver:
  `web_search`, `web_fetch`, and `run_python_code`.
- `benchmarks/` contains benchmark runners and should not be required by the
  web API.
- `ui/` is the dependency-free local frontend. It stores chat sessions in browser
  `localStorage`; the backend is stateless with respect to frontend sessions.

## Backend API

Run the API from the repository root:

```bash
uv run uvicorn agent_harness.api.server:app --reload --host 127.0.0.1 --port 8000
```

Useful endpoints:

- `GET /health` confirms the harness runtime started.
- `GET /api/config` returns public frontend config derived from `.env`, including
  the currently supported model name.
- `POST /api/inquiries` accepts `{ "query": "...", "session_id": "..." }` and
  returns the final answer plus reader-safe ReAct trace steps.

The API intentionally serializes harness runs because the runtime service
registry is process-global. Do not remove that lock unless the registry and
checkpointing model are made request-scoped.

## Local Development

- Install Python dependencies with `uv sync --python 3.12`.
- Configure `.env` from `.env.example` before running real inquiries.
- Run Python checks with:

```bash
uv run python -m compileall agent_harness workflows plugins benchmarks
```

- Run the frontend from `ui/`:

```bash
python3 -m http.server 5173
```

## Trace Safety

The frontend and API should present trace summaries, tool names, arguments, and
observations. Do not expose raw hidden model thinking fields as user-facing
reasoning text.
