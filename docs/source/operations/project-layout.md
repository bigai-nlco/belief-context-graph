---
title: "Project Layout"
description: "Where the SDK, backends, Agent, viewer, benchmarks, and deployment files live."
icon: "folder-tree"
---

```text
belief-context-graph/
├── bcg/
│   ├── graph.py               # public graph schema and container
│   ├── memory.py              # application memory facade
│   ├── runner.py              # construction lifecycle
│   ├── llm.py                 # model and embedding clients
│   ├── cli.py                 # top-level command
│   ├── run.py                 # batch driver
│   ├── online_server.py       # HTTP server
│   ├── online_driver.py       # NDJSON replay
│   ├── visualize_beliefs_graph.py
│   ├── benchmark/
│   └── construct/             # core graph-construction engines
│       ├── unified/         # API-driven implementation
│       └── hybrid/             # modular local/small-model implementation
├── agent-cli/                 # bundled terminal Agent runtime
├── dashboard/
│   ├── bcg_viewer/            # live stream viewer and server
│   └── src/                   # dashboard application
├── deploy/
├── scripts/
├── tests/
├── docs/                      # this documentation site
├── pyproject.toml
└── Makefile
```

## Public API boundary

The root package exports:

```python
from bcg import BCG, BCGMemory, BCGRunner
```

The stable public entry points are `BCG`, `BCGMemory`, and `BCGRunner`. The core graph-construction logic lives in the sibling packages `bcg.construct.unified` and `bcg.construct.hybrid`; `BCGRunner` and the HTTP/CLI layers select and orchestrate one backend.

## Documentation source

- `docs/source/**/*.md` — editable Markdown pages
- `docs/assets/styles.css` — offline theme and responsive layout
- `docs/assets/pages.js` — compiled page content and navigation data
- `docs/assets/app.js` — client-side navigation, search, and interactions
- `docs/index.html` and `docs/serve.py` — offline entry points
- `docs/check_offline_docs.py` — structural and content validation
- `.github/workflows/docs.yml` — automated offline-bundle validation
