# BCG Dashboard

Vite-based graph dashboard. Three explicit data sources (step 12), all
normalized through `src/normalize.ts`:

| Source | Adapter | Input |
|---|---|---|
| Live BCG server | `loadLiveGraph` | `GET /graph?problem_id=` per `contracts/http.schema.json`; active session resolved via `/health` |
| Artifact replay | `loadArtifactReplay` | persisted memory document (`schema: bcg.memory.v2`) or raw graph JSON |
| Bundled sample | `sampleMemory` | static demo data |

Live-source failures surface as errors (status bar + console); the sample
is only used when explicitly chosen, never as a masked fallback for API
errors.

## Environment

See `.env.example` (`VITE_BCG_API_URL`, `VITE_BCG_PROBLEM_ID`,
`VITE_MEMGRAPH_URI`, `VITE_MEMGRAPH_LAB_URL`).

## Feature matrix

| Feature | Dashboard | Conclusion |
|---|---|---|
| Live graph from construct server | contract endpoint `/graph` | implemented |
| Artifact / JSONL replay | `loadArtifactReplay` | implemented (UI picker future) |
| Directory import | not yet wired | devtool, out of scope until wired |
| Run control | not present | out of scope (devtool service only) |
| Timing / metrics | metrics header | implemented |
| Subgraph | not present | out of scope |
| Memgraph/TongGraph connection | env-driven | devtool |

The legacy `bcg_viewer/` was removed in step 16 after reaching parity for the
features above.

## Scripts

```bash
npm run dev       # dev server (proxies /graph and /health to VITE_BCG_API_URL)
npm test          # vitest (normalizer, layout, data sources)
npm run build     # tsc + vite build
```
