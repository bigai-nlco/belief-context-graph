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

## Feature matrix (old `bcg_viewer/` vs dashboard)

| Feature | Old Viewer (`bcg_viewer/`) | Dashboard | Conclusion |
|---|---|---|---|
| Live graph from construct server | 3 guessed URLs, none exist | contract endpoint `/graph` | migrate → dashboard |
| Artifact / JSONL replay | `build_stream_manifest.py` + static viewer | `loadArtifactReplay` + future UI picker | migrate → dashboard |
| Directory import | Python `serve_viewer.py` | not yet wired (devtool) | keep as devtool until wired |
| Run control | not present | not present | out of scope (devtool service only) |
| Timing / metrics | stream manifest tables | metrics header | migrate → dashboard |
| Subgraph | not present | not present | out of scope |
| Memgraph/TongGraph connection | bolt + lab links | env-driven (devtool) | keep as devtool |

Old Viewer status: **deprecated**. It stays as a read-only reference until
the dashboard reaches feature parity (step 16 removal window); no further
feature work lands in `bcg_viewer/`.

## Scripts

```bash
npm run dev       # dev server (proxies /graph and /health to VITE_BCG_API_URL)
npm test          # vitest (normalizer, layout, data sources)
npm run build     # tsc + vite build
```
