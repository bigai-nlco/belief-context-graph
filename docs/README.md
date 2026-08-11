# BCG Documentation — branded Fern edition

This is the second documentation revision for the refactored
`belief-context-graph` repository.

The design goal is deliberately hybrid:

- **Documentation** keeps the visual language of the original BCG docs.
- **SDK Reference** uses compact, collapsible resource navigation but documents
  the real in-process Python API.
- **HTTP API** uses Fern's generated API Reference experience: endpoint pages,
  request/response schemas, HTTP snippets, and the API Explorer.

Fern is being used as a documentation/reference engine, not as the brand.

## What changed from the first Fern revision

### Visual design

The palette in `fern/styles.css` is ported from the original
`docs/assets/styles.css`:

```text
paper background       #faf7f2
panel                   #ffffff
ink                     #1a1a1a
muted text              #555555
rules                   #e8e2d8
brick accent            #b8442f
brick soft              #fbf0ec
teal selection          #0f766e
teal selection soft     #eef8f6
user semantic blue      #1e5a9c
assistant semantic pink #8b2c5b
tool semantic green     #2e6e3a
decision purple         #7c3aed
warning orange          #b3500e
```

The normal pages use Georgia-style headings, warm code/table surfaces, brick
links and heading rules, and teal active navigation—matching the original docs
rather than a monochrome Fern theme.

### Navigation

Only three top tabs remain:

```text
Documentation
SDK Reference
HTTP API
```

GitHub appears once as a normal upper-right link.

Sidebar sections use Fern's collapsible navigation:

```yaml
collapsed: true
```

or:

```yaml
collapsed: open-by-default
```

so users see bold resource headings first and expand only what they need.

### SDK Reference

The first revision had one sidebar entry for nearly every Python method.
This revision groups related methods into user-facing tasks.

Example:

```text
Graph
  Overview
  Create and update knowledge
  Read and serialize graph state
  Connections and maintenance
  Low-level graph operations
```

The same pattern is used for Memory, Runner, Model client, and Configuration
and types.

### HTTP API

The HTTP API stays OpenAPI-driven and paginated. Each endpoint gets its own
Fern reference page with the two-column request/response layout.

The OpenAPI source is:

```text
fern/openapi.yml
```

It represents the current routes checked from the repository:

```text
GET  /health
GET  /graph
POST /turn
POST /turns
POST /input
POST /run
POST /finalize
POST /release
```

## Local preview

Install Fern CLI:

```bash
npm install -g fern-api
```

Then, from the directory that contains the `fern/` folder:

```bash
cd docs
fern docs dev
```

The local development server hot reloads MDX and OpenAPI changes.

## Publish

Before production publishing, replace the placeholder organization/domain in:

```text
fern/fern.config.json
fern/docs.yml
```

Then:

```bash
cd docs
fern generate --docs
```

## Important: old GitHub Pages workflow

The BCG repository's previous docs workflow expected a static offline site and
ran `docs/check_offline_docs.py`.

This package is Fern source, so that old workflow must be replaced rather than
kept unchanged.
