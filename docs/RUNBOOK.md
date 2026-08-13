# Documentation runbook

## Preview the docs

From the BCG repository root:

```bash
npm install -g fern-api
cd docs
fern docs dev
```

Edit:

```text
docs/fern/pages/**/*.mdx
docs/fern/openapi.yml
docs/fern/apis/sdk-reference/openapi.yml
docs/fern/docs.yml
docs/fern/styles.css
```

and the local preview will hot reload.

## What to edit for each kind of change

### Change normal documentation wording

Edit files under:

```text
fern/pages/
```

### Change sidebar grouping

Edit:

```text
fern/docs.yml
```

The main navigation sections intentionally use `collapsed: true`.

### Change colors or visual details

Edit:

```text
fern/styles.css
```

The BCG palette is defined at the top as `--bcg-*` CSS variables.

For major surfaces that Fern supports natively, also edit the `colors:` block
in `fern/docs.yml`.

### Change an HTTP endpoint

Update:

```text
fern/openapi.yml
```

The HTTP API pages are generated from that specification.

### Change SDK Reference content

Update:

```text
fern/apis/sdk-reference/openapi.yml
```

The SDK Reference tab is generated from this specification, not from hand-authored
MDX. Each Python method is modeled as a synthetic endpoint (for example `POST
/graph/add-node` for `BCG.add_node`) with an `x-bcg-python-symbol` annotation and
an `x-fern-examples` Python code sample. `x-fern-explorer: false` keeps the
"Try it" playground disabled, since these are not real network endpoints.

<Warning>
`fern/pages/sdk/` no longer exists. It held the previous hand-authored SDK
pages (Graph / Memory / Runner / Model client / Configuration overviews and
sub-pages) and was removed once `docs.yml` switched the `sdk` tab over to the
generated `sdk-reference` API. Do not recreate files under that path — edit
`fern/apis/sdk-reference/openapi.yml` instead, and update the matching
`layout` entries under the `sdk` tab in `fern/docs.yml` if you add or rename
an endpoint/section.
</Warning>

## Run BCG while viewing live graph docs examples

Terminal 1:

```bash
uv run bcg construct server unified \
  --config ~/.bcg/config.yaml \
  --host 127.0.0.1 \
  --port 8848
```

Terminal 2:

```bash
npm --prefix dashboard run dev
```

## Publish

```bash
cd docs
fern generate --docs
```

Set your real Fern organization and docs domain before the first publish.
