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

## Run BCG while viewing live graph docs examples

Terminal 1:

```bash
uv run bcg construct server api_based \
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
