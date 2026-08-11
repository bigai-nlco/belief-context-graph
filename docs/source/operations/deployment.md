---
title: "Deployment"
description: "Operate the graph construction server and its model dependencies."
icon: "cloud"
---

## Single-host deployment

A practical first deployment runs:

- BCG HTTP server
- selected model endpoint
- optional local embedding model
- optional stance and NER assets
- persistent output volume

```bash
bcg construct server unified \
  --config /etc/bcg/model_config.json \
  --host 0.0.0.0 \
  --port 8848 \
  --output-dir /var/lib/bcg/outputs
```

## Existing server mode

The reference Agent can connect to a separately hosted graph server. Run `bcg setup` and choose **Connect to an existing Graph server**.

## Health checks

Use:

```http
GET /health
```

The response includes active and all-known problem IDs.

## Process model

The built-in server is a threaded Python HTTP server. It is suitable for local development, controlled internal deployments, and reference integration.

For larger production deployments, place it behind infrastructure that provides:

- TLS
- authentication and authorization
- request size limits
- rate limits
- process supervision
- structured logs
- persistent shared session coordination if horizontally scaled
- output retention and encryption

## Kubernetes example

The repository includes:

```text
deploy/tonggraph-server.yml
```

Review and adapt it to your own model endpoint, storage, credentials, and network policy.

<Warning>
Do not expose the built-in HTTP server directly to an untrusted network. It has no native authentication layer.
</Warning>
