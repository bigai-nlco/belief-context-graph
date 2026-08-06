---
title: "Security"
description: "Protect credentials, trajectories, evidence, and graph audit data."
icon: "shield-check"
---

BCG processes high-context agent data. Treat every layer as sensitive.

## Credentials

- store secrets in environment variables or `~/.bcg/.env`
- reference them through `api_key_env`
- never commit populated model configuration or `.env`
- restrict file permissions
- rotate credentials used by model and embedding endpoints

## HTTP server

The built-in server does not authenticate requests. Put it behind:

- TLS termination
- identity-aware proxy or API gateway
- authorization rules
- rate and payload limits
- private network boundaries

## Artifact sensitivity

Artifacts may contain:

- full trajectories
- user and tool content
- exact evidence excerpts
- model prompts
- model output
- merge reasoning
- entities
- decision paths
- token and timing metadata

Set retention, access, encryption, and deletion policy accordingly.

## Prompt and graph injection

Graph content is derived from earlier messages and tool output. Treat it as untrusted data when inserting it into a model prompt. Delimit it, apply authorization filters, and avoid interpreting graph text as system instructions.

## Multi-tenant use

Use separate namespaces, output roots, problem IDs, and authorization boundaries. Do not rely on string naming conventions alone for tenant isolation.
