# Contributing Guide

This repository welcomes human-authored and AI-assisted pull requests. AI PRs are first-class citizens here. We want transparency so reviewers know what to look for.

## Development Setup

Use `uv` for the Python package:

```bash
uv sync --all-groups
```

Install pre-commit hooks before opening PRs:

```bash
uv run pre-commit install
```

The dashboard is a separate Vite scaffold:

```bash
cd dashboard
npm install
npm run dev
```

Do not commit local secrets, `.env`, virtual environments, `node_modules`, build outputs, or generated cache files.

## Branch and PR Policy

Do not push directly to `main`. Open a pull request from a topic branch.

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages. Examples:

```text
feat: add belief graph schema
fix: handle empty model responses
docs: refine contribution guide
chore: update ci workflow
```

Before asking for review:

- Keep the PR focused and small enough to review.
- Fill in the PR description with what changed, why, and how it was tested.
- Disclose whether an LLM coding agent was used.
- Wait for CI to pass.
- Address Codex review findings that are relevant.
- Resolve bot review conversations after addressing them instead of leaving them for maintainers.

## Required Local Checks

Run the same quality checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q bcg tests
uv run python -m unittest discover -s tests
```

Run pre-commit before pushing:

```bash
uv run pre-commit run --all-files
```

Do not bypass hooks with `--no-verify` unless a maintainer explicitly asks you to do so and the reason is documented in the PR.

## Codex Review

If Codex is available locally, run a local review before requesting human review:

```bash
codex review --base origin/main
```

Address the findings before asking for review. If a finding is intentionally not addressed, explain why in the PR description or a PR comment.

If you are using an LLM coding agent, instruct it to resolve bot review conversations it has addressed instead of leaving them for maintainers.

## Code Style

Python code must pass Ruff lint and formatting.

Project conventions:

- Target Python 3.11+.
- Prefer explicit, typed public interfaces.
- Keep `bcg/py.typed` in the package.
- Use `bcg.tracing` for tracing integrations instead of importing Langfuse directly in feature code.
- Stubbed interfaces should fail explicitly with `NotImplementedError`.
- Avoid broad refactors in feature PRs.

Ownership and layout:

- Backend package code belongs under `bcg/`.
- Dashboard/frontend code belongs under `dashboard/`.
- Unit tests belong under `tests/`.
- Public examples belong under `examples/`.
- Reusable developer scripts belong under `scripts/`.

Documentation conventions:

- Prefer clear README, CONTRIBUTING, and public docs over scattered internal notes.
- Do not commit internal documentation unless it is necessary for the project.
- Use docstrings where they help public APIs or non-obvious behavior.
- Follow [PEP 257 docstring conventions](https://peps.python.org/pep-0257/) for Python docstrings.
- Do not commit personal agent instruction files such as `AGENTS.md`, `CLAUDE.md`, or local prompt notes unless maintainers explicitly request a repository-level version.

Never commit or push secrets in environment variables, source code, examples, fixtures, logs, or documentation.

## Sensitive Areas

Changes to LLM behavior, tracing, CI, environment variables, packaging, public API, or dashboard build configuration should include focused tests or a clear explanation when tests are not yet possible.

## AI Coding Agent Instructions

When using Codex, Claude Code, or another LLM coding agent:

- Tell the agent to read this guide before editing.
- Tell the agent to preserve unrelated user changes.
- Tell the agent not to commit secrets or generated local artifacts.
- Tell the agent to run the required checks and report exact results.
- Tell the agent to state which files it changed.
- Tell the agent to address and resolve bot review conversations after making fixes.
- Tell the agent not to rewrite unrelated docs, formatting, or generated files.

Use AI openly. The goal is transparent, reviewable work.
