# Confidence / task relevance change

## What changed

Factual `confidence` remains the existing posterior and is not rewritten. During query-aware context selection only, its ranking contribution is scaled by task relevance (the existing retrieval similarity):

`selection_confidence = confidence * (0.5 + 0.5 * task_relevance)`

At relevance 1.0 it retains all weight; at relevance 0 it retains half. This is a small, reversible change that helps prevent high-confidence but off-task beliefs from taking context budget. It intentionally does not claim that relevance is factual truth probability.

Files changed:
- `bcg/construct/_shared/context_selection.py`
- `tests/bcg/test_context_selection.py`
- `scripts/sandbox_confidence_smoke.py` (no dependencies or API key required)

## No-key sandbox smoke check

From the project root, with Python 3.11+:

```bash
python scripts/sandbox_confidence_smoke.py
```

Expected output:

```text
PASS: task relevance scales selection weight; factual confidence is unchanged
```

The pytest suites can be run after installing development dependencies:

```bash
uv sync --locked --extra dev --extra graph --extra document-readers --inexact
uv run pytest -q tests/bcg/test_context_selection.py tests/bcg/test_confidence_propagation.py
```

This project’s full BCG construction/agent workflow still needs model configuration for LLM extraction; the deterministic smoke check above does not.

## Pull and work on GitHub

The supplied link points to `main`; this archive itself has no Git metadata. To get an editable clone:

```bash
git clone --branch main https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph
git switch -c confidence-task-relevance
```

To work on an already-created branch instead, replace the previous two commands with:

```bash
git fetch origin <branch-name>
git switch --track origin/<branch-name>
git pull
```

Apply the files from this archive over that checkout, inspect with `git diff`, then:

```bash
git add bcg/construct/_shared/context_selection.py tests/bcg/test_context_selection.py scripts/sandbox_confidence_smoke.py CONFIDENCE_CHANGE_README.md
git commit -m "Adjust context confidence by task relevance"
git push -u origin HEAD
```

If the project owner protects `main`, push a feature branch and open a PR. I cannot push on your behalf from this snapshot because it contains no Git remote metadata or authenticated GitHub session.
