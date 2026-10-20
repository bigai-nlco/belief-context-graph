# Public benchmark integrations

This package contains FrontierAgent's evaluation runtime and integrations for
public benchmarks, including BrowseComp, HLE, OfficeQA, GDPval, and
FrontierScience. It owns the registry, dataset adapters, judges, subprocess
runner, download/standardization scripts, and gitignored local artifacts.

The bundled [FrontierSearchBench](../frontier_search_bench/) queries and scorers
are kept in a sibling directory. The adapter in
`families/frontier_search.py` lets the shared runner collect its answers while
keeping FrontierAgent-specific runtime code out of the benchmark
implementation.

DeepResearch Bench II and DRBench have a dedicated [report benchmark adapter](reports/README.md)
for the current BCG ReAct and Agent Team workflows. It preserves full Markdown
reports, enforces per-task source/file restrictions and records deterministic
diagnostics. Use `python -m benchmarks.public.runner.run_reports --help`; these
report-only runs do not use the short-answer accuracy judge in `run_subprocess`.

Run a one-question smoke evaluation after installing the eval dependencies and
placing the selected dataset under `datasets/`:

```bash
uv run python -m benchmarks.public.runner.run_subprocess \
  --benchmark browsecomp --pipeline stateful-react-agent --profile default \
  --limit 1 --concurrency 1 --out ./results/smoke
```

See the [benchmark registry](../README.md) and
[evaluation guide](../../docs/eval.md) for dataset setup, all benchmark keys,
FrontierSearchBench scoring, and extension instructions.
