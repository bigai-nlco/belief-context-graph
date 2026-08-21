# Benchmark (bcg.apps.benchmark)

Read-only consumers: benchmark loads dataset rows (never writes them) and
consumes construct-run artifacts produced by `bcg construct run` for
scoring. The artifact shapes it reads are covered by
`contracts/stream.schema.json` (result.json) and the memory document
contract; see `tests/test_artifact_contract.py`.

Data policy:

- Full benchmark datasets are **not** committed to the repository.
- `tests/fixtures/benchmark/` holds minimal fixed fixtures for loader
  smoke tests (`browsecomp.jsonl`); the adapter tests also synthesize rows
  in tmp directories.
- Loading unsupported or missing data fails loudly
  (`BenchmarkDataError`), never silently with an empty run.

## Benchmark Adapter

The reference Agent can be evaluated head-to-head in **Default**, **BCG**, and **Summary** modes against **BrowseComp** and **BrowseComp (ZH)**. All modes use the same Agent model, prompt, and scorer; only context management changes.

```bash
bcg benchmark run browsecomp browsecomp_zh --modes default,bcg,summary \
    --thinking off \
    --summary-model gpt-4.1-mini \
    --summary-thinking off \
    --max-problems 100 \
    --workers 8 \
    --output-dir results/browsecomp-comparison
```

See [Evaluate with benchmarks](https://belief-context-graph.docs.buildwithfern.com/operate/benchmarking) for dataset setup, scoring, output artifacts, and every `bcg benchmark run` option.
