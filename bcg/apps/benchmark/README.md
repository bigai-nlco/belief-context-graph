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

The reference Agent can be evaluated head-to-head in **Default**, **Recent-Only**, **RAG**, **Summary**, and **BCG** modes against **BrowseComp** and **BrowseComp (ZH)**. All modes use the same Agent model, prompt, and scorer; only context management changes. Every bounded mode permanently retains the initial user input.

Per-task Agent model request/response traces are recorded under `model-io/` by default. Pass `--no-record-model-io` for lower disk usage; trajectories, task results, and token summaries are still retained.

```bash
bcg benchmark run browsecomp browsecomp_zh --modes default,recent-only,rag,summary,bcg \
    --thinking off \
    --summary-model gpt-4.1-mini \
    --summary-thinking off \
    --recent-turns 2 \
    --rag-top-k 6 \
    --rag-max-chars 12000 \
    --max-problems 100 \
    --workers 8 \
    --output-dir results/browsecomp-comparison
```

RAG stores one SQLite database per task under `BENCHMARK/rag/rag-memory/` and writes retrieved-context snapshots under `BENCHMARK/rag/rag-contexts/`.

See [Evaluate with benchmarks](https://belief-context-graph.docs.buildwithfern.com/operate/benchmarking) for dataset setup, scoring, output artifacts, and every `bcg benchmark run` option.
