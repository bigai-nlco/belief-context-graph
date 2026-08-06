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
