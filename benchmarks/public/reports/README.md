# Deep-research report benchmarks with BCG

These adapters run **the current FrontierAgent workflows**, not the older BCG
TypeScript reference agent. Both `stateful-react-agent` (single agent) and
`agent_team` (independent per-job graphs with cross-graph references) use their
existing BCG memory implementation. The default pipeline here is `agent_team`.

The data adapters, renderers, preparation and deterministic metrics were migrated
from `bigai-nlco/belief-context-graph`, branch `feature/add-benchmark`, commit
`f5340c6f78eb9bc072b93bf2ffef9337f78e9e99`. See [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
The old Agent runner and unrelated changes from that branch were not imported.

| Benchmark | Data | Retrieval | What is measured here |
|---|---|---|---|
| DeepResearch Bench II (`deepresearch_bench_ii`, short name `drb2`) | `tasks_and_rubrics.jsonl`, language/theme filters | Web search and full-page fetch, with task-specific source restrictions | Report length/citations and blocked-source citation audit; official-protocol rubric scores via `judge` |
| DRBench (`drbench`) | Original `subsets/` and `tasks/`, or full upstream clone | Task-scoped read-only text renderings of enterprise files, email and chat; web disabled | File read/surfaced recall, citation recall/precision and distractor citation rate |

**Scoring boundary:** `run` saves `official_score: null` and
`scoring_status: official_pending`. For DRB2, the separate `judge` command then
applies the official rubric protocol to the saved reports
([below](#drb2-rubric-judging)). DRBench insight-recall judging is not implemented
yet. Diagnostic metrics are **not accuracy**, and file citation recall is **not**
official insight recall. DRBench is the branch's local-file **L1 variant**, not
its full Docker/enterprise-service environment.

## Prepare

Run all commands from the FrontierAgent checkout. Agent credentials continue to
come from this repository's `.env` / exported `OPENAI_*` variables and the chosen
workflow profile. This runner does not silently use the old `~/.bcg/.env` keys.
The Graph model and its credentials remain configured on the Graph server.

```bash
uv sync --extra eval
uv run python -m benchmarks.public.runner.run_reports prepare drb2 drbench
```

Defaults to `benchmarks/public/datasets/`; set `FRONTIER_AGENT_DATASETS_DIR` or
`--data-root` to move it. Download/import records revision and checksums in each
dataset's `MANIFEST.json`. Datasets and results are gitignored. DRBench downloads
are approximately 100 MB according to the source adapter; rendering can take time.
Do not commit private grading material or redistribute restricted dataset content.

For existing data, use one import at a time:

```bash
uv run python -m benchmarks.public.runner.run_reports prepare drb2 \
  --from /path/to/DeepResearch-Bench-II --no-download
uv run python -m benchmarks.public.runner.run_reports prepare drbench \
  --from /path/to/drbench --no-download
```

You can also skip importing and supply `--data-file PATH` on `list`/`run`: DRB2
accepts its JSONL; DRBench accepts a data directory or a repository clone.

## Select tasks and run

```bash
uv run python -m benchmarks.public.runner.run_reports list drb2 --language zh
uv run python -m benchmarks.public.runner.run_reports list drbench --subset minival
```

Start a compatible BCG Graph server separately, then inspect settings without
calling any model or creating a result directory:

```bash
export BCG_GRAPH_URL=http://127.0.0.1:8850
uv run python -m benchmarks.public.runner.run_reports run drb2 \
  --pipeline agent_team --language zh --limit 2 --seed 42 \
  --workers 1 --out results/drb2-bcg-team-smoke --dry-run
```

Remove `--dry-run` to actually run the selected tasks. For single-agent BCG:

```bash
uv run python -m benchmarks.public.runner.run_reports run drb2 \
  --pipeline stateful-react-agent --language zh --limit 2 --seed 42 \
  --workers 1 --out results/drb2-bcg-react-smoke

uv run python -m benchmarks.public.runner.run_reports run drbench \
  --pipeline agent_team --subset minival --limit 2 \
  --workers 1 --out results/drbench-bcg-team-smoke
```

Omit `--limit` for the full selected split. `--task-ids '1,2'`,
`--task-ids 'drb2:1-5'`, and `--task-ids @ids.txt` are supported. Explicit IDs
retain their requested order and must exist after filters; otherwise `--seed`
shuffles deterministically before `--limit`. `--profile` accepts a workflow profile
name or a YAML file. Models and reasoning settings come from that profile, not
hard-coded benchmark defaults.

`--workers` is concurrent **questions**, not the number of sub-agents. Each worker
is a separate process. Tool limits are shared across the whole team per question:
`--search-calls 20`, `--fetch-calls 30`, `--fetch-chars 20000` by default. The outer
per-question timeout defaults to 3600 seconds for DRB2, 2400 for DRBench. `--timeout
0` disables only that outer ceiling; inner workflow turn/request limits still
apply. `--max-turns` optionally overrides coordinator and specialist turn limits.

Graph health is checked before starting model work. If Graph later fails, native
BCG preserves full context, and the result is marked `graph_fallback` instead of
silently describing a raw-context execution as a normal BCG run. The existing
workflow sandbox requirement remains in force (bubblewrap on local runs or an
explicitly configured isolated deployment).

## Data and tool boundaries

- Only the question and permitted company/persona/date context go to the Agent.
  Rubrics, gold insights, source labels and blocked article titles are host-only.
- Only `env.json`-listed files are rendered for DRBench. Sources outside the data
  root, answer-file entries and escaping destination paths are rejected. PDFs use
  FrontierAgent's existing `pypdf` dependency, so PDF text extraction can differ
  from the original branch's PyMuPDF renderer. This is recorded as renderer v2.
- Main and specialist agents receive the same per-task retrieval restrictions.
  Shell, arbitrary Python, downloads, tool discovery and unrestricted file tools
  are not exposed. Task board, delegation, reporting and BCG evidence tools remain.
- DRBench `read_file`, `glob_search` and `grep_search` access only staged files.
  `grep_search` is literal case-insensitive matching, not a shell or regex runner.
  Original filenames remain the citation identity.
- DRB2 filters blocked URLs, DOI/title matches and published evaluator paths.
  Search filtering is conservative: a mixed response containing a blocked source
  is suppressed as a whole. Fetch walks redirects explicitly, validates public
  addresses, checks every hop, bounds response bytes and screens extracted text.
  It uses direct HTTP rather than an opaque scrape proxy, with no hidden summary
  LLM. This prevents known blocked matches but cannot guarantee detection of all
  unknown mirrors or paraphrases; the final citation audit is still retained.

## DRB2 rubric judging

`judge` grades a finished DRB2 run's saved reports with the official evaluator's
protocol, ported in [`drb2_judge.py`](drb2_judge.py) from
[DeepResearch-Bench-II](https://github.com/imlrz/DeepResearch-Bench-II) at commit
`7c7a9a8` (Apache-2.0). The prompt, JSON parsing, exact rubric-item validation,
50-item batches, 10 attempts per batch, 150,000-character report truncation and
the score aggregation are verbatim. It never starts the Agent, so a judge model
can be changed or retried without rerunning research.

```bash
# Plan and settings only; no model call.
uv run python -m benchmarks.public.runner.run_reports judge \
  --out results/drb2-bcg-react-smoke --dry-run
uv run python -m benchmarks.public.runner.run_reports judge \
  --out results/drb2-bcg-react-smoke
```

The judge uses `JUDGE_API_KEY` / `JUDGE_BASE_URL` (falling back to `OPENAI_*`) and
`JUDGE_MODEL`, default `gpt-5.5-luna`; the upstream evaluator pins `gpt-5.5`.
`DRB2_JUDGE_REASONING_EFFORT` defaults to upstream's `medium` and is never lowered
on retries. The model actually used is written to every graded result. A 16-token
probe with the grading model and parameters runs first; an unusable judge stops the
command before any report is graded.

Pass the same `--data-root`/`--data-file` as the run: the rubrics are read from the
raw `tasks_and_rubrics.jsonl` rows, as upstream does, and their digest must match
the run's `dataset_sha256`. Only reports of tasks with agent status `completed` are
graded; other tasks are marked `not_judged`. Rerunning `judge` grades remaining
and `judge_error` tasks and skips scored ones; `--force` regrades everything and
`--task-ids` limits the selection. `--workers` (default 10) is concurrent reports.

Per task, `result.json` gains `rubric_scores` (`inforecall`, `analysis`,
`presentation`, `total`, `blocked_rate`), `official_score` (= `total`),
`scoring_status: official_protocol_scored` and the judge settings; the full
per-item scores, reasons, evidence and token usage are in
`trials/<id>/judge/drb2_rubric.json`. `summary.json` gains a `rubric` block with
four views of the official mean over tasks: `valid_only` (graded reports only, as
the leaderboard averages), `zero_filled` (tasks without a usable report count as
0), and `_leaderboard` variants of both that drop tasks 112 and 114, which upstream
currently excludes. The summary's `official_score` is `valid_only_leaderboard.total`;
`scoring_status` is `official_protocol_partial` until every selected task is
either graded or `not_judged`. These are scores computed **with the official
protocol by this repository**; leaderboard scores are recomputed by the
benchmark's maintainers.

`judge/drb2_result.jsonl` uses upstream's `run_evaluation.py` output format, so
upstream's `aggregate_scores.py --input <out>/judge/drb2_result.jsonl` reproduces
the per-task numbers. The `export/` reports can also be graded by the upstream
evaluator directly, for a cross-check against this port.

Note an upstream property kept as is: its JSON clean-up only parses replies whose
object keys start on separate lines (the pretty-printed format its prompt shows).
A compact single-line reply fails validation and is retried within the attempt
budget.

## Results, Graph records and resuming

```text
<out>/
  run.json                       selected IDs, settings and dataset fingerprints
  summary.json                   status counts, mean diagnostics, token totals/means
  _cases/<id>.json                private evaluator input, never exposed as a tool
  trials/<id>/
    report.md                    complete final report, not a short-answer extraction
    result.json                  status, diagnostics, usage, official score pending
    tools.jsonl                  permitted tool requests/results
    agent.log                    worker stdout/stderr
    sandbox/worktree/            DRBench text renderings + FILES.md
    agent/                       native Agent trajectories and BCG records
    judge/drb2_rubric.json       DRB2 only, after `judge`: per-item scores and usage
  judge/drb2_result.jsonl        DRB2 only: upstream-format judge output
  judge/judge.log                DRB2 only: per-batch judge log
  export/deepresearch_bench_ii/report/bcg-<pipeline>/idx-<n>.md
```

ReAct Graph records are normally `trials/<id>/agent/bcg/`; Team records are
`trials/<id>/agent/trajectories/bcg_team/`, including `index.json` and each graph's
`state.json`/`requests.jsonl`. Server-side construction logs stay on the server.
Local state snapshots retain graph nodes/relations, problem IDs and acknowledged
source turns. Existing full trajectories and exact graph-injection records remain.

Agent and Graph usage are separate. Input includes cached input; output includes
reasoning. Counts use only observed JSONL LLM usage events and per-graph cumulative
usage, without double-counting JSON trajectory copies. Missing usage is `null`, not
zero. Graph totals here include final/handoff writes; native online-only fields
remain available for separate analysis. No model price or dollar-cost estimate is
invented. Failed/timed-out attempts remain in status counts.

Rerunning the exact same command resumes tasks without `result.json`. Tasks with a
recorded error/timeout are also terminal and are not selectively retried. Different
settings/data/selection require a new output directory. An interrupted task that
already has partial artifacts also requires a new output directory rather than
silently appending a second attempt to its graphs and token logs.

The final report is produced by the research Agent itself; the separate optional
reporter is disabled for these runs. Official judges can consume saved reports
later without rerunning research. DRB2's export layout matches the source adapter's
official evaluator input convention. No official score is claimed until evaluated.

## Offline verification

```bash
uv run pytest -q tests/test_report_benchmarks.py tests/test_react_bcg.py tests/test_team_bcg.py
```

Tests use synthetic data and model/Graph doubles. They exercise the real BCG ReAct
and Agent Team nodes, including a specialist reading staged enterprise evidence.
They do not measure report quality, official benchmark scores, or model cost.
