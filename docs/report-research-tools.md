# DeepResearch Bench II research tools

The report runner supplies the same five research tools in Default and BCG modes. BCG additionally supplies its existing `bcg_recover` memory tool. Team orchestration tools are preserved. These bindings apply to newly started report tasks; running tasks and historical reports are unchanged. DRBench's separate staged-file tools are unchanged.

| Tool | Inputs | Behavior |
| --- | --- | --- |
| `web_search` | `q` (string or list) | Existing guarded online search, with the task/team search budget. |
| `web_fetch` | `url` | Existing guarded HTML/text/PDF retrieval. Approved extracted text is cached and the response includes a `source_id`, original URL, preview and read-only Python path. |
| `find_in_source` | `source_id`, `query`, optional `match_mode`, `offset`, `start_line`, `max_matches` | Case-insensitive literal matching: `any` keywords (default), `all` keywords on the same line, or one exact `phrase`. Ranked numbered excerpts and offset pagination. No new HTTP or LLM request. |
| `read_source` | `source_id`, optional `start_line`, `line_count`, `start_column`, legacy `end_line` | Read numbered excerpts from a cached source. Returns a continuation cursor if the response limit is reached. No new HTTP or LLM request. |
| `run_python_code` | `code`, optional `timeout` | Run offline Python against approved sources and task-local scratch files, with persistent files between calls. |

All bound-tool executions, including exhausted-budget refusals, are recorded in `tools.jsonl` and the Agent trajectory. These refusals have `outcome=budget_exhausted` and make no upstream request; do not count them as paid HTTP requests. The tool schemas remain available at quota zero. Source lookup and Python do not consume search/fetch quotas, although their returned text still contributes to subsequent Agent context. Existing URL/DOI/title/content blocks remain in force before source text enters the cache. Final reports cite original URLs, not source IDs or workspace paths.

## Reading a long source

1. Call `web_fetch(url="https://example.org/report")` and receive `S1`.
2. Call `find_in_source(source_id="S1", query=["annual revenue", "operating income"], match_mode="any")` to locate either phrase beyond the preview. For one exact phrase, use `query="annual revenue", match_mode="phrase"`.
3. Call `read_source(source_id="S1", start_line=120, line_count=31)` to inspect the surrounding text. Legacy `end_line=150` is also accepted and overrides `line_count`.
4. Use `run_python_code` for calculations or file processing, if needed.

```python
import csv
import json
from pathlib import Path

# For a cached source whose text is CSV; HTML/PDF sources contain extracted text.
rows = list(csv.DictReader(Path('/sources/S1.txt').open()))
values = [float(row['value']) for row in rows]
result = {'mean': sum(values) / len(values)}
Path('/workspace/result.json').write_text(json.dumps(result))
print(result)
```

Subsequent Python calls can read and modify `result.json`. Source files and `index.json` are read-only inside Python. Re-fetching unchanged content reuses its source ID; changed content gets a new ID, preserving earlier evidence. The source cache is task-local, not a global URL cache; a new `web_fetch` still counts as a fetch.

## Files and limits

Per-task host paths, relative to its trial directory:

```text
sandbox/
  worktree/             # Python scratch files, mounted at /workspace
  worktree-sources/     # Approved source text, mounted read-only at /sources
    S1.txt
    index.json          # Source IDs, URLs, titles, hashes and partial-text flags
    access.json         # Returned character spans, recent reads and source queries
tools.jsonl
research_context.jsonl # Execution-state block actually added to each Agent request
```

- Source cache: at most 200 versions, 2,000,000 characters per source and 20,000,000 characters in total. Sources truncated by retrieval or cache limits are explicitly marked partial; reading the cache does not fetch missing content.
- `read_source`: 1-based inclusive line numbers; default 100 lines, maximum 400 lines and 20,000 body characters per call. Follow the returned `start_column` cursor for very long lines.
- `find_in_source`: 1–16 distinct literal terms, up to 500 characters total; default 10 and maximum 20 matching lines per call. `any`/`all` split a string on whitespace; list entries preserve multiword phrases. `all` requires co-occurrence on one line, not across a document. `phrase` requires one string. Results are ranked by distinct matched terms, then line number. Keep query/mode/start_line unchanged and advance `offset` for subsequent pages. Excerpts are bounded; read the line for full context. The underlying Python `SourceStore.find` API retains its old `phrase` default for compatibility; the Agent-facing tool defaults to `any`.
- Python: 100,000 code characters; default 60 seconds, maximum 120 seconds; bounded output around 20,000 characters. Timeout and cancellation terminate the isolated process group.
- Python resource limits: 1 GiB virtual memory per process, 32 MiB per file, 128 file descriptors and a 256-process user limit. These are resource limits, not an aggregate task disk quota or cgroup memory quota.
- Python calls within one task/team are serialized to avoid concurrent scratch-file writes. Different tasks have separate workspaces and can execute concurrently.

## Sandbox requirements

Python uses the repository's existing Bubblewrap backend with a strict benchmark mode. It requires Linux, `bwrap`, and usable user, PID and network namespaces. No Docker daemon is needed on a compatible host. If isolation cannot start, the tool returns an error rather than executing on the host.

Each execution has fresh PID/network namespaces and private temporary files. It receives read-only system/interpreter libraries, the read-only approved-source directory and only its own writable task workspace. Host home directories, other task directories and host processes are not exposed. Provider credentials are not forwarded. Network access stays disabled even if the general sandbox environment allows it. The usual non-benchmark Python tool is unchanged.

Python libraries must already be installed in the runner's interpreter; the tool cannot download packages. The local smoke test exercises standard-library CSV/JSON/file operations and NumPy. This sandbox shares the host kernel; do not treat it as a separate VM or a complete hostile multi-tenant resource boundary.

## Offline smoke test

From the repository root, using the prepared environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_REPORT_SANDBOX_TESTS=1 \
  .venv/bin/python -m pytest tests/test_report_research_tools.py \
  tests/test_report_execution_state.py tests/test_react_bcg.py tests/test_bcg_task_state.py -q
```

These tests use mocked source retrieval and a fake Agent/Graph backend, plus real Bubblewrap executions. They need no API key and incur no model/search charges. They cover source pagination, citation identity, blocked-source rejection, cross-call files, calculations, task isolation, read-only sources, hidden credentials and host processes, blocked network access, output/timeout limits, cancellation cleanup and production Default/BCG tool registration. They are capability/regression tests, not benchmark accuracy measurements.

## Execution records and BCG task coverage

The single-Agent DRB2 harness appends a bounded `<research_execution_state>` block to the first system message in both Default and BCG. It lists remaining quotas, recent tool outcomes, available source IDs, returned character counts and recent read ranges. The complete access ledger is also available read-only at `/sources/access.json`. Records count only text actually returned by source previews/read/find, not entire documents fetched into the cache. They cannot observe what the Agent understood or which portions Python read internally.

After six consecutive tool-result rounds without new successful output, an advisory asks the Agent to reconsider outstanding requirements, recover relevant evidence, change the evidence route or synthesize with explicit gaps. Identical output, tool errors and zero new cached-source characters are mechanical signals, not a semantic quality judgment. Retries without tool results do not advance this counter. A new successful output clears it.

This block is added after BCG projection. It never enters belief extraction or persistent conversation history. No extra model calls, dynamically disabled tools, new stop conditions or changes to existing search/fetch/turn limits are introduced. Default retains its full chronological history. BCG retains its existing user inputs, recent raw interactions, preceding graph-turn window, rendering style and confidence propagation. The existing final-turn policy is unchanged.

BCG task-state extraction, within the existing model call, now explicitly preserves separately checkable requested outcomes and substantive missing evidence. It must not invent tasks or infer completion from a search, report or confidence. Task-state rendering allocates its existing budget to task/status records first, then real evidence paths, then recorded plans. It never edits stored beliefs or their confidence. This favors global requirement coverage; when the budget is tight, fewer optional evidence paths may fit. The recent belief graph is still injected separately in its original format.

### Fixed-input check against RUN-0025

Baseline code checkpoint: `711b052`. The local replay used the 120 original `find_in_source` inputs, original cached source files and final graph snapshot from DRB2 task 2. It made no model or network requests and did not modify the historical run.

| Check | Baseline | Updated |
| --- | --- | --- |
| Source queries returning no matching line | 87 / 120 | 8 / 120 |
| Task records shown at the same 1,800-character budget | 1 (`T2`) | 3 (`T2`, `T4`, `T5`) |
| Task-state characters | 1,697 | 1,663 |
| Optional factual edges in task-state excerpt | 2 | 0 |

For example, the original query `exclusion error exclusion targeting` had no exact phrase match in S6; keyword mode finds 99 matching lines and paginates them. More matches do not establish relevance or answer quality. The updated task-state excerpt retains the original program-analysis, failure-categorization and summary-table requirements instead of spending that space on one task's optional plan/path. No historical node is relabelled by this replay.

These checks establish tool behavior and context coverage only. The revised task extraction prompt has schema/grounding contract tests, not a live quality ablation. A subsequent paired Default/BCG run on identical tasks/model/budgets is needed to measure report scores, iterations and token cost. Both arms must use these shared tool and execution-record changes; old Default results alone would not be a fair control.
