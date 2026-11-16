# DeepResearch Bench II research tools

The report runner supplies the same five research tools in Default and BCG modes. BCG additionally supplies its existing `bcg_recover` memory tool. Team orchestration tools are preserved. These bindings apply to newly started report tasks; running tasks and historical reports are unchanged. DRBench's separate staged-file tools are unchanged.

| Tool | Inputs | Behavior |
| --- | --- | --- |
| `web_search` | `query` | Existing guarded online search, with the task/team search budget. |
| `web_fetch` | `url` | Existing guarded HTML/text/PDF retrieval. Approved extracted text is cached and the response includes a `source_id`, original URL, preview and read-only Python path. |
| `find_in_source` | `source_id`, `query`, optional `start_line`, `max_matches` | Case-insensitive literal search within one cached source. Returns line numbers, character columns and excerpts. No new HTTP or LLM request. |
| `read_source` | `source_id`, optional `start_line`, `end_line`, `start_column` | Read numbered excerpts from a cached source. Returns a continuation cursor if the response limit is reached. No new HTTP or LLM request. |
| `run_python_code` | `code`, optional `timeout` | Run offline Python against approved sources and task-local scratch files, with persistent files between calls. |

All calls are recorded in the task's existing `tools.jsonl` and normal Agent trajectory. Source lookup and Python do not consume search/fetch quotas, although their returned text still contributes to subsequent Agent context. Existing URL/DOI/title/content blocks remain in force before source text enters the cache. Final reports cite original URLs, not source IDs or workspace paths.

## Reading a long source

1. Call `web_fetch(url="https://example.org/report")` and receive `S1`.
2. Call `find_in_source(source_id="S1", query="annual revenue")` to locate evidence beyond the preview.
3. Call `read_source(source_id="S1", start_line=120, end_line=150)` to inspect the surrounding text.
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
tools.jsonl
```

- Source cache: at most 200 versions, 2,000,000 characters per source and 20,000,000 characters in total. Sources truncated by retrieval or cache limits are explicitly marked partial; reading the cache does not fetch missing content.
- `read_source`: 1-based inclusive line numbers; default 100 lines, maximum 400 lines and 20,000 body characters per call. Follow the returned `start_column` cursor for very long lines.
- `find_in_source`: literal, case-insensitive query of up to 500 characters; default 10 and maximum 20 matching lines per call. Follow the returned line cursor for more matches.
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
RUN_REPORT_SANDBOX_TESTS=1 .venv/bin/python -m pytest \
  tests/test_report_research_tools.py tests/test_react_bcg.py -q
```

These tests use mocked source retrieval and a fake Agent/Graph backend, plus real Bubblewrap executions. They need no API key and incur no model/search charges. They cover source pagination, citation identity, blocked-source rejection, cross-call files, calculations, task isolation, read-only sources, hidden credentials and host processes, blocked network access, output/timeout limits, cancellation cleanup and production Default/BCG tool registration. They are capability/regression tests, not benchmark accuracy measurements.
