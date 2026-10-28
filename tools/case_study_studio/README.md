# BCG Case Study Studio

A run and annotation workbench for BCG Deep Research Agent cases. Shared mode lists existing experiments, lets signed-in reviewers start and stop tests, shows progress, and stores independent reviewers' annotations on the server. Optional local mode keeps the original browser-only workflow. The Chinese interface retains the eight-dimension rubric, source-linked critical processes, and failure attribution.

The workbench and copy editor use an [ACL Rolling Review-inspired palette](https://aclrollingreview.org/dates): white/cool-gray surfaces, navy controls and headings, burgundy links and selections, and separate warm warning states. Colors are bundled locally; the interface does not load styles or assets from the reference website.

## Shared runs and annotation

From the prepared Agent checkout:

```bash
.venv/bin/python tools/case_study_studio/serve.py --shared --host 0.0.0.0 --port 8876
```

Open `http://<server-LAN-IP>:8876`. Fill in a nickname, choose a `RUN-0001`-style run number, and open a finished task with **人工标注**. Task text, final report, trajectories, per-call injected beliefs, and complete Graph records load automatically. Existing report batches under `results/*/batch.json` are cataloged without rerunning models or modifying their files. Task completion is not an official quality score.

The shared Studio starts and displays **DeepResearch Bench II** only (internal key `drb2`). DRBench launch controls and tasks are not shown. Mixed historical batches retain their run numbers but list only DeepResearch Bench II tasks, with report counts, Token totals, and annotation counts scoped to those tasks. Historical batch elapsed time retains its original scope. DRBench result files and saved annotations are not deleted, and the separate command-line adapter and local-only importer are unchanged.

- **Progress:** updates every three seconds, including each task's model calls, turns, searches, fetches, Agent Token, Graph Token, status, and annotation counts. Unknown usage stays unknown. Running tasks become available for annotation after their results are fixed.
- **Identity:** each nickname receives a private personal identity code stored in that browser. Use **标注身份 → 查看个人身份码** to keep it for another browser. A nickname alone cannot take over an existing identity. Annotations are isolated per reviewer; another reviewer starts with a blank form. Revision checks prevent simultaneous tabs from silently overwriting one another.
- **Saving:** shared annotations autosave to SQLite, with a browser recovery draft when local storage is available. Evidence is frozen for stable citations. After a same-reviewer edit conflict, export the current draft before using **加载服务器版本**. Exporting a workspace never includes identity codes.
- **Existing local drafts:** local-only drafts remain in their original browser and are not automatically uploaded or assigned to a reviewer. The shared run list has no local-draft import entry.

### Starting and stopping tests

After signing in with a nickname, any reviewer can use **启动测试** and **停止测试** without an administrator code. Starting a test requires a test name, a server-approved Agent profile, task IDs, and concurrency. Stopping a test still requires clicking **确认停止**. API keys stay on the host; the browser cannot supply commands, paths, model endpoints, or credentials. A click on **确认启动测试** can consume API credit. Merely viewing results and annotating does not call models.

The server-approved profiles are in `profiles/`: local Qwen3.5-9B (Thinking on, up to two workers) or GPT-6-Luna via LiteLLM (low, up to six workers). Graph remains unified GPT-6-Luna / none with CPU MiniLM embeddings. Both retain two completed turns and use a 6,600-character graph view. A batch accepts up to 32 DeepResearch Bench II task IDs; the default set is `2,5,8,9,23,27,30,37,38,49,52,58,63,64,76,77,81,87,88,93,96,98,100,101,103,105,108,110,119,120,121,128`. The launch form reads these defaults and the task limit from the server. Search and fetch budgets default to zero (unlimited), with a finite per-task timeout and turn cap. Only one batch runs at a time; selected workers run concurrently within that batch.

Starting a job requires the existing `.venv`, prepared benchmark datasets, bubblewrap, and the sibling `../belief-context-graph-gpt56-luna/.venv`. Local Qwen must already be listening at `127.0.0.1:8001` with a 49,152-token context. Graph construction uses `LITELLM_OPENAI_API_KEY` from `~/.bcg/.env`; web search uses `SERPER_API_KEY` and optional `SERPER_ENDPOINT`. Optional `BCG_EMBEDDING_MODEL` selects the local MiniLM directory; otherwise `~/PLM/all-MiniLM-L6-v2` is used if present. No secrets belong in the YAML profiles. These server-side presets are independent of historical result folders.

Each launch writes `results/studio/RUN-<number>/`: `batch.json`, resolved non-secret profiles, `drb2/`, `graphs/`, and local diagnostic logs. A dedicated Graph server starts on an available loopback port and stops after completion/cancellation. Stopping a batch never stops the shared Qwen vLLM. Closing the browser or restarting only the Studio HTTP server does not cancel the worker. Unexpectedly exited jobs are marked interrupted, not automatically rerun.

Private state is under `runtime/case-study-studio/` (or `--state-dir`): SQLite identities/annotations/run catalog and frozen evidence. No administrator code is generated or checked. It is Git-ignored and excluded from all static routes. Back up this directory to preserve run numbering and annotations; use SQLite's backup API for a live database. This is a trusted-LAN service over plain HTTP, not an Internet-facing authentication system. Restrict port 8876 to the intended LAN; use TLS and stronger identity management before any public exposure.

## Local-only workbench

From the repository root:

```bash
python3 tools/case_study_studio/serve.py --port 8876
```

Open **http://127.0.0.1:8876**. No npm install, build, Agent installation, Graph server, or API key is needed. Python 3.10+ and a modern browser supporting ES modules, IndexedDB, Web Crypto, and native dialogs are required. Use the HTTP server, not `file://`. Keep a consistent hostname and port: browser storage is scoped to the origin. Stop the server with Ctrl+C.

### Share the local-only workbench on a trusted LAN

```bash
python3 tools/case_study_studio/serve.py --host 0.0.0.0 --port 8876
```

Open `http://<server-LAN-IP>:8876` from another computer on the same reachable network. Plain HTTP is supported: file fingerprints use a local SHA-256 fallback when Web Crypto's secure-context digest API is unavailable. Evidence is still parsed only in the visitor's browser, not uploaded to the server.

Without `--shared` there is no login or server-side annotation storage. Each browser has its own independent cases and annotations. Changing from localhost to a LAN IP changes the storage origin: export the JSON workspace at the old address and reimport it at the new one to migrate existing drafts. The host must remain running, and the network must allow TCP port 8876.

## Import a case

Choose **导入 Case** and select the automatically generated `case-studies.json.gz` batch bundle, or select several `case-study.json.gz` task bundles together. Gzip bundles are decompressed in the browser; no extraction or server upload is needed. Every task/attempt/mode becomes a separate Case with blank annotations. Drag-and-drop also supports multiple bundles.

After a native BCG workflow finishes, its run/artifact directory receives `case-study.json.gz`. The report benchmark runner and the subprocess benchmark runner also create a batch `case-studies.json.gz` at the output root after their selected tasks finish. Large batches are split into `case-studies-001.json.gz`, etc.; import the parts separately when the combined uncompressed size exceeds 250 MiB. Each worker writes only its own task bundle, and the batch is assembled after workers finish. Timeout/error attempts with available records are included, not silently discarded.

Bundles contain the task and delivered answer, Agent trajectories, BCG state, injection audits, and historical Graph snapshots. The Agent now saves each **acknowledged Graph update** locally in `graph_snapshots.jsonl`, including when the Graph server is remote. One update can acknowledge several Graph turns; unreturned intermediate states are not reconstructed. Existing runs without this file can include server-side `belief_graph.jsonl` from their local `graphs/` tree or an explicitly supplied `--graph-dir`. Missing records are shown under **记录提示**.

Package an existing task or batch without rerunning any models:

```bash
python -m workflows._shared.case_study /path/to/task-or-batch
# Optional: collect matching historical graphs from an existing server output directory.
python -m workflows._shared.case_study /path/to/task-or-batch --graph-dir /path/to/graphs
```

Selecting a directory prefers generated batch bundles, then single-case bundles, rather than mixing loose files from different tasks. Reimporting identical evidence skips it and preserves annotations. An updated evidence revision is a separate Case, so existing citations and annotations are not overwritten. Files are SHA-256 checked before any cases are saved; a corrupted bundle rejects the import as a whole.

Legacy single-case JSON/JSONL/Markdown/TXT import is unchanged. Use **补充附件** for loose evidence belonging to the active Case, not for multi-case bundles. Ordinary ZIP archives still need extraction. Case bundles and exported annotation workspaces are different formats: the former contain evidence, the latter also contain your annotations.

Supported records:

| File | What it provides |
| --- | --- |
| Benchmark task result (`tasks/<id>.json`, trial `result.json`) | Task identity, question when present, final response, recorded metrics |
| Agent trajectory JSON / JSONL | Messages, thinking text, tool calls/results; native `t: llm/result` and legacy completed message records |
| Agent session state with `history` | Full transcript, not necessarily the actual projected model requests |
| BCG `state.json` | Acknowledged graph inputs and the uploaded graph snapshot |
| BCG Graph `result.json`, or JSON with `nodes` / `all_nodes` and `relations` | Beliefs, confidence, source attributes, and one-hop relation navigation |
| BCG `requests.jsonl` | Per-call system-memory text, selected IDs and fallback audit |
| Legacy `graph-contexts/*.jsonl` | Graph render text; Graph turn numbers are not Agent call numbers |
| Legacy `model-io/*.jsonl` with request `payload.messages` / `payload.input` | Actual request System / Developer text and raw request/response records |
| `report.md`, other Markdown / TXT | Final results rendered as read-only Markdown; raw HTML is displayed as text |

Paths inside a result file are references only. Browsers cannot open arbitrary local paths; select the referenced files explicitly. Importing only a task result does not supply its trajectory or graph. The UI shows missing evidence rather than fabricating it. Streaming `message_update` records are omitted from the display when completed `message_end` records are present, and original JSONL line numbers remain available for citations.

One import is limited to 300 selected files, 200 MiB per selected file, 250 MiB of decompressed input, and 1,000 bundled Cases (up to 300 evidence files per Case). Large legacy streaming logs are reduced to completed messages before JSON parsing, while their original line numbers and full-file SHA-256 fingerprints are preserved. Loose evidence must belong to a single task. Generated bundles omit configuration, `.env`, grading references and arbitrary workspace files; credential fields and known environment key values are redacted. Treat all other task content as private and review it before sharing. No data is sent to external services; no remote scripts, fonts or images are loaded.

## Annotation workflow

Start with **开始标注教程** on the empty workspace, or **标注教程** in the sidebar or inside **分析**. The five-step guide covers file preparation, basic information, a source-linked event, dimension ratings, and completion/export. Each step includes click locations and a shortcut to the relevant annotation form. **打开真实 Demo** loads the real “Europa Clipper 与 JUICE 任务对比” run with empty annotations. The worked example points to call 4, beliefs B10–B12 and the recorded subsequent search; it does not pre-fill ratings.

The **标注步骤** bar is inside **分析** and replaces its duplicate tab navigation. Switching steps or using a tutorial shortcut updates only the annotation form. The evidence tab, selected Agent, search/filter/page, context call, Graph snapshot/node, expanded records, and reading scroll position remain unchanged. **从运行轨迹引入** explicitly opens the trajectory view; citing or deleting an event updates the form and event counts without rebuilding the evidence panel.

**运行轨迹** displays only Agent execution records. Reports, Graph inputs/snapshots, injection audits and Model I/O files remain in their dedicated views, not in the trajectory selector. A single Agent opens directly; multiple Agents use **选择 Agent**, labeled from recorded Agent metadata rather than filenames. Matching JSON/JSONL exports at the same path are alternate representations, with the more complete transcript selected and JSONL preferred on ties. Original source records and citation IDs remain unchanged.

**每轮 Belief 注入情况** pairs each injection with the same Agent's output for that call. Previous/next buttons change both together: the injected beliefs and relations are followed by rendered Assistant text, recorded Thinking, and structured tool calls. The injection area renders only the recorded `<belief_graph>` block as Markdown, without `context_blocks_guide` or the full system prompt. If no unique graph block is available, it shows an empty state rather than the full input. Complete original inputs remain in imported evidence and workspace exports. The corresponding tool results are available in a collapsed section. Matching uses explicit Agent-call numbers and recorded Agent identity or source ownership, never JSONL row order or a Graph turn number. Missing or ambiguous mappings are labeled rather than guessed. Early calls without a Graph still show their outputs. This is a read-only paired view; **运行轨迹**, citations, saved annotations, and imported evidence are unchanged.

**关键过程 → 轨迹证据** also shows the cited call's actual injected beliefs before its output or saved excerpt. It uses the same Agent/call mapping and Markdown rendering as the per-call view, showing only `belief_graph`. A tool-result citation uses the injection for the call that produced that result, not a later graph containing the result. Existing saved processes acquire this view without rewriting annotations or expanding excerpts. Missing or conflicting injection records are indicated explicitly; the final graph is never used as a substitute.

Assistant records display their text, any recorded Thinking/Reasoning, and one card per tool call with the tool name and arguments. An empty Assistant body no longer displays the full log object as if it were model output. Complete records remain in the collapsed **原始 JSON** section. The display is derived from the original records, including cases already saved in the browser, without rewriting evidence or existing annotations.

**任务和结果** contains only the original task and final answer/report. File inventories, metric details, import notices and case-management controls are omitted from this panel. Imported source data and metadata remain in the workspace and exports.

Final results render Markdown headings, lists, tables, emphasis, quotes, links and code blocks using a locally bundled parser. Wide tables scroll within the panel. Raw HTML is escaped, unsafe link schemes are disabled, and Markdown images appear as links without fetching them. Links open a new tab only when clicked. The original Markdown is preserved in storage and exports. Imported report content stays locked in the copy editor.

1. Fill in the reviewer, Agent type, task goal, and outcome in **基本信息**.
2. Read the transcript, **每轮 Belief 注入情况**, graph snapshot, and final report. Use **← 上一轮 / 下一轮 →** to move through the injection records; the current call label and position remain visible. In **Belief Graph**, **全部 Graph** shows the complete standalone graph record; **每轮 Graph** contains only historical snapshots, navigated with **← 上一份 / 下一份 →**. Filenames are omitted from these views; source information remains in the imported records and citations. The complete graph is not counted as the first historical snapshot. Missing historical records are shown as missing, not reconstructed from the final graph. Each view retains its own selected snapshot.
3. Create key processes only from **运行轨迹** using **引用为关键过程** beside an execution record. Optionally select an excerpt before clicking. **轨迹证据** appears first and opens by default: text and Thinking render as safe Markdown, and whole-record tool calls use the trajectory's structured cards. Selected or truncated quotations remain limited to the saved excerpt, never silently expanded to the full record. Source fingerprints and the original quotation remain under **来源与原始摘录**. The form shows only three text areas: **BCG 应发挥的作用**, **实际发生的情况**, and **影响与改进线索**, followed by attribution and evidence-strength selectors. Focus on where an expected BCG benefit failed to materialize, with one or two sentences per field. Names, source positions, excerpts and the trajectory's Agent name are carried over automatically; no judgment is inferred. **补充证据检查** contains optional subproblem/source/graph/injection/alternative-factor details, and **过程信息** allows renaming; both start collapsed. Existing detailed annotations stay intact and editable; Markdown exports omit empty optional fields but retain filled ones. The process panel's **从运行轨迹引入** button opens the trajectory without creating a blank entry. Graphs, injection audits and final reports remain supporting evidence, not process-creation entry points.
4. In **维度评价**, select the dimensions evidenced by the key processes you recorded. All eight appear at the top as multi-select buttons; only selected dimensions expand below. Selection means the trajectory shows an opportunity for BCG to affect that dimension, not that the task type suggests one or that the Agent necessarily used it. Link one or more **关键过程**, then rate **是否利用** and **是否产生效果** and explain the judgment. **查看** opens the process without changing the left panel; its **关联维度** buttons return to a selected dimension.
5. **归因总结** first shows a read-only recap of selected dimensions, their ratings, notes and linked processes. Links reopen the dimension/process without changing the evidence panel. This recap is derived from annotations, not an automatically inferred conclusion. Write the main benefit or problem and evidence strength, then optionally select multiple priority improvement stages, describe a concrete change and record how to validate it. Comparison conditions and alternative sources are in an optional disclosure. The completion checklist updates as fields are filled. **完成标注** checks each applicable dimension's links. Positive or negative attribution needs a process linked to that specific dimension with source reference, behavior, impact and evidence strength; an unrelated completed process does not satisfy it. Later edits reopen the review as a draft.

The conclusion keeps existing benefit, problem, evidence, improvement and comparison fields intact. The optional `focus` (selected improvement stages) and `verification` fields are preserved in workspace JSON and Markdown reports. Older workspaces without them still open unchanged; no causes or suggestions are prefilled.

Links use stable process IDs, not the displayed E01/E02 numbers. Renaming or reordering processes keeps associations intact. Deleting a process warns about affected dimensions and removes only its links. Older free-text evidence is retained as a read-only note and is not automatically treated as a process link.

Progress counts selected dimensions only. Deselecting hides a dimension without erasing its ratings, notes or process links; reselecting restores that draft. If no relevant dimensions were found after reading the trajectory, explicitly check **已查看轨迹，未发现相关维度** before completing the review. Legacy **无法判断** opportunity ratings stay visible for confirmation instead of silently becoming confirmed selections. The stored `opportunity` field remains compatible with existing JSON workspaces.

Click **＋ 自定义维度** to add a case-local dimension with a name and optional description. It is selected immediately and uses the same process links, utilization/effect ratings, evidence notes and completion checks as the eight preset dimensions. Custom dimensions can be renamed or deleted with confirmation; deleting a dimension leaves the original processes and other ratings intact. Names, descriptions, ratings and links autosave and round-trip through workspace JSON; selected custom dimensions also appear in Markdown/CSV exports. Existing workspaces without custom dimensions remain supported.

No numeric aggregate score is generated, because one event may support several dimensions. Imported benchmark status is not treated as a human correctness judgment. Token counters use provided task metrics, not sums over possibly duplicated logs. Missing counters remain unknown. The Graph usage label exposes whether the source counts final flushes or leaves the accounting scope unspecified.

Previously saved events, including those created from other views or entered manually, remain intact and can still be reviewed and exported. New events require a source record in the Agent trajectory; manual creation and editable source-location fields are no longer offered.

## Save, export, and restore

- Edits autosave into **IndexedDB in the current browser**. A visible warning is shown if saving fails. Clearing site data or using another browser will not preserve these drafts.
- A newly opened case starts at **基本信息**. The last active case and each case's review step, evidence tab, selected process/Graph/context record, filters, page and outer-panel scroll positions are saved separately in browser `localStorage`. Refreshing or switching cases restores that view without changing annotations, review status or timestamps. Old cases with no saved view also start at basic information. View state is not included in workspace exports; the copy editor uses its own view-state key.
- Each entry in **CASE QUEUE** has a delete button. Confirming removes only that case, its imported evidence and annotations from this browser. Other cases and original files stay unchanged. Export a workspace first to keep a backup.
- **工作包 JSON** exports all loaded cases, evidence and annotations. Reimport it to restore the workspace. Replacing existing cases requires confirmation. Check for sensitive content before sharing.
- **Markdown** exports the current case study, selected dimension-to-process links, citations and source-file fingerprints.
- **CSV** exports one row per selected dimension with linked process labels, source references and stable `event_ids`. Cases without a selection retain one case-level row. Unselected drafts remain in the full JSON workspace, not in the evaluation tables. Spreadsheet formulas in user text are escaped.

This is a single-reviewer local workbench, not a shared annotation server. The server only serves an allowlist of static assets, defaults to localhost, and has no file-upload endpoint. LAN access shares the interface, not the host's cases or other reviewers' annotations. Share exported workspaces explicitly when needed.

## Edit the interface copy

Open **`/copy-editor.html`**, or click **文案编辑** at the bottom of the workbench. The editor shows the actual page on the left and a copy-editing panel on the right.

1. Use **点选文字** to select a heading, label, button, tutorial paragraph or input hint. The searchable catalog also includes hidden tutorial pages and unopened form panels.
2. Edit the text on the right. Changes automatically preview and autosave to a separate browser-local draft. **恢复此条** and **撤销上次修改** undo edits.
3. Use **浏览页面** to operate the preview, open the real Demo, switch tabs or navigate the tutorial. Switch back to **点选文字** to continue editing.
4. Click **导出修改稿** to download `bcg-ui-copy.json` and send it to the maintainer. **导入修改稿** restores that file on another browser.

Matching interface strings update together. Dropdown display labels can change without changing the saved annotation values. Dynamic numbers use `{{n1}}`-style placeholders that must remain intact. Task text, reports, Graph nodes, evidence, filenames and filled-in annotations are not copy-editing targets. Drafts do not change the standard page or other visitors' views.

The maintainer can review and publish an exported draft without editing the original run files:

```bash
# Validate the draft and show changes without writing.
node tools/case_study_studio/apply-copy.mjs /path/to/bcg-ui-copy.json

# After review, apply the shared interface copy.
node tools/case_study_studio/apply-copy.mjs /path/to/bcg-ui-copy.json --write
```

The second command atomically updates `copy-overrides.json`. Refreshing the standard page loads that published version; the existing static server need not restart. The command rejects stale drafts whose base revision no longer matches the published copy. Text is applied as text, never HTML. There is no anonymous LAN endpoint that writes into the repository.

## 共享页面中的 AI 标注

每条任务的「结果」栏有「人工标注」和「AI 标注」两个入口。任务结束且已有非空报告后可点击启动 AI 标注。Review 使用 `~/.bcg/.env` 中的 `LITELLM_OPENAI_API_KEY`，模型固定为 `gpt-6.1-sol`，`reasoning_effort=medium`。不继承 Agent 的模型和 Key。

后台独立进程按队列执行，默认同时处理一份 Review。关闭网页或重启网页服务不会终止已启动的 Review。状态和分批进度显示在任务旁。完成后再次点击「AI 标注」进入原标注工作台，阅读关键过程、维度评价、归因总结及对应轮的 Belief 和输出。AI 标注只读，与每个昵称的人工标注分开保存，不自动覆盖人工草稿。「标注情况」在任一人工标注完成或 AI 标注完成时显示「已标注」，并注明人工 / AI 来源。人工草稿和排队、运行中的 AI 标注显示「标注中」，不计入已标注题数。

同题重复点击复用同一份 Review。失败后点击「重试 AI 标注」，复用已经完成的抽取批次。失败的 Review 不会自动重新入队，也不会给历史任务自动补跑。单次运行中的网络和格式修正重试沿用独立流程的次数上限。AI Review 的 Token 单独记录，不加入 Agent / Graph 的实验 Token。

后台数据存放在服务状态目录的 `ai-reviews/<review-id>/`，包括完整 Prompt、引用、结果、用量和工作日志。SQLite 的 `ai_reviews` 表记录任务关联和执行状态。Review 与人工页面读取同一份冻结的证据包，原始任务文件保持不变。

独立命令行流程见 [`../ai_review/README.md`](../ai_review/README.md)。

## Tests

```bash
node --test tools/case_study_studio/tests/*.test.mjs
uv run pytest -q tools/case_study_studio/tests tests/test_ai_review.py

# Optional browser checks with the existing server and installed Chrome.
STUDIO_URL=http://127.0.0.1:8876 node tools/case_study_studio/tests/copy_editor_smoke.mjs
```

AI Review 的浏览器集成测试使用独立模拟服务，不调用真实模型。`--root` 每次指定新的临时目录：

```sh
uv run python tools/case_study_studio/tests/shared_smoke_server.py \
  --root /tmp/bcg-ai-review-qa --port 8877 --mock-ai
# 在另一个终端运行：
STUDIO_URL=http://127.0.0.1:8877 node tools/case_study_studio/tests/ai_review_browser_smoke.mjs
```

## Real task Demo

**真实任务 Demo** loads a real BCG Deep Research Agent run, using GPT-6-Luna through LiteLLM (Agent low / Graph none, two recent completed turns, 6600-character Graph budget). The custom research task compares NASA Europa Clipper and ESA JUICE; it is not a scored benchmark.

- 22 Agent calls, 29 web searches, 186.632 seconds.
- Final Graph: 74 nodes and 53 relations; no graph fallback.
- Agent: 123,949 tokens. Graph LLM: 38,753 tokens including final flush. Combined: 162,702 tokens. Local embedding tokens are retained in the source snapshot but excluded from these LLM totals.
- Includes the unedited generated report, native trajectory, all 22 injection audits, the complete final Graph state, and 46 historical snapshot views.
- Snapshot views retain each snapshot's own node content, confidence, source references and relations; repeated audit collections are omitted from the timeline to reduce download size. Complete final evidence and confidence history remain in `state.json`.
- `demo-case.json` is a sanitized local asset with source SHA-256 hashes. It contains no API credentials or private runtime config. Original run files remain under `results/case-study-demo-20261020/litellm/`.

The Demo has empty annotation fields and no assigned quality score. The run includes repeated searches for already-injected information, which the tutorial uses as a concrete analysis starting point. Previously saved fictional demos and user annotations are kept as separate cases.

To rebuild the asset from this run:

```bash
python3 tools/case_study_studio/build_demo.py results/case-study-demo-20261020/litellm
```
