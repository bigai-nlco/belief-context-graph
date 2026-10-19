# Changelog

## [Unreleased] 新增 DeepResearch Bench II 与 DRBench 报告型 benchmark

### 概览

`bcg benchmark` 新增两个“深度研究报告”类 benchmark，可以与现有 benchmark 用同一条命令运行和对比（default / bcg / summary / recent-only / rag）：

| Benchmark | 名称（别名） | 检索方式 | 本轮指标 |
|---|---|---|---|
| [DeepResearch Bench II](https://github.com/imlrz/DeepResearch-Bench-II) | `deepresearch_bench_ii`（`drb2`） | 开放 web：`web_search` + 新增 `web_fetch`；每题的专家源文章被屏蔽 | 报告长度、引用 URL 数、屏蔽源引用审计 |
| [DRBench](https://github.com/ServiceNow/drbench) | `drbench` | L1 本地文件模式：只读任务工作区（`read`/`grep`/`find`/`ls`），web 默认关闭 | 文件读取召回、引用召回/精确率、干扰项引用率 |

主要变化：

- **报告型任务**：Agent 的最终消息就是完整 Markdown 报告（不要求 `FINAL ANSWER:`）。报告型任务不计算 accuracy，状态为 `report_ready`，报告单独存档。
- **可插拔打分器**：本轮只运行确定性打分器。DRB2 rubric 评审与 DRBench 官方 `insights_recall` 的 LLM judge 接口已预留（`report_scoring.LLM_JUDGES`），下一轮接入后可直接对已保存的报告打分，无需重跑 Agent。
- **新工具 `web_fetch`**：读取网页/PDF 全文；优先调用 Serper 兼容的 `/scrape`（自动由 `SERPER_ENDPOINT` 推导，请求带 `includeMarkdown: true`，优先使用返回的 `markdown`），失败或正文过短时直连抓取（HTML→文本）；支持 `offset` 分页。
- **检索黑名单**：DRB2 每题的 `blocked` 源在 `web_search` 结果中被过滤，`web_fetch` 拒绝访问（不消耗预算，也不会请求代理）。
- **工作区沙箱 `BCG_TOOL_ROOT`**：报告型 benchmark 的文件工具只能访问任务工作区，数据集中的答案文件（`eval.json`、`qa_dict.json` 等）即使用绝对路径也读不到。
- **数据准备命令** `bcg benchmark prepare`、**选题命令** `bcg benchmark list`、**预演** `--dry-run`、**增强的 `--task-ids`**（作用域、区间、`@文件`、未知 ID 报错、保持给定顺序）。
- **每个 benchmark 的预算 profile**，可用全局参数、`--bench-opt` 或环境变量覆盖；实际生效值写入 `run.json`。

---

### 升级步骤

```bash
# 1. Python 依赖：DRBench 渲染 pdf/docx/pptx/xlsx 需要 benchmarks extra（uv.lock 已更新）
uv sync --extra benchmarks

# 2. 重新构建 Agent（web_fetch、黑名单、BCG_TOOL_ROOT 都在 agent-cli 中）
make build-agent            # 等价于 npm --prefix agent-cli run build

# 3. DRBench 的 grep/find 工具依赖 ripgrep 与 fd（不装时 Agent 会尝试下载，离线环境会失败）
sudo apt install ripgrep fd-find

# 4. 检查 ~/.bcg/.env 中的搜索代理配置（web_fetch 复用同一个 key）
#    SERPER_ENDPOINT=http://10.2.152.50:9999/search   -> web_fetch 自动使用 .../scrape
#    SERPER_API_KEY=<代理 token>
```

---

### 测试命令示例

#### 1）准备数据

```bash
# 数据已手动放在 datasets/ 下：只校验、写 MANIFEST.json、预渲染 DRBench（不会重新下载）
bcg benchmark prepare deepresearch_bench_ii drbench

# 数据不存在时自动从 GitHub 下载（DRBench 只下载 drbench/data，约 100 MB）
bcg benchmark prepare drb2 drbench --data-root datasets

# 固定到某个 commit/tag，保证可复现
bcg benchmark prepare drbench --revision <40 位完整 commit SHA 或 tag> --force

# 从本地 clone 导入（整仓 clone 即可，会自动定位 drbench/data）
bcg benchmark prepare drbench --from ~/code/drbench
```

手动放置数据时的目录结构（也兼容直接把整个仓库 clone 到 `datasets/drbench/`）：

```text
datasets/
├── deepresearch_bench_ii/
│   ├── tasks_and_rubrics.jsonl      # DRB2 仓库根目录的文件
│   └── MANIFEST.json                # prepare 生成
└── drbench/
    ├── subsets/                     # drbench 仓库 drbench/data/subsets
    ├── tasks/                       # drbench 仓库 drbench/data/tasks
    ├── _derived/                    # prepare / 首次运行生成的文本渲染缓存，可随时删除
    └── MANIFEST.json
```

#### 2）挑选题目

```bash
bcg benchmark list drbench --drbench-subset minival          # 15 道 minival 题
bcg benchmark list drbench --drbench-difficulty easy
bcg benchmark list drb2 --drb2-language zh                   # 66 道中文题
bcg benchmark list drb2 --drb2-theme "Health" --json         # JSON 行，便于脚本处理
```

#### 3）先预演，不调用模型

```bash
bcg benchmark run drb2 drbench \
  --task-ids "drb2:1,2;drbench:DR0001" \
  --dry-run
```

会打印：选中的题目、每个 benchmark 实际生效的预算与工具、第一题的 system prompt 和 user prompt；DRBench 还会实际装配一次工作区并列出文件。缺少 ripgrep/fd 时会给出警告。

#### 4）冒烟测试

```bash
# DRB2：中英各一题，对比 default 与 bcg，共享 web 缓存
bcg benchmark run drb2 \
  --task-ids 1,2 \
  --modes default,bcg \
  --web-cache results/web-cache \
  --workers 2 \
  --output-dir results/drb2-smoke

# DRBench：两道 easy 题，只用本地文件
bcg benchmark run drbench \
  --task-ids DR0001,DR0002 \
  --modes default,bcg \
  --output-dir results/drbench-smoke

# 两个一起跑，并分别设置超时
bcg benchmark run drb2 drbench \
  --task-ids "drb2:1;drbench:DR0001" \
  --bench-opt drb2.timeout=7200 --bench-opt drbench.timeout=1800 \
  --output-dir results/both-smoke

# 从文件读取题目清单
cat > smoke.txt <<'TXT'
# 中文 + 英文
drb2: 1, 67
drbench: DR0001-DR0003
TXT
bcg benchmark run drb2 drbench --task-ids @smoke.txt --output-dir results/smoke-file
```

#### 5）前期测试
```bash
# drb2 subset
bcg benchmark run drb2 \
  --task-ids "drb2:2,5,30,49,103,121" \
  --modes bcg \
  --web-cache results/web-cache \
  --workers 6 \
  --output-dir results/drb2

# drbench subset
bcg benchmark run drbench \
  --task-ids "drbench:DR0001,DR0004,DR0023,DR0024,DR0060,DR0097" \
  --modes bcg \
  --workers 6 \
  --output-dir results/drbench

```

#### 6）扩大规模

```bash
bcg benchmark run drbench --drbench-subset minival --modes default,bcg --workers 4
bcg benchmark run drb2 --drb2-language en --max-problems 10 --seed 7 --web-cache results/web-cache
```

#### 7）开发者测试

```bash
uv run pytest -q tests/test_report_benchmarks.py tests/test_benchmark_adapters.py
uv run ruff check . && uv run ruff format --check .
npm --prefix agent-cli test -- --run test/web-fetch-tool.test.ts test/web-search-tool.test.ts
make test        # 全量：Python 470 项 + agent-cli 177 项
```

---

### 参数说明

#### `bcg benchmark run`：新增参数

| 参数 | 示例 | 说明 |
|---|---|---|
| `--task-ids` | `--task-ids 1,5` | 不带作用域：对本次所有 benchmark 生效。DRB2 的 ID 是 `idx`（1–132），原始 `id`（如 `task1+`）和 `idx-1` 也可以用。 |
| | `--task-ids "drb2:1,2;drbench:DR0001"` | 按 benchmark 指定，组之间用 `;` 分隔。没被提到的 benchmark 不按 ID 过滤；提到了未运行的 benchmark 会报错。 |
| | `--task-ids DR0001-DR0005`、`60-62` | 区间展开，保留前导零。若某个 ID 字面上就含 `-` 且存在，则按字面匹配。 |
| | `--task-ids @smoke.txt` | 从文件读取；每行是一组（可带作用域），`#` 之后是注释。 |
| | | 未知 ID 会直接报错；显式指定的题目按给定顺序运行、不打乱（仍受 `--max-problems` 截断）。 |
| `--dry-run` | `--dry-run` | 只打印选题、生效预算、工具和 prompt，不调用模型、不写结果。 |
| `--timeout` | `--timeout 7200` | 每题 Agent 超时（秒），对本次所有 benchmark 生效；`0` 表示不限时。不传时用 profile 默认值（DRB2 3600、DRBench 2400、其余 900）。 |
| `--graph-max-turns` | `--graph-max-turns 400` | BCG 模式每题的 Graph 消息上限（含初始 system/user）。不传时用 profile 默认值（DRB2/DRBench 300、其余 160）。 |
| `--search-max-calls` | `--search-max-calls 40` | 每题 `web_search` 调用上限，覆盖 `SERPER_MAX_CALLS`。DRB2 默认 20。 |
| `--fetch-max-calls` | `--fetch-max-calls 50` | 每题 `web_fetch` 调用上限（含 `offset` 续读），覆盖 `WEB_FETCH_MAX_CALLS`。默认 30。 |
| `--fetch-max-chars` | `--fetch-max-chars 30000` | 每次 `web_fetch` 返回的最大字符数（≥500），覆盖 `WEB_FETCH_MAX_CHARS`。默认 20000；超出部分提示 Agent 用 `offset` 续读。 |
| `--bench-opt` | `--bench-opt drb2.timeout=7200` | 按 benchmark 覆盖，可重复。键：`timeout`、`graph_max_turns`、`search_max_calls`、`fetch_max_calls`、`fetch_max_chars`、`web`（布尔：`on/off/true/false/1/0`）。 |
| | `--bench-opt drbench.web=on` | 等价于 `--drbench-web`。 |
| `--web-cache` | `--web-cache results/web-cache` | 缓存 `web_search`/`web_fetch` 的原始响应，跨 mode、跨运行共享，减少开放 web 时变带来的对比噪声，也省额度。黑名单在读取缓存之后才应用，所以一份缓存可以供所有题目使用。 |
| `--drb2-language` | `--drb2-language zh` | 只保留 `zh` 或 `en` 题。 |
| `--drb2-theme` | `--drb2-theme "Health"` | 只保留某个主题（大小写不敏感），可用 `list` 查看主题。 |
| `--drbench-subset` | `--drbench-subset minival` | `val`（默认，100 题）、`minival`（15 题）或 `sanity`（SANITY0）。 |
| `--drbench-difficulty` | `--drbench-difficulty hard` | 只保留 `easy` / `medium` / `hard`。 |
| `--drbench-web` / `--no-drbench-web` | `--drbench-web` | DRBench 是否额外开放 `web_search`/`web_fetch`（默认关）。15 道题各有 3 条只能从网上获得的 external insight。 |

#### 预算优先级

从高到低：`--bench-opt BENCH.KEY=…` > 全局参数（`--timeout` 等）> 显式设置的环境变量（`SERPER_MAX_CALLS`、`WEB_FETCH_MAX_CALLS`、`WEB_FETCH_MAX_CHARS`）> profile 默认值。

要永久修改默认值，编辑 `bcg/apps/benchmark/profiles.py` 中的 `PROFILES`。

| Profile | timeout | graph_max_turns | search_max_calls | fetch_max_calls | fetch_max_chars | web | 文件工具 |
|---|---|---|---|---|---|---|---|
| `deepresearch_bench_ii` | 3600 | 300 | 20 | 30 | 20000 | 开 | 无 |
| `drbench` | 2400 | 300 | 20* | 30* | 20000* | 关 | read, grep, find, ls |
| 原有 QA benchmark | 900 | 160 | 不设置（沿用 Agent/环境） | — | — | — | 不变 |

\* 仅在开启 web 时有意义。

#### `bcg benchmark run`：已有参数在报告型 benchmark 下的行为

| 参数 | 说明 |
|---|---|
| `--max-problems N` | 过滤之后再截断；与 `--task-ids` 同时使用时截断的是指定列表。 |
| `--shuffle/--no-shuffle`、`--seed` | 未指定 `--task-ids` 时生效；指定 ID 时不打乱。 |
| `--data-root` | 默认 `datasets`。 |
| `--data-file drb2=PATH` | 指向 `tasks_and_rubrics.jsonl`（或其所在目录）。 |
| `--data-file drbench=PATH` | 指向 DRBench 数据目录（含 `subsets/`、`tasks/`，或整仓 clone）。 |
| `--allow-no-search` | 未设置 `SERPER_API_KEY` 时允许闭卷运行（DRB2 将没有任何工具）。 |
| `--overwrite` | 重跑已有结果。报告型任务的 `report_ready` 结果会被缓存复用。 |
| `--modes`、`--workers`、`--output-dir` | 不变。 |

#### `bcg benchmark list`

| 参数 | 示例 | 说明 |
|---|---|---|
| `BENCHMARKS...` | `list drb2 drbench` | 一个或多个 benchmark。 |
| `--data-root`、`--data-file` | | 同 `run`。 |
| `--drb2-language`、`--drb2-theme`、`--drbench-subset`、`--drbench-difficulty`、`--gaia-split` | | 同 `run`。 |
| `--json` | `--json` | 每行输出一个 JSON：`benchmark`、`task_id`、`aliases`、`details`。 |

#### `bcg benchmark prepare`

| 参数 | 示例 | 说明 |
|---|---|---|
| `BENCHMARKS...` | `prepare drb2 drbench` | 目前支持这两个。 |
| `--data-root` | `--data-root datasets` | 数据根目录。 |
| `--revision` | `--revision main`、`--revision <40 位 SHA>` | 下载时使用的分支、tag 或完整 commit SHA（默认 `main`；不支持短 SHA）。commit 通过 `git ls-remote` 解析（GitHub API 被限流时也能用）并写入 MANIFEST。 |
| `--from` | `--from ~/code/drbench` | 从本地 clone 导入（一次只能导入一个 benchmark）；有 `.git` 时会记录 commit。 |
| `--download/--no-download` | `--no-download` | 数据缺失时是否下载；`--no-download` 下缺数据会报错。 |
| `--force` | `--force` | 即使数据已存在也重新下载或导入（会清空 `_derived/`）。 |
| `--render/--no-render` | `--no-render` | 是否预渲染 DRBench 文本（不预渲染时，首次运行会自动渲染）。 |

#### 新增环境变量（由 runner 按题自动设置，一般无需手动设置）

| 变量 | 说明 |
|---|---|
| `BCG_FETCH_ENDPOINT` | `web_fetch` 的 scrape 端点。默认由 `SERPER_ENDPOINT` 推导（`/search` → `/scrape`）；使用官方 Serper 时为 `https://scrape.serper.dev`。 |
| `BCG_FETCH_API_KEY` | scrape 的 key，默认使用 `SERPER_API_KEY`。 |
| `WEB_FETCH_MAX_CALLS` / `WEB_FETCH_MAX_CHARS` | 抓取预算，见上文。 |
| `WEB_FETCH_DIRECT=0` | 关闭“scrape 失败后直连抓取”的兜底。 |
| `BCG_WEB_BLOCKLIST` | 每题黑名单 JSON 的路径（写在 `<output>/.blocklists/`，位于工作区之外）。 |
| `BCG_WEB_CACHE_DIR` | 由 `--web-cache` 设置。 |
| `BCG_TOOL_ROOT` | 文件工具的沙箱根目录，报告型 benchmark 自动设为任务工作区。 |
| `SERPER_LANGUAGE` / `SERPER_COUNTRY` | DRB2 中文题自动设为 `zh-cn`/`cn`，英文题为 `en`/`us`。 |

---

### 结果产物

```text
results/<run>/
├── run.json                 # 新增 effective_profiles（实际生效预算）、datasets（各数据集 MANIFEST）、web_cache_dir
├── summary.json
├── .blocklists/             # DRB2 每题黑名单（Agent 不可见）
├── .agent-config/<benchmark>/<mode>/     # 原为 .agent-config/<mode>/
├── export/deepresearch_bench_ii/report/<mode>/idx-<n>.md   # 官方评测器输入格式
└── <benchmark>/<mode>/
    ├── tasks/<task>.json
    ├── reports/<benchmark>-<mode>-<task>.md       # 新增：报告原文
    └── trajectories/ …（不变）
```

单题结果 JSON 新增字段：`kind`、`report`、`report_export`、`timeout_seconds`、`fetch_calls`、`fetch_calls_attempted`、`fetch_calls_blocked`，以及 `web_policy`（`fetch_refused` 拒绝抓取次数、`search_results_filtered` 被过滤的搜索结果数、`cache_hits` 缓存命中次数）。`metadata` 中不包含 rubric、黑名单或文件标签。

### 指标定义（确定性，不调用 LLM）

结果表的列名为了在 80 列宽的终端里放得下做了缩写：

| 列 | 指标 | 定义 |
|---|---|---|
| Ready | `reports_ready` | 状态为 `report_ready` 的题数 / 总题数。 |
| Chars | `report_chars` | 报告字符数（均值）；另有 `report_words`（中文按字计）。 |
| URLs | `urls_cited` | 报告中不重复的 URL 数。 |
| Fetch | `fetch_calls_mean` | 每题实际执行的 `web_fetch` 次数。 |
| Read | `file_read_recall` | DRBench：insight 文件中被 `read` 打开过的比例。 |
| — | `file_surfaced_recall` | DRBench：被 `read` 打开过，或在 `grep` 结果中有内容命中的 insight 文件比例（`ls`/`find` 只列文件名，不计入）。 |
| Cite | `citation_recall` | DRBench：报告中引用到的 insight 文件比例。文件按原文件名匹配；邮件导出按其中任一邮件主题匹配；聊天导出按频道名匹配；均为整词匹配，`report.pdf` 不会命中 `q3-report.pdf`。 |
| Prec | `citation_precision` | DRBench：被引用的已标注文件中 insight 文件所占比例。 |
| Distr | `distractor_citation_rate` | DRBench：干扰文件被引用的比例（越低越好）。另有 `distractor_read_rate`。 |
| Blocked | `blocked_citation_hits` | DRB2：报告中引用被屏蔽专家源的次数（URL/DOI/标题匹配），对应官方 judge 的 `-1`。 |

汇总中还有 `report_metrics_by_language`（DRB2）和 `report_metrics_by_difficulty`（DRBench）。超时、`max_tokens`、`agent_error`、`turn_limit` 的题目也会对已有的部分报告和轨迹打分，因此失败的尝试会拉低平均值，与 accuracy 的口径一致。

### 已知限制与说明

- 本轮不运行 LLM judge。DRB2 rubric 分数与 DRBench 官方 `insights_recall` 需要下一轮实现；接口和注册表已就位（`bcg/apps/benchmark/report_scoring.py`），DRB2 报告已按官方目录结构导出，可以直接交给官方评测器交叉验证。
- DRB2 官方 prompt 本身就包含“禁止查看以下文章与 URL”的说明，这里保持原文不改；工具层黑名单在此基础上做强制执行，报告审计用于兜底统计。
- DRBench L1 模式用文本渲染替代官方 Docker 环境（Nextcloud/Mattermost/Roundcube）。抽查显示约 97% 的 insight 黄金答案数字可以在渲染文本中找到，未命中的部分在上游自带的 `.md` 中同样缺失。
- 上游 DR0038 的 `env.json` 中有一处文件名大小写与实际文件不一致，loader 会在同目录下做大小写不敏感的匹配。
- `web_fetch` 直连兜底不解析 PDF（PDF 依赖代理的 `/scrape`，你的代理已确认支持）。
- 开放 web 没有快照，结果随时间变化；做 mode 对比时建议使用 `--web-cache`。
- `scripts/release-manifest.py --check` 在本次改动之前就已失败（JSON 解析错误），与本次改动无关。

### 代码变更清单

**新增**

- `bcg/apps/benchmark/drb2.py`：DRB2 loader、黑名单、引用审计。
- `bcg/apps/benchmark/drbench.py`：DRBench loader、文本渲染与缓存、工作区装配、确定性 recall。
- `bcg/apps/benchmark/profiles.py`：每个 benchmark 的 prompt、工具与预算 profile，以及覆盖逻辑。
- `bcg/apps/benchmark/selection.py`：`--task-ids` 解析。
- `bcg/apps/benchmark/report_scoring.py`：可插拔报告打分器。
- `bcg/apps/benchmark/prepare.py`：`prepare` 命令的实现。
- `agent-cli/src/core/tools/web-fetch.ts`：`web_fetch` 工具。
- `agent-cli/src/core/tools/web-policy.ts`：黑名单与响应缓存。
- `tests/test_report_benchmarks.py`、`agent-cli/test/web-fetch-tool.test.ts`。

**修改**

- `bcg/apps/benchmark/{models,loaders,runner,cli}.py`：报告型任务、新 benchmark 注册、runner 与 CLI 扩展。
- `agent-cli/src/core/tools/{web-search,path-utils,index}.ts`、`agent-cli/src/{index.ts,core/sdk.ts,cli/args.ts}`：搜索黑名单与缓存、`BCG_TOOL_ROOT`、`web_fetch` 注册与导出、帮助文本。
- `bcg/construct/_shared/context_selection.py`：BCG 建图时，把提到 `web_fetch` 的执行指令也排除在任务约束之外。
- `pyproject.toml`、`uv.lock`：`benchmarks` extra 新增 pymupdf、python-docx、python-pptx、openpyxl。
- `docs/fern/pages/guides/benchmarking.mdx`：新增章节。
