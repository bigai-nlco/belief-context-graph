# Changelog

## 测试子集
```bash
# drb2 subset: drb2:2,5,30,49,103,121
bcg benchmark run drb2 \
  --task-ids "drb2:2,5,30,49,103,121" \
  --modes bcg \
  --thinking low  \
  --web-cache results/web-cache \
  --workers 1 \
  --output-dir results/drb2

# drbench subset: drbench:DR0001,DR0004,DR0023,DR0024,DR0060,DR0097
bcg benchmark run drbench \
  --task-ids "drbench:DR0001" \
  --modes bcg \
  --thinking low  \
  --workers 1 \
  --output-dir results/drbench

```

## [Unreleased] 官方对齐的 LLM-as-judge：DeepResearch Bench II 与 DRBench

### 概览

新增 `bcg benchmark judge` 命令，用**与官方逐项对齐**的 LLM judge 给已保存的报告打分；`bcg benchmark run` 新增 `--judge` 选项（默认关闭），可在运行结束后立即评测。评测与运行解耦：换 judge 模型、重评失败项，都不需要重跑 Agent。

| Benchmark | Judge | 默认模型 | 指标 |
|---|---|---|---|
| DeepResearch Bench II | 官方 rubric judge（三值 1/0/-1） | `gpt-5.5-luna`（官方为 gpt-5.5） | total、inforecall、analysis、presentation、blocked_rate |
| DRBench | 官方 `score_report` 流程 | `gpt-4o-mini`（`score_report` 库默认） | insights_recall、distractor_recall（越低越好）、report_quality |

judge 默认使用与 Agent 相同的 endpoint（`OPENAI_BASE_URL` / `OPENAI_API_KEY`，例如 litellm）。DRBench 的 factuality 将在下一轮实现。

### 与官方对齐的内容

**Prompt 逐字节一致。** 所有 prompt 都用 `ast` 从官方源码中提取（DRBench 的 prompt 在官方代码中是 f-string），存放在 `bcg/apps/benchmark/judges/prompts/`。`SOURCES.json` 记录了上游 commit（DRB2 `b38f360`，DRBench `0d699ec`）和每个 prompt 的 sha256，由测试锁定。

**DeepResearch Bench II（`run_evaluation.py`）**

- 报告超过 150,000 字符时截断。
- rubric 按 info_recall → analysis → presentation 的顺序展开，每 50 条一批，每批都附带完整报告、task 原文和原样的 `blocked` 字段。
- 请求为单条 user 消息，参数为 `reasoning_effort=medium`、`max_completion_tokens=32768`。
- 每批最多尝试 10 次；JSON 解析和条目校验逻辑都是官方原样移植。
- 任意一批失败，整题记为 error，并且**不计入平均**（与官方汇总规则一致）。

**DRBench（`score_report`）**

- 报告截断到 60,000 字符，然后由 LLM 拆解为 insight（`temperature=0`，3 次尝试）。
- 对每条黄金 insight 或干扰项各做一次评分（`temperature=0.7`、`max_tokens=1000`，即官方 `AIAgentManager` 的默认值；3 次尝试）。
- report_quality 包含 5 个维度（`temperature=0`），失败时使用官方的重试 prompt；三次都失败则给中性分 0.5。

### 与官方不同之处（均记录在每份评测结果的 `deviations` 中）

1. **report_quality 使用报告原文。** 官方 `score_report` 处理纯文本报告时没有传入 `report_text`，实际评的是空报告。
2. **失败的 judge 调用会被标记。** 官方会把失败的调用静默计为 0 分（report_quality 为 0.5）。这里分数仍按官方规则计算，但同时计入 `judge_failures` 并在汇总中提示，可以用 `--retry-failed` 只重评这些项。
3. **缺少字段的回复会重试。** DRBench 的评分回复如果是 JSON 但缺少 `answer` 等字段，这里按失败重试；官方会因为 `KeyError` 直接中断。
4. **不复刻官方的模型白名单。** DRBench 官方遇到不在白名单内的模型名，会静默换成 Llama-3-8B；这里指定什么模型就用什么模型。

### 官方行为提示

DRB2 官方的 JSON 清洗正则无法解析**单行的多字段 JSON**，只接受每个 key 独占一行的格式化 JSON（即 prompt 示例中的格式）。这里忠实复现了这一行为：遇到单行 JSON 会触发重试，不影响最终分数，但可能增加调用次数。

### 参数自动适配

如果 endpoint 拒绝某个官方参数（例如推理模型拒绝 `temperature`，或要求用 `max_completion_tokens` 代替 `max_tokens`），会自动去掉或改名后重试一次，并把调整写入评测结果的 `judge.param_adaptations`。

注意：如果用推理模型担任 DRBench judge，官方的 `max_tokens=1000` 可能不够容纳推理 token。可以用 `--judge-param drbench.scoring.max_tokens=8000` 调大，但这会标记为偏离官方。

### 命令与参数

#### `bcg benchmark judge RUN_DIR`

| 参数 | 示例 | 说明 |
|---|---|---|
| `RUN_DIR` | `results/drb2` | 某次 `run` 的 `--output-dir`。 |
| `--benchmark/-b` | `-b drbench` | 只评测指定的 benchmark（可重复）；默认评测目录中存在的全部。 |
| `--modes` | `--modes default,bcg` | 只评测指定的 mode。 |
| `--task-ids` | `--task-ids "drb2:1,2;drbench:DR0001-DR0003"` | 语法与 `run --task-ids` 相同。 |
| `--judge-model` | `--judge-model drb2=gpt-5.5-luna --judge-model drbench=gpt-4o-mini` | 不带前缀的 `MODEL` 对所有 benchmark 生效；`BENCH=MODEL` 只对一个 benchmark 生效（可重复）。默认值见上表。 |
| `--judge-base-url` | `--judge-base-url http://litellm:4000/v1` | 默认使用 `OPENAI_BASE_URL`，与 Agent 相同；也接受完整的 `/chat/completions` 地址。 |
| `--judge-api-key-env` | `--judge-api-key-env JUDGE_KEY` | 存放 key 的环境变量名，默认 `OPENAI_API_KEY`。 |
| `--judge-timeout` | `--judge-timeout 900` | 单次请求的超时时间（秒），默认 600（与 DRB2 官方一致）。 |
| `--judge-param` | `--judge-param drb2.reasoning_effort=high` | 覆盖官方参数，可重复。**一旦使用，就会标记为偏离官方**，并生成新的 judge-id。 |
| | `--judge-param drbench.scoring.temperature=0` | DRB2 可用的 key：`reasoning_effort`、`max_completion_tokens`、`temperature`、`chunk_size`、`max_retries`、`max_paper_chars`。DRBench 可用的 key：`{extraction,scoring,quality}.{temperature,max_tokens,max_completion_tokens,reasoning_effort}`。 |
| `--metrics` | `--metrics insights_recall` | DRBench 要计算的指标，默认三个全算。 |
| `--judge-workers` | `--judge-workers 8` | 同时评测的报告数，默认 4。每份报告内部的调用顺序与官方一致，按顺序执行。 |
| `--overwrite` | | 对已有评测结果的报告强制重评，不使用缓存。 |
| `--retry-failed` | | 只重评状态为 error 或含有失败调用的报告；其中已成功的调用会从缓存中复用。 |
| `--dry-run` | | 列出待评测的报告，并粗略估算调用次数和输入 token 数。 |
| `--data-root`、`--data-file` | | 评分材料（rubric、黄金 insight）的位置，同 `run`。 |

#### `bcg benchmark run` 的新增或变化参数

| 参数 | 示例 | 说明 |
|---|---|---|
| `--judge/--no-judge` | `--judge` | 运行结束后，对本次运行的 DRB2 / DRBench 报告做官方评测，默认关闭。评测的题目和 mode 与本次运行相同。 |
| `--judge-model` | `--judge-model drb2=gpt-5.5-luna` | 现在可以重复，并支持 `BENCH=MODEL` 写法。不带前缀的值同时作用于 BrowseComp judge 和报告 judge；BrowseComp 的默认值仍是 Agent 模型。 |
| `--judge-param` | 同上 | 同 `judge` 命令。 |
| `--judge-workers` | `--judge-workers 8` | 同 `judge` 命令。 |
| `--judge-timeout` | | 报告 judge 的超时取 `max(--judge-timeout, 600)`。 |

judge 的参数会在运行开始**之前**完成校验，避免跑完 Agent 之后才发现参数写错。

### 使用示例

```bash
# 先预估
uv run bcg benchmark judge results/drb2 --dry-run

# 按官方默认配置评测
uv run bcg benchmark judge results/drb2

# 只评测部分题目和 mode
uv run bcg benchmark judge results/drb2 --task-ids "drb2:1,2" --modes bcg

# 换一个 judge 模型（与原有结果并存，judge-id 不同）
uv run bcg benchmark judge results/drbench -b drbench --judge-model drbench=gpt-4o

# 只重评失败项
uv run bcg benchmark judge results/drb2 --retry-failed

# 运行完成后立即评测
uv run bcg benchmark run drb2 --task-ids 1,2 --modes default,bcg --output-dir results/drb2 --judge
```

### 产物

```text
results/drb2/
├── deepresearch_bench_ii/bcg/judgements/2.gpt-5.5-luna-3e86a7c0.json   # 每份报告的评测结果
├── judge_summary.json                                                 # 按 benchmark/mode/judge-id 汇总
├── export/deepresearch_bench_ii/result.<judge-id>.jsonl               # 官方 aggregate_scores.py 的输入格式
├── export/drbench/scores.<judge-id>.jsonl                             # 每题的官方 score_report 结果
└── .judge-cache/<judge-id>/                                           # 原始回复缓存（用于续跑）
```

- **judge-id** 由模型名和一段 8 位哈希组成，哈希覆盖请求参数、设置、指标和 prompt 的 sha256。任何可能影响分数的改动都会产生新的 id，因此多个 judge 的结果可以并存。
- **单份评测结果**包含：judge 配置（模型、实际使用的参数和适配情况、prompt sha256、官方 commit、是否偏离官方）、报告的 sha256（报告变化后会自动重评）、官方结构的结果、各批次或各题的原始回复与失败原因、usage，以及 deviations。
- **任务结果 JSON** 中新增 `judgements.<judge-id>` 字段，记录状态、分数和评测文件路径。
- **`judge_summary.json`** 中每个 mode 包含：任务数、成功数、错误数、`judge_failures`、覆盖率、各指标均值（只统计成功评测的题，与官方一致），以及按语言（DRB2）或难度（DRBench）的分组和 usage。DRBench 另外提供 `insights_recall_enterprise`，即只统计企业文件 insight 的召回率，这是 bcg 自定义的参考指标，不属于官方指标。

### 如何验证对齐

```bash
uv run pytest tests/test_benchmark_judges.py                     # 无官方仓库时，跳过 4 项对拍测试

BCG_DRB2_REPO=~/code/DeepResearch-Bench-II BCG_DRBENCH_REPO=~/code/drbench \
  uv run pytest tests/test_benchmark_judges.py                   # 13 项全部运行

# 用官方汇总脚本交叉核对
python ~/code/DeepResearch-Bench-II/aggregate_scores.py \
  --input results/drb2/export/deepresearch_bench_ii/result.<judge-id>.jsonl \
  --tasks-file datasets/deepresearch_bench_ii/tasks_and_rubrics.jsonl --output-prefix agg
```

对拍测试的内容：

- 5 个 prompt 与官方源码逐字节一致；
- DRB2 的解析、校验、汇总函数，以及 DRBench 的 JSON 提取和质量解析函数，在同一组边界输入上与官方函数结果完全相同（官方函数通过 AST 从源码中直接载入）；
- 导出的结果经官方 `aggregate_scores.py` 汇总后，与 `judge_summary.json` 的数字一致。

### 代码变更

**新增**

- `bcg/apps/benchmark/judges/`：`client.py`、`drb2.py`、`drbench.py`、`official_sources.py`，以及 `prompts/` 目录；
- `bcg/apps/benchmark/judging.py`；
- `scripts/sync_judge_prompts.py`；
- `tests/test_benchmark_judges.py`。

**修改**

- `cli.py`：新增 `judge` 命令，`run` 新增 `--judge` 等选项；
- `drb2.py`、`drbench.py`：reference 中保留官方评测所需的原始字段（`official_content` 和 `task_config`），这些字段只供 judge 使用，不会进入 Agent 的工作区或 prompt；
- `report_scoring.py`：移除上一轮的占位类，改为指向真正的 judge 实现。
- `.pre-commit-config.yaml`：`end-of-file-fixer` 和 `trailing-whitespace` 两个钩子排除 `judges/prompts/*.txt`。官方 prompt 原文中含有行尾空格，`insight_scoring.txt` 结尾也没有换行，这两个钩子会修改它们，破坏逐字节一致。同理，应用补丁时**不要**使用 `git apply --whitespace=fix`。

---

## [Unreleased] 构图目录收集到运行目录（`<output-dir>/graphs/`）

### 背景

Graph 服务是常驻的共享进程，所有会话都写到 `~/.bcg/graphs/<Agent 会话ID>_<随机uuid>/`。目录名看不出属于哪个数据集、模式或题目，也不在 benchmark 结果旁边，后续可视化时很难定位。

### 变化

- **`bcg benchmark run` 自动收集。** bcg 模式的每道题结束后，runner 从 Agent 轨迹第一行的会话头中读取会话 ID，找到唯一对应的 `<会话ID>_*` 目录。等 Graph 服务 release 该会话之后（即它从 `/health` 的 `all` 列表中消失，最多等待 60 秒），再把目录**移动**到：

  ```text
  <output-dir>/graphs/<benchmark>/<mode>/<task_id>/
  ```

  同时写入 `benchmark_link.json`，记录对应的 `task_result`、`model_io`、`trajectory`、`graph_contexts`、`report` 文件（相对路径，整个运行目录移动后仍然有效），以及会话 ID、原始目录名和收集方式。
  单题结果 JSON 新增 `graph_dir`、`graph_collection`、`graph_collection_note` 三个字段。
  如果超时后服务端仍持有该会话，则改为复制，原目录保留（状态为 `copied_active`）。
- **新命令 `bcg benchmark collect-graphs RUN_DIR`。** 为改动之前跑出的结果补收图目录。
- **不需要修改 Agent 和 Graph 服务，也不需要重启正在运行的 Graph 服务。** 目录的定位只依赖 Agent 原本就会输出的会话头。

### 参数

| 命令 / 参数 | 示例 | 说明 |
|---|---|---|
| `run --collect-graphs/--no-collect-graphs` | `--no-collect-graphs` | 是否收集（默认开启）。只对 bcg 模式生效。 |
| `run --graph-store PATH` | `--graph-store /data/bcg-graphs` | Graph 服务的写入目录。默认是 `~/.bcg/graphs`（`BCG_HOME` 会改变它）；只有自行启动的服务用了别的 `--output-dir` 时才需要指定。 |
| `collect-graphs RUN_DIR` | `collect-graphs results/drb2` | 遍历 `RUN_DIR/*/*/tasks/*.json` 中的 bcg 模式结果，按轨迹会话 ID 找到图目录并移动过来。 |
| `collect-graphs --copy` | `--copy` | 复制而不是移动，`~/.bcg/graphs` 保持不变。 |
| `collect-graphs --force` | `--force` | 已经收集过的题目也重新收集。 |
| `collect-graphs --graph-store PATH` | | 同上。 |

收集状态：`moved`（已移动）、`copied`（已复制）、`copied_active`（服务端仍持有会话，已复制）、`already_collected`（已收集过，跳过）、`not_found`（存储中没有对应目录）、`no_session_id`（轨迹中没有会话头）、`no_trajectory`（找不到轨迹文件）、`error`（文件操作失败）。

### 使用示例

```bash
# 新运行：自动收集
uv run bcg benchmark run drb2 --task-ids 2 --modes bcg --output-dir results/drb2
#   ...
#   Artifacts: results/drb2
#   Graphs:    results/drb2/graphs

# 已有运行：补收
uv run bcg benchmark collect-graphs results/drb2
uv run bcg benchmark collect-graphs results/drb2 --copy    # 保留 ~/.bcg/graphs 中的原目录

# 可视化（bcg_viewer 新增 --run-dir）
python3 ../bcg_viewer/serve_viewer.py --run-dir results/drb2 --port 8123
```

### 代码变更

- 新增 `bcg/apps/benchmark/graphs.py` 和 `tests/test_benchmark_graphs.py`。
- 修改 `bcg/apps/benchmark/runner.py`（`RunConfig` 新增 `collect_graphs`、`graph_store_dir`、`graph_release_timeout`；`_run_one` 在任务结束后执行收集）。
- 修改 `bcg/apps/benchmark/cli.py`（新增上述参数和 `collect-graphs` 命令；运行结束时打印 `Graphs:` 路径）。

---

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

#### 5）扩大规模

```bash
bcg benchmark run drbench --drbench-subset minival --modes default,bcg --workers 4
bcg benchmark run drb2 --drb2-language en --max-problems 10 --seed 7 --web-cache results/web-cache
```

#### 6）开发者测试

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
