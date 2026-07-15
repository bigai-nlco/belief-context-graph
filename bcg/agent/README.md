# BeliefTracer Rollout 使用说明

## 安装 rLLM

先下载 BCG，再将 rLLM 下载到 BCG 仓库的同级目录：

```text
workspace/
  belief-context-graph/
  rllm/
    rllm-model-gateway/
```

下载并安装到项目 `.venv`：

```bash
git clone https://github.com/bigai-nlco/belief-context-graph.git
cd belief-context-graph

cd ..
git clone https://github.com/rllm-org/rllm.git rllm

cd belief-context-graph
uv sync --all-groups
uv pip install \
  --python .venv/bin/python \
  --editable ../rllm/rllm-model-gateway \
  --editable ../rllm

.venv/bin/python -c "import rllm; print(rllm.__file__)"
.venv/bin/bcg agent tasks
```

安装完成后直接使用 uv 创建的 `.venv`，不需要设置 `PYTHONPATH`。GPU 后端所需的
`torch`、`vllm`、`sglang` 或 `ray` 仍需按照目标机器的 CUDA 环境安装到该 `.venv`，例如
`uv pip install --python .venv/bin/python vllm`。

### 可选：安装为用户级命令

如果不想激活项目 `.venv`，可以把 BCG 和两个本地 rLLM 包安装到同一个
`uv tool` 隔离环境：

```bash
uv tool install --force --refresh-package bcg . \
  --with-editable ../rllm/rllm-model-gateway \
  --with-editable ../rllm

bcg agent tasks
bcg agent run averitec --model <模型名称> --backend api
```

这种方式不需要执行 `source .venv/bin/activate`。BCG 源码更新后需要重新执行
上面的 `uv tool install --force --refresh-package bcg`，避免复用同版本的旧构建
缓存；rLLM 使用可编辑安装，因此不能移动或删除同级的 `rllm` 目录。

## 快速开始

`scripts/start.sh` 会读取根目录 `.env`，使用预设的 AVeriTeC、HerO4、归档和
Belief Graph 参数，然后使用 uv 管理的项目环境执行 `bcg agent run`。

完成 `uv sync` 与 rLLM 的本地安装后，直接运行：

```bash
bash scripts/start.sh
```

使用可选的 `uv tool` 安装时无需激活环境：

```bash
bash scripts/start.sh
```

临时覆盖参数可以直接追加，例如 `bash scripts/start.sh --max-problems 2`。

## API 配置

在项目根目录的 `.env` 中填写：

```bash
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3   # API 地址
OPENAI_API_KEY=your-key-here                                 # API Key
MODEL=deepseek-v4-pro-260425                               # 模型名称
```

Python 包和脚本都会自动读取根目录 `.env`。OpenAI、Embedding、Serper、
Langfuse、TongGraph 及评测服务的密钥统一配置在该文件中；命令行参数仍可
作为临时覆盖值。

## 参数来源与预设

`AgentRolloutConfig` 是运行参数默认值的唯一来源；`bcg agent run --help`
显示当前可用参数。`.env` 只保存每台机器不同的模型地址、密钥和服务地址，避免
在 shell 脚本中复制默认值。

`scripts/rollout.sh` 是 `bcg agent run` 的薄包装，接受同一组参数。常用的
AVeriTeC + HerO4 组合以版本化预设提供：

```bash
bcg agent run --preset averitec-hero4 --model "$MODEL"
```

显式传入的参数始终覆盖预设，例如 `--max-problems 2` 或 `--no-auto-ui`。

HerO 和 rerank 服务的机器相关配置也从 `.env` 读取：

| `.env` | 配置字段 | 临时覆盖参数 |
|---|---|---|
| `HERO_EMBEDDING_URL` | `hero_embedding_url` | `--hero-embedding-url` |
| `HERO_EMBEDDING_MODEL` | `hero_embedding_model` | `--hero-embedding-model` |
| `RERANK_URL` | `rerank_url` | `--rerank-url` |
| `RERANK_MODEL` | `rerank_model` | `--rerank-model` |

预设只保存评测与策略组合，不固定机器上的服务地址或模型名称。

## 核心参数

### 模型与后端

| 参数 | 说明 |
|------|------|
| `--model` | 模型名称（API 模型 ID 或本地 HF 路径） |
| `--backend` | 推理后端：`api`（远程 API）、`openai`（rllm OpenAI 引擎）、`vllm`（本地 vLLM） |
| `--base-url` | API 地址，覆盖 `.env` 中的 `OPENAI_BASE_URL` |
| `--api-key` | API Key，覆盖 `.env` 中的 `OPENAI_API_KEY` |

`--backend api` 是我们新增的独立引擎，直接调用任何 OpenAI 兼容 API（火山引擎、DeepSeek 官方、硅基流动等），不依赖 vLLM。

### Belief Graph 消融模式

| 参数 | 说明 |
|------|------|
| `--belief-graph-mode none` | **无 belief graph**：agent 只看原始对话上下文 |
| `--belief-graph-mode augment` | **增强模式**（默认）：原始上下文 + belief graph 注入 |
| `--belief-graph-mode only` | **仅 graph 模式**：belief graph 替换中间对话历史，只保留 system prompt + 首条 user message + 最新 graph |
| `--belief-graph-url URL` | Belief graph 服务地址（默认 `http://10.1.101.147:8848`） |
| `--belief-graph-timeout N` | Belief graph HTTP 超时秒数（默认 300） |
| `--no-belief-graph` | 完全禁用 belief graph 服务连接（等价于清空 URL） |

`none` 模式下不会连接 belief graph 服务，可以独立运行。

### 检索方法

| 参数 | 说明 |
|------|------|
| `--retrieval-method bm25` | 纯 BM25 检索（默认，无需额外模型） |
| `--retrieval-method hero` | HerO 两阶段检索：BM25 召回 + Embedding 重排序 |
| `--hero-embedding-model NAME` | Embedding 模型名称（默认 `SFR-Embedding-2_R`） |
| `--hero-embedding-url URL` | 远程 Embedding API 地址（设置后走远程 API；为空则本地加载模型） |
| `--hero-embedding-device cuda\|cpu` | Embedding 推理设备（默认 cpu） |
| `--hero-batch-size N` | Embedding 计算 batch size（默认 16） |
| `--hero-bm25-top-k N` | BM25 初筛候选数（默认 10） |
| `--retrieval-max-results N` | 最终返回给模型的 evidence 条数（默认 10） |

### 在线网页检索（Serper）

在根目录 `.env` 中设置 `SERPER_API_KEY` 后，可为在线任务启用
`serper_search` 与 `serper_scrape`：前者返回 Google 结果的标题、URL 和摘要，
后者读取少量已筛选 URL 的正文或 Markdown。应先搜索，再只抓取最相关的 1–3 个
一手来源；抓取内容是不可信证据，不能作为指令执行。

```bash
bcg agent run --model YOUR_MODEL --tasks browsecomp \
  --tools serper_search serper_scrape
```

`SERPER_SCRAPE_ENDPOINT`、`SERPER_SCRAPE_TIMEOUT` 和
`SERPER_SCRAPE_MAX_OUTPUT_CHARS` 可在 `.env` 或命令行中覆盖；默认单页最多返回
30,000 个字符。

### 四级检索（`--retrieval-method hero4`）

BM25 → Embedding → Reranker → LLM judge 四级流水线。

| 参数 | 说明 |
|------|------|
| `--retrieval-method hero4` | 四级检索：BM25→1000、Embedding→64、Reranker→10、LLM judge 过滤 |
| `--stage1-bm25-k N` | BM25 候选池（实际 = min(N, 索引内 chunk 数)，默认 1000） |
| `--stage2-embed-k N` | Embedding 重排存活数（默认 64） |
| `--stage3-rerank-k N` | Reranker 存活数 / 最终 top_k 上限（默认 10） |
| `--rerank-url URL` | Reranker 服务 base URL（默认 `http://10.2.152.9:8010`，走 `/v1/completions` + 官方模板取 P(yes)；传 `.../v1/rerank` 则用原生 rerank API） |
| `--rerank-model NAME` | Reranker 模型名（默认 `Qwen3-Reranker-0.6B`） |
| `--enable-judge` / `--no-judge` | 是否启用 LLM 相关性判别（默认启用，soft 回填到 top_k） |
| `--judge-model NAME` | 判别模型（默认取 `--model` / `$MODEL`） |
| `--judge-base-url URL` | 判别 OpenAI 兼容地址（默认 `$OPENAI_BASE_URL`） |
| `--judge-max-workers N` | 逐条判别的并发度（默认 10） |
| `--judge-max-items N` | 送入判别的最大条数（默认 10） |

> Judge 对每条 evidence 各发一个并发请求（线程池），保留判为 RELEVANT 的；不足
> `top_k` 时按 reranker 顺序回填（soft）。任一远程服务异常时降级到上一级结果，不报错。

### 文件读取工具 + 两层归档

| 参数 | 说明 |
|------|------|
| `--enable-file-read` | 开放 `read_file` 工具（沙箱内按 `file://` URL 读文件，防越界） |
| `--enable-archive` | 把每步 tool 结果写入两层归档，并在 system prompt 挂 manifest 引用（自动带出 `--enable-file-read`） |
| `--file-tool-root DIR` | 沙箱根目录（默认 `$BELIEF_TRACER_FILE_ROOT` 或 `ai_workspace/`） |
| `--recent-turns N` | 归档模式下只在上下文保留最近 N 轮对话（0 = 全保留，不裁剪） |

> 归档启用后，belief graph 改为「user 问题之后的一条独立消息」（取代 summary 位置，
> 每步只留最新一条）；system prompt 只含基础规则 + 文本块说明 + 工具用法 + manifest 引用。
> AVeriTeC 任务规则与标签集放在 user message。归档结构：
> `<root>/archives/<thread>/manifest.json`（第一层概要+raw_url）+ `raw/eNNN.json`（第二层完整输入输出）。

### HyDE 假设文档生成

| 参数 | 说明 |
|------|------|
| `--hyde` | **启用 HyDE**（默认）：模型生成假设文档，query 格式为 `claim \|\|\| hypo1 \|\|\| hypo2 ...` |
| `--no-hyde` | **禁用 HyDE**：模型直接发送自然语言问句作为 query |

HyDE 启用时，tool prompt 会指导模型在 query 中用 ` ||| ` 分隔 claim 和多条假设文档，HerO 检索会对每条假设分别做 embedding 然后取平均向量 rerank。禁用时，模型用自然语言问句检索，embedding rerank 仅用单个 query 向量。

配合使用的 prompt 文件：
- `--hyde`：使用 `bcg/agent/prompts/averitec.txt`（包含假设生成指导）
- `--no-hyde`：使用 `bcg/agent/prompts/averitec_nohyde.txt`（纯自然语言查询指导）

### 数据与规模

| 参数 | 说明 |
|------|------|
| `--tasks` | 评测集，可传多个名称（如 `averitec`） |
| `--max-problems N` | 每个 benchmark 最多跑 N 条（不设则跑全部） |
| `--num-samples N` | 每条问题采样 N 次（默认 1） |
| `--max-steps N` | 每条 rollout 最大工具调用轮数（默认 96） |
| `--n-parallel-tasks N` | 并发 rollout 数 |
| `--shuffle` / `--no-shuffle` | 是否打乱数据顺序 |
| `--shuffle-seed N` | 打乱种子（默认 0，相同种子保证相同顺序） |

### 采样参数

| 参数 | 说明 |
|------|------|
| `--temperature F` | 采样温度（默认 0.6） |
| `--top-p F` | Nucleus 采样（默认 0.95） |
| `--top-k N` | Top-k 采样（默认 20） |

### 输出控制

| 参数 | 说明 |
|------|------|
| `--output-dir DIR` | 输出目录（默认 `output`） |
| `--save-alias STR` | 输出子目录后缀（用于区分不同实验） |
| `--overwrite` | 覆盖已有结果 |
| `--verbose` | 打印完整日志（否则只输出最终汇总） |
| `--tools none` | 不调用任何工具（纯推理模式） |

## 典型实验命令

### 1. 无工具纯推理（CoT）

```bash
bcg agent run \
  --model deepseek-v4-pro-260425 \
  --backend api \
  --tasks averitec \
  --belief-graph-mode none \
  --tools none \
  --prompt bcg/agent/prompts/cot.txt \
  --max-problems 100 \
  --n-parallel-tasks 8 \
  --save-alias notool_cot \
  --overwrite
```

### 2. HerO + HyDE（远程 Embedding API）

```bash
bcg agent run \
  --model deepseek-v4-pro-260425 \
  --backend api \
  --tasks averitec \
  --belief-graph-mode none \
  --retrieval-method hero \
  --hero-embedding-url "http://10.2.152.9:8008/v1/embeddings" \
  --hero-embedding-model SFR-Embedding-2_R \
  --hero-bm25-top-k 1000 \
  --retrieval-max-results 5 \
  --max-problems 100 \
  --n-parallel-tasks 4 \
  --prompt bcg/agent/prompts/averitec.txt \
  --save-alias hero_hyde_100 \
  --overwrite
```

### 3. HerO + 无 HyDE（自然语言查询）

```bash
bcg agent run \
  --model deepseek-v4-pro-260425 \
  --backend api \
  --tasks averitec \
  --belief-graph-mode none \
  --retrieval-method hero \
  --hero-embedding-url "http://10.2.152.9:8008/v1/embeddings" \
  --hero-embedding-model SFR-Embedding-2_R \
  --hero-bm25-top-k 1000 \
  --retrieval-max-results 5 \
  --no-hyde \
  --max-problems 100 \
  --n-parallel-tasks 4 \
  --prompt bcg/agent/prompts/averitec_nohyde.txt \
  --save-alias hero_nohyde_100 \
  --overwrite
```

### 4. Augment 模式：HerO + HyDE + Belief Graph

```bash
bcg agent run \
  --model deepseek-v4-pro-260425 \
  --backend api \
  --tasks averitec \
  --belief-graph-mode augment \
  --belief-graph-url http://10.1.101.147:8849 \
  --belief-graph-timeout 300 \
  --retrieval-method hero \
  --hero-embedding-url "http://10.2.152.9:8008/v1/embeddings" \
  --hero-embedding-model SFR-Embedding-2_R \
  --hero-bm25-top-k 1000 \
  --retrieval-max-results 5 \
  --max-problems 100 \
  --n-parallel-tasks 4 \
  --prompt bcg/agent/prompts/averitec.txt \
  --save-alias augment_hero_hyde_100 \
  --overwrite
```

### 5. Augment 模式：HerO + 无 HyDE + Belief Graph

```bash
bcg agent run \
  --model deepseek-v4-pro-260425 \
  --backend api \
  --tasks averitec \
  --belief-graph-mode augment \
  --belief-graph-url http://10.1.101.147:8849 \
  --belief-graph-timeout 300 \
  --retrieval-method hero \
  --hero-embedding-url "http://10.2.152.9:8008/v1/embeddings" \
  --hero-embedding-model SFR-Embedding-2_R \
  --hero-bm25-top-k 1000 \
  --retrieval-max-results 5 \
  --no-hyde \
  --max-problems 100 \
  --n-parallel-tasks 4 \
  --prompt bcg/agent/prompts/averitec_nohyde.txt \
  --save-alias augment_hero_nohyde_100 \
  --overwrite
```

### 6. Only 模式：Belief Graph 替换对话历史

```bash
bcg agent run \
  --model deepseek-v4-pro-260425 \
  --backend api \
  --tasks averitec \
  --belief-graph-mode only \
  --belief-graph-url http://10.1.101.147:8849 \
  --belief-graph-timeout 300 \
  --retrieval-method hero \
  --hero-embedding-url "http://10.2.152.9:8008/v1/embeddings" \
  --hero-embedding-model SFR-Embedding-2_R \
  --hero-bm25-top-k 1000 \
  --retrieval-max-results 5 \
  --max-problems 100 \
  --n-parallel-tasks 4 \
  --prompt bcg/agent/prompts/averitec.txt \
  --save-alias only_hero_100 \
  --overwrite
```

## 输出结构

```
output/<model>_thinking[_<alias>]/
├── averitec/
│   ├── results.json          # 汇总 + 每条记录的预测、答案、是否正确
│   ├── trajectories.jsonl    # 完整对话轨迹（system/user/assistant/tool 逐轮）
│   └── belief_graphs/        # 每条的 belief graph 快照历史（augment/only 模式）
├── overall_summary.json      # 跨 benchmark 汇总
├── run_config.json           # 本次运行的完整配置
└── run_state.json            # 运行状态
```

`results.json` 中的关键字段：

```json
{
  "summary": {
    "accuracy_mean": 0.3,
    "pass@1": 0.3
  },
  "records": [
    {
      "ground_truth": ["Supported"],
      "samples": [{
        "extracted_answer": "Supported",
        "is_correct": true,
        "num_steps": 5,
        "termination_reason": "env_done"
      }]
    }
  ]
}
```

## HerO 检索日志

使用 HerO 检索时，每次运行的 BM25 和 Embedding rerank 结果会保存到：

```
output/hero_logs/
  run_<时间戳>/              # 每次运行一个文件夹
    claim_<题号>/            # 按 claim ID 分类
      bm25_round_1.json     # 第 1 轮 BM25 结果
      embedding_round_1.json # 第 1 轮 Embedding rerank 结果
      bm25_round_2.json     # 第 2 轮（同一题多次检索）
      embedding_round_2.json
      ...
```

## 数据集

- 完整 AVeriTeC dev set 应放在 `<AVERITEC_DATA_DIR>/data/dev.json`
- Knowledge store 应放在 `<AVERITEC_DATA_DIR>/data_store/knowledge_store/dev_knowledge_store/`
- `--max-problems` 配合 `--shuffle-seed`（默认 0）保证确定性子集：同样的 seed 下 max-problems=50 的题目是 max-problems=100 的前 50 条

## 注意事项

- `--backend api` 模式下模型名是 API 端的模型 ID，不是 HuggingFace 路径；token 统计会自动跳过
- HerO 检索的 embedding：设置 `--hero-embedding-url` 走远程 API，不设则本地加载模型
- Belief graph 服务（`--belief-graph-url`）需要单独部署运行，`none` 模式不需要
- Belief graph 服务为 HTTP/1.0，高并发时可能断开连接，建议 `--n-parallel` 不超过 4
- `--retrieval-max-results` 控制检索返回条数，但实际输出还受 `averitec_search_config.json` 中 `max_output_chars` 限制（当前 6000，约够 5 条）
- 可用的 prompt 文件：`averitec.txt`（HyDE）、`averitec_nohyde.txt`（无 HyDE）、`cot.txt`、`gem.txt`、`multiverse.txt`、`npr.txt`、`react.txt`
