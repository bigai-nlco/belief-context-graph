# BCG AI Review

对一份已完成的 Agent 运行结果自动生成标注。关注 **Agent 对实际注入 Belief Graph 的使用**，同时寻找发挥作用和错失机会的关键过程。不是抽点/建边质量评测，也不是 Benchmark 官方评分。

本模块独立于网页工作台，能够在命令行运行，不启动 Agent 或 Graph Server，不覆盖原始结果和人工标注。支持本仓库 native ReAct 的 `t=llm` / `t=result` JSONL 轨迹，以及包含这些记录的单题 `bcg-case-bundle`。其他轨迹格式应先提供适配器，不能直接按行号推断轮次。

## 配置与运行

复用项目环境中的 `httpx`、`pydantic`、`python-dotenv`。在 `~/.bcg/.env` 中配置：

```dotenv
LITELLM_OPENAI_API_KEY=your-key
# 可选，未配置时使用本项目的 LiteLLM 入口：
LITELLM_OPENAI_BASE_URL=https://litellm.mybigai.ac.cn/
```

Review 固定使用 `gpt-6.1-sol`，请求参数 `reasoning_effort=medium`。使用 Chat Completions 的非流式 JSON 输出，不继承 Agent 的 `OPENAI_API_KEY`、模型或 OpenRouter 路由。记录的是请求的思考强度，无法仅凭请求参数证明上游实际执行的内部思考预算。

先查看整理后的材料和 Prompt，不产生 API 消耗：

```sh
uv run python -m tools.ai_review \
  --input results/your-run/drb2/trials/2 \
  --output results/ai-reviews/your-run-2 \
  --dry-run
```

检查后去掉 `--dry-run` 执行；也可以直接执行。`--input` 可传单题目录、该目录的 `result.json` 或单题 `case-study.json.gz`。`--env-file PATH` 可指定配置文件。

```sh
uv run python -m tools.ai_review \
  --input results/your-run/drb2/trials/2 \
  --output results/ai-reviews/your-run-2
```

同一输入、Prompt 版本和配置在同一输出目录已有完整结果时，直接返回，不重复计费。失败或进程中断后，显式加 `--retry` 续跑，复用校验通过的抽取批次。证据或配置变化必须使用新输出目录。

## 流程与 Prompt

完整模板在 [`prompts.py`](prompts.py)，结构化输出在 [`schema.py`](schema.py)。

1. 从任务目录或经过哈希校验的运行包读取任务、最终报告、Assistant 输出、工具结果和逐轮注入记录。只按同一 Agent 的实际 `call`/`turn` 匹配，不用最终全图倒推历史可见内容。
2. 默认每批最多 6 个调用、目标 60,000 字符的逐轮材料。单轮不会拆断，因此目标字符数是软上限。任务与最终报告另附在每批中。过长字段会显式标记截断，覆盖情况写入结果；不声称读过被截断内容。
3. 模型寻找关键过程，给出具体 Graph 原文、当轮 Assistant 原文，以及可选的最终报告原文。涉及搜索时必须说明具体信息需求、Graph 是否已满足、核验目的、预期增量和实际结果。代码检查同轮原文引用、分类前提和判断一致性。
4. 汇总模型先归并跨批次的同类问题，再按影响和证据强度选择最多 6 个代表过程。每个过程只展开一次，可关联多个维度。同组其他轮次保留各自的实际注入和输出。合理核验与无效重复、正向与负向例子不能合并。维度只记录等级、关联过程和简短补充；总结只提炼最重要的 1–2 项优点、问题及改进，不复述全部事件。
5. 在同一次汇总请求中，对照任务要求和最终报告评估任务结果，输出达标、部分达标、未达标、未完成或尚未评判，并给出判断依据、任务与报告原文引用及具体限制。代码核验引用确实来自对应原文。证据不足时允许尚未评判；材料被截断时不能判定全部达标。
6. 保存 AI 草稿。任务结果、依据和原文会出现在 `review.json`、`review.md` 以及工作台基本信息中。这是 AI 对任务完成情况的评估，不是 Benchmark 官方评分，也不代表已独立核查报告中的全部事实。

此项评估使用原有 LiteLLM `gpt-6.1-sol`、`reasoning_effort=medium`，仍是分批分析加一次汇总，不额外增加模型阶段。Prompt 版本为 `bcg-agent-use-review-v3`。旧版已完成标注保持原样，不因升级自动重跑或计费。同一版本失败后重试复用已验证批次；旧版失败任务显式重试时，旧材料归档到 `previous-versions/`，按新规则重新分析，不复用未经新规则核验的判断。旧调用用量仍保留在累计记录中。

### 区分核验与无效搜索

| 判断 | 依据 |
|---|---|
| 合理核验 | 针对具体缺口获取原文细节、独立来源、最新状态、适用范围或冲突解释，检索动作与缺口对应。获取失败不自动变为无效重复。 |
| 新增信息搜索 | 处理 Graph 尚未覆盖的不同命题或子问题，不能仅因主题接近而判重复。 |
| 无效重复 | Graph 已充分覆盖同一命题及必要范围，没有具体核验缺口，完整可见的结果也没有新增证据。三个条件同时成立。 |
| 无效工具重试 | 请求前已知预算耗尽或同一通道不可用，仍以相同方式调用。必须引用调用前已知的状态，不用本轮首次失败倒推。 |
| 证据不足 | 目的、范围或结果不可判定，不能当成确定的问题。 |

同一 query 可以服务不同的核验目的；不同 query 也可能在重复找同一事实。判断依据信息需求与可见证据，而非文字相似度。代码检查引用、分类前提及汇总是否自相矛盾，语义判断仍由模型完成，不能保证每次判断都正确。

### 阅读与去重

- **关键过程**：一次记录完整证据与行为分析；搜索判断和同类轮次按需展开。
- **维度评价**：结论加关键过程链接，补充说明默认折叠，不复制整段事件。
- **归因总结**：优先问题、可保留的有效行为、1–2 项改进与验证方式。通用归因限制集中写一次。

所有原始候选保存在 `candidates.json`，归并关系保存在 `summary.json` 的 `process_groups`。代码拒绝重复分组、混合相反判断、悬空引用以及直接复制长段过程分析的维度备注/总结。人工标注的字段与编辑方式不变。

分析中区分无效重复搜索与交叉验证，区分直接引用 Graph 与行为一致。提供相邻两次调用用于检查其他信息来源，但这不是完整模型输入的重建。隐藏 Thinking、模型内部因果关系、单次运行的 Token 节省都不由 Review 推定。没有同题对照不评价 A7；单 Agent 不评价 A8。

`--chunk-turns`、`--chunk-chars`、`--max-batches` 控制分批规模。超过批次数量上限会在任何 API 请求前停止。结构或引用校验失败最多修正一次；网络重试最多两次，仅限瞬时网络错误、429、502、503、504。失败不会作为成功标注发布。

## 输出

- `review.json`：可供系统读取的 AI 标注、定位信息、覆盖范围、模型配置与 Review 用量。
- `review.md`：可直接阅读的分析报告，包含 Graph 注入原文、当轮输出和对应的最终报告节选。
- `manifest.json`：状态、证据指纹、Prompt 版本及耗时。`completed` 才表示整个流程完成。
- `input-*.json`、`findings-*.json`、`summary-input.json`、`summary.json`：分步材料和结果。
- `calls.jsonl`：实际请求内容、可见模型输出、usage 与重试记录，不记录认证头。文件包含运行证据，应放在本地结果目录，不提交 Git。

`input_tokens` 包含 Cache，`output_tokens` 包含 reasoning，`reasoning_tokens` 是输出的子集；`total_tokens = input_tokens + output_tokens`。Review 的用量独立于 Agent/Graph。只有接口明确返回费用时才累加 `reported_cost`，`cost_complete=false` 表示费用不完整，不能把零当成免费。

每个事件带有 `sourceLocator = {path, digest, index, call}`，供工作台按证据文件哈希和原始行号重新绑定引用。不要把浏览器随机生成的记录 ID 当成跨导入的稳定标识。

## 测试

```sh
uv run pytest -q tests/test_ai_review.py tests/test_ai_review_rules.py
uv run ruff check tools/ai_review tests/test_ai_review.py tests/test_ai_review_rules.py
```

测试使用本地模拟响应，覆盖跨轮匹配、错误引用、缺失注入、思考强度与独立 Key、用量统计、失败续跑、完成结果复用和身份信息不进入日志。
