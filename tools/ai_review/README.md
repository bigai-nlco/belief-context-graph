# BCG AI Review

对一份已完成的 Agent 运行结果自动生成标注。关注 **Agent 对实际注入 Belief Graph 的使用**，同时寻找发挥作用和错失机会的关键过程。不是抽点/建边质量评测，也不是 Benchmark 官方评分。

本模块独立于网页工作台，能够在命令行运行，不启动 Agent 或 Graph Server，不覆盖原始结果和人工标注。支持 BCG Deep Research Agent（`feat/bcg-deep-research-agent` 分支）native ReAct 的 `t=llm` / `t=result` JSONL 轨迹，以及包含这些记录的单题 `bcg-case-bundle`。运行 Review 只需已有的结果文件，不需要安装或启动该 Agent。其他轨迹格式应先提供适配器，不能直接按行号推断轮次。

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
3. 模型寻找关键过程，给出具体 Graph 原文、当轮 Assistant 原文，以及可选的最终报告原文。代码逐条检查引用来自正确轮次且确为连续原文。
4. 汇总模型只选择已经校验的候选过程，并将维度评价关联到这些过程，生成优点、问题、改进办法和验证方法。不会强行凑齐所有维度或正反例。
5. 保存 AI 草稿。运行结果默认“尚未评判”，需要人工核对后再决定是否采纳。

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
uv run pytest -q tests/test_ai_review.py
uv run ruff check tools/ai_review tests/test_ai_review.py
```

测试使用本地模拟响应，覆盖跨轮匹配、错误引用、缺失注入、思考强度与独立 Key、用量统计、失败续跑、完成结果复用和身份信息不进入日志。
