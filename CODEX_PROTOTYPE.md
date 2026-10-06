# Codex BCG 接入原型

当前状态：代码与离线测试已实现，**尚未接入真实 Codex，也没有调用真实模型进行验证**。等人工检查之后，再做实际接入测试。Claude Code 的 Messages API 适配不在这一版中。

## 副本与修改范围

这个目录复制自 `../belief-context-graph-gpt56-luna` 的当前工作区，包括当时未提交的源码、文档和 dashboard 修改。原目录未修改。原仓库基于 `746ad2c`，副本的完整源码快照提交为 `b646e41`。

复制范围不包括实验结果、`outputs_stream`、虚拟环境、`node_modules`、缓存、运行日志、`var` 和私有 `.env`；这些不是接入实现所需的源码。副本有独立 `.git`，不是指向原仓库的 worktree，也没有推送到 GitHub。

本次新增代码在 [`bcg/integrations/codex_proxy/`](bcg/integrations/codex_proxy/)，原有代码只在 [`bcg/apps/cli.py`](bcg/apps/cli.py) 增加了一个可选命令入口。没有改建图 Prompt、confidence propagation、Graph 模型、原有 Agent 或 Benchmark 的实现。

## 接入位置

```text
Codex：保留原本的 CLI、工具执行、文件编辑、审批和沙箱
   │ HTTP Responses API（完整 input）
   ▼
BCG 本地代理 ───────────────────► 原模型服务 /responses
   │                                │
   │ 建图 / 选取 Graph               │ JSON 或 SSE 原样转发
   ▼                                ▼
现有 BCG Graph Server             Codex
```

这不是一个试图清空 Codex 历史的 MCP tool。代理只改变**发给模型的请求**，不修改 Codex 本地的聊天记录、审批状态或工具定义。API 响应中的工具调用仍由 Codex 执行。

## 上下文策略

默认 `mode: bcg`、`recent_turns: 2`：

1. 保留原始 `instructions`，以及 input 中所有 system/developer 消息。
2. 永久保留 input 中所有 user 消息，包括最初问题和后续补充。
3. 保留最近两轮完整的 Assistant/Tool 交互，以及所有尚未完成的工具交互。
4. 初始用户问题先入图；只有将要从模型请求中淘汰的完整交互才增量入图。后续 user 消息始终原文保留，到达建图历史边界时也按时间顺序入图。
5. Graph Server 确认写入成功后，通过现有 `/context-selection` 的 `focused` 策略选图，节点字符预算默认 6600、搜索深度 4。随后才从当前模型请求中移除相应旧交互。
6. 将选出的原始 belief、confidence 和关系以 Markdown 放进 `<belief_graph>...</belief_graph>`，追加到 Responses 的 `instructions`（system/developer 指令通道）。不改写 belief、不重新计算置信度、不虚构关系。

这里“一轮”不是一条消息：

```text
一轮 = Assistant reasoning/text + 全部并行 tool calls + 对应的全部 tool outputs
发送给 Graph 时 = 一个 Assistant turn + 一个合并后的 Tool Result turn
```

没有工具调用的完整 Assistant 答复也算一轮。若协议中无法可靠区分连续 Assistant 输出，会保守地放在同一组，不冒险截断。`recent_turns: 0` 会淘汰所有已完成的组，但仍保留用户、指令和未完成调用。`mode: default` 不建图、不裁剪，只做同样的 Responses 协议转发设置。

### Thinking 和工具调用

- 最近保留轮次的 reasoning item（包括 `encrypted_content`）、assistant 的 `phase`、tool-call ID 和其余字段保持原值，不根据文本重建。
- 用于建图时：优先使用可见的 `reasoning_text`，否则使用 `summary`；两者都没有就不生成 `<thinking>` 内容。加密 reasoning 不解密、不写入 Graph，也不会记录到代理审计日志。
- Function tool calls 使用原始名称和解析后的完整 arguments，包装成 `<tool_call>`；custom tool calls 的原始文本作为 `arguments.input`，不丢失 patch 等内容。
- 多个 Tool Result 保留各自 `call_id`，合成一个含多个 `<tool_result>` 块的 Graph turn，继续使用原有代码做配对、抽点和建边。
- 默认在请求已有 `reasoning` 配置时补 `summary: auto`，不改变 `effort`。不支持 summary 的上游可设 `request_reasoning_summary: false`。

**窗口限制针对显式 input items。** 代理不知道加密 reasoning 内部包含哪些历史信息，不能承诺保留的加密 item 完全不涉及更早的内容。

### 系统上下文示意

```text
<原 Codex instructions，保持原文>

<context_blocks_guide>
Earlier completed assistant/tool steps have been omitted from this request.
...belief / confidence / relation 的含义，以及不可信记忆的使用边界...
</context_blocks_guide>

<belief_graph>
## Beliefs
- [B2] config.py defines parse_port(value) as int(value), without a port-range check. (confidence 0.95)
- [B3] test_config.py requires parse_port to reject ports outside 1..65535. (confidence 0.95)

## Relations
- [B3] supplements [B2] — The test specifies a constraint not enforced by the implementation.
</belief_graph>
```

上面是说明用的虚构例子，不是实测结果。完整指引文本在 [`graph.py`](bcg/integrations/codex_proxy/graph.py) 的 `CONTEXT_GUIDE`，也可以用下面的离线命令看到完整拼接结果。

## 现在可以检查的内容（不联网）

在本副本目录下，可暂时借用原目录的 Python 解释器执行副本代码，不需要安装全局 `bcg`：

```bash
PYTHON=../belief-context-graph-gpt56-luna/.venv/bin/python

# 确认导入的是副本，不是原目录中的代码
"$PYTHON" -c 'import bcg.integrations.codex_proxy as p; print(p.__file__)'

# 查看完整的模拟请求转换，不启动任何服务
"$PYTHON" -m bcg.integrations.codex_proxy preview \
  --request tests/fixtures/codex_proxy/request.json \
  --snapshot tests/fixtures/codex_proxy/snapshot.json \
  --selection tests/fixtures/codex_proxy/selection.json

# 运行离线测试：模型与 Graph HTTP 请求均使用 MockTransport
# HTTP 转发测试临时监听 127.0.0.1 随机端口，测试结束即关闭
"$PYTHON" -m pytest -q tests/test_codex_proxy.py

# 查看相对复制快照的实现改动
git diff b646e41
```

`preview` 使用明确标注的虚构 graph fixture，不代表真实 Graph Server 会抽出同样的节点。它用于检查窗口边界、输入输出结构、注入位置和内容。

本次验证：新增代理测试与原有 CLI、HTTP contract、context selection、config、refactor contract 回归共 **130 项通过**，Ruff 检查及格式检查通过。使用的是本副本导入路径；没有启动真实 Codex，没有调用真实 Graph 或付费模型，测试临时服务均已关闭。

## 检查通过之后才进行的真实接入

以下仅为后续接入步骤，不会自动执行，也没有修改 `~/.codex/config.toml`。

1. 准备一个现有 BCG Graph Server。其 Graph 模型、key 和 backend 仍由原有 Graph 配置管理；代理不替你选择或启动模型。
2. 修改 [`examples/codex-proxy.yaml`](examples/codex-proxy.yaml) 的 `upstream_base_url` 和 `graph_base_url`。前者必须支持 `/responses`。
3. 在代理进程环境中设置 `BCG_CODEX_UPSTREAM_API_KEY`（真实模型 key）和 `BCG_CODEX_GATEWAY_TOKEN`（单独生成的本地代理口令）。不要把真实 key 填入受 Git 跟踪的配置文件。
4. 配置文件通过 `upstream_api_key_env` 指定要读的环境变量，也可以改为你已有的 `OPENROUTER_OPENAI_API_KEY` 等名字。代理模块本身不会自动读取 `~/.bcg/.env`；请显式 export。Graph 服务用自己的环境，不会收到 Agent 的 key。
5. 之后才运行：

```bash
python -m bcg.integrations.codex_proxy serve --config examples/codex-proxy.yaml
```

如已把副本安装到独立环境，同一入口也可写为 `bcg codex-proxy serve ...`。此处没有安装或覆盖系统中的 `bcg` 命令。

打印供检查的 Codex TOML 片段：

```bash
python -m bcg.integrations.codex_proxy codex-config --model YOUR_MODEL_ID
```

该命令**只输出文本**。片段使用独立 provider/profile，关闭该 provider 的 WebSocket 和请求压缩，不更改审批或沙箱。未来手动合并配置时，Codex 进程需要同一个本地 `BCG_CODEX_GATEWAY_TOKEN`，以及每个独立会话不同的 `BCG_CODEX_SESSION_ID`。恢复同一会话可使用同一 ID；切换任务、分支会话或并行 Agent 应使用不同 ID。

第一轮真实验证应限定为一个新建的短 Codex 会话，不启用 subagents，确认工具完整配对、没有 fallback、Graph 写入内容正确，再扩大范围。当前安装版本、模型服务及其认证方式的真实兼容性仍待该阶段验证；本原型只支持 API key，不转发 ChatGPT OAuth 登录凭据。

## 失败处理和边界

| 情形 | 第一版行为 |
| --- | --- |
| Graph 写入失败或超时 | 本次保留完整原始 input。写入结果不确定时，下次创建新的 graph ID 重建前缀，避免重复写进同一图。可能增加建图费用，不会自动付费重试当前请求。 |
| Graph 选择失败、空选图或节点关系不一致 | 本次保留原始 input；已确认写入的历史不会重复写入。 |
| 同一请求被 Codex 重试 | 已确认入图内容按摘要去重；模型请求仍会正常转发，代理不宣称模型调用幂等。 |
| 历史被回退、修改或出现分支 | 从新 graph ID 重建，避免使用未来/另一分支的信息。旧 Graph 记录不自动删除。 |
| 多模态或未知 Responses item | 不裁剪、不建图，完整转发并记录 fallback；不是“支持了多模态 BCG”。 |
| `previous_response_id`、conversation、item reference、后台响应、服务端 compaction | 明确拒绝；否则上游可能把被移除的历史补回来。必须使用完整、显式的无状态 replay。 |
| Codex 的 `/responses/compact`、WebSocket、Chat Completions | 暂不支持；自动或手动 compaction 触发时需要进一步适配，不会声称成功。 |
| 同一会话的并行模型请求 | 返回 409。不同会话互相隔离，默认最多 128 个内存会话。 |
| 代理进程重启 | 从客户端下次传来的完整历史重新建图，不持久化密钥或本地会话状态。 |
| 对话最后两轮 / 最终答案 | 正常运行期间保持原文，不额外建图。尚未接入会话结束 hook，因此本版不做自动 final flush。 |

不会关闭 Codex 的审批、安全检查、工具权限或沙箱。BCG 数据仍可能包含错误或间接提示注入，分隔和 guidance 不能提供绝对安全保证。只把代理连接到你明确允许接收该项目代码/会话内容的模型及 Graph 服务。

## 记录与文件定位

- 代理只记录统计到 `var/codex-proxy/requests.jsonl`，达到约 10 MiB 滚动一次，最多约 20 MiB。包括 fallback 原因、裁剪轮数、selected node/edge IDs、耗时、Agent usage、Graph 累计 usage，不记录原始 Model I/O。
- `graph_usage_cumulative` 是该 `problem_id` 的累计用量，**不能把每条日志中的这个字段直接相加**。代理不把 Agent 和 Graph 的 token 混为一项。
- 真正的 Graph/trajectory/prompt 记录仍在 Graph Server 启动时指定的输出目录中，使用日志里的 `problem_id`（`codex_...`）对应查找。Graph Server 的记录策略不受代理日志设置控制。
- session lock 与 ledger：[`context.py`](bcg/integrations/codex_proxy/context.py)。Responses 分组与序列化：[`protocol.py`](bcg/integrations/codex_proxy/protocol.py)。HTTP/SSE：[`server.py`](bcg/integrations/codex_proxy/server.py)。离线测试：[`tests/test_codex_proxy.py`](tests/test_codex_proxy.py)。

协议依据：[Codex 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)、[Responses reasoning 与无状态 replay](https://developers.openai.com/api/docs/guides/reasoning)。文档能力不等于本原型已经通过真实 Codex 集成测试。
