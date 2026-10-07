# Codex BCG 接入原型

当前状态：已完成真实 Codex 流程验证，并进一步实现了可在原生持续会话使用的三种可选接入方式：`assist`、`compact` 和 `gateway`。安装方法、Context 能力边界和切换方式见 [原生 Codex 接入说明](docs/codex-native-integration.md)。本文保留最初请求代理的实现说明；Claude Code 适配不在这一版中。

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
2. 原样保留当前 input 中所有 user 消息，包括最初问题和后续补充；无法恢复已被 Codex 自身 compaction 移除的原文。
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

最新离线验证：新增代理、live-test setup、实测记录校验与原有 CLI、HTTP contract、context selection、config、refactor contract 回归共 **157 项通过**，Ruff 检查及格式检查通过。使用的是本副本导入路径；这些离线测试没有启动真实 Codex、调用真实 Graph 或付费模型，测试临时服务均已关闭。真实验证结果另见下文。

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

第一轮真实验证应限定为一个新建的短 Codex 会话，不启用 subagents，确认工具完整配对、没有 fallback、Graph 写入内容正确，再扩大范围。除 API key 路径外，原生 Codex 登录路径见下方“Codex 登录 + 独立 OpenRouter 建图”；认证测试通过不等于完整 BCG 流程已经通过。

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

- 代理默认只记录统计到 `var/codex-proxy/requests.jsonl`，达到约 10 MiB 滚动一次，最多约 20 MiB。包括 fallback 原因、裁剪轮数、selected node/edge IDs、耗时、Agent usage、Graph 累计 usage。
- `record_model_io: true` 是显式调试选项，会将完整输入、实际转发请求和终态响应保存到 `audit_dir/model_io/`，不保存 HTTP headers；文件权限为 0600，目录总量上限约 100 MiB。这些记录可能包含代码和加密 reasoning item，不应公开上传。默认关闭。
- 可选的 `max_session_requests` 限制单会话请求数；到达限制后，在继续建图/调用模型之前返回 429。`replay_reasoning_context: true` 会在请求未指定时，根据是否已有模型输出设置 `reasoning.context` 为 `current_turn` 或 `all_turns`。
- `graph_usage_cumulative` 是该 `problem_id` 的累计用量，**不能把每条日志中的这个字段直接相加**。代理不把 Agent 和 Graph 的 token 混为一项。
- 真正的 Graph/trajectory/prompt 记录仍在 Graph Server 启动时指定的输出目录中，使用日志里的 `problem_id`（`codex_...`）对应查找。Graph Server 的记录策略不受代理日志设置控制。
- session lock 与 ledger：[`context.py`](bcg/integrations/codex_proxy/context.py)。Responses 分组与序列化：[`protocol.py`](bcg/integrations/codex_proxy/protocol.py)。HTTP/SSE：[`server.py`](bcg/integrations/codex_proxy/server.py)。离线测试：[`tests/test_codex_proxy.py`](tests/test_codex_proxy.py)。

协议依据：[Codex 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)、[Responses reasoning 与无状态 replay](https://developers.openai.com/api/docs/guides/reasoning)。文档能力不等于本原型已经通过真实 Codex 集成测试。

## BrowseComp 真实接入测试入口

**Agent 是本机 Codex，而不是 `bcg agent`。** Codex 的原生循环负责模型调用、推理、工具执行和最终答案。Python 测试脚本只负责启动/关闭服务及传入一道 BrowseComp 问题，不调用旧 Agent 的 rollout 或 benchmark runner。BCG 作为外部上下文层，在 Codex 发出的 Responses 请求进入模型前完成“旧轮次入图 → 保留最近两轮 → system/instructions 注入 Graph”。普通 Codex 使用时保留它原有的工具和权限；单题流程测试专门限制为只读 Serper 搜索，避免读取本地标准答案。

### Codex 登录 + 独立 OpenRouter 建图

```bash
PYTHONPATH=. "$PYTHON" -m scripts.codex_browsecomp_smoke \
  --provider codex --auth-only \
  --output results/codex-browsecomp-live/codex-auth-probe

PYTHONPATH=. "$PYTHON" -m scripts.codex_browsecomp_smoke \
  --provider codex \
  --dataset /path/to/browse_comp/data.json \
  --task-id browsecomp-0000 \
  --output results/codex-browsecomp-live/codex-bcg-attempt-01
```

本测试路径固定使用：Agent = `gpt-6-luna` / `low`，Graph = OpenRouter `qwen/qwen3.8-flash` / `none`。Graph key 取自 `~/.bcg/.env` 的 `OPENROUTER_OPENAI_API_KEY`；Codex 的认证来自 `~/.codex/auth.json`（可用 `--codex-auth-file` 指定），不使用 OpenRouter key。`--auth-only` 只验证 Codex 登录，不启动 Graph，也不求解 BrowseComp，不能作为 BCG 端到端成功的证据。

完整测试结束后会生成 `verification.json`：同时检查 Codex 完成事件、原始输入/保留轮次未被改写、call/result 配对、实际搜索结果、Graph 写入记录、裁剪和 system/instructions 注入以及 SSE 终态。若 Codex 正常退出但这些条件不满足（例如 Graph 失败后保留全历史），脚本退出码为 4，不报告成功。此机械检查仍不能替代答案正确性、Graph 语义和真实记录来源的人工审查。

测试在私有、隔离的 `CODEX_HOME` 中临时放置 0600 权限的认证副本，由 Codex 自己读取及刷新，退出时删除副本；不修改原始 auth.json 或 config.toml。代理不会读取刷新 token，而是在 `upstream_auth: codex_chatgpt` 下转发 Codex 已提供的 bearer 与 account header，且只允许官方 `https://chatgpt.com/backend-api/codex` 目标、不跟随重定向。独立的 `X-BCG-Gateway-Token` 验证本机访问，不能作为上游登录凭据。所有 HTTP 认证 header 都不写入模型 I/O 或 audit。原生 `/models` 检查以及 `additional_tools` 工具目录也予以保留。

代理示例配置：[`examples/codex-proxy-chatgpt.yaml`](examples/codex-proxy-chatgpt.yaml)。使用 `bcg codex-proxy codex-config --model gpt-6-luna --chatgpt` 可以打印供审查的 provider 配置片段，不自动安装到全局 Codex。

2026-10-07 模型变更：此前 Graph `openai/gpt-6-luna / none` 请求返回 403。按用户明确要求，仅将建图模型切换为 `qwen/qwen3.8-flash`，保留 `reasoning_effort: none`、unified 建图、最近两轮、focused/6600 及其余设置。

2026-10-07 真实验证：`results/codex-browsecomp-live-20261007/codex-qwen38-bcg-attempt-01/` 的单题 `browsecomp-0000` 已完成，`verification.json` 的 18 项机械检查全部通过。真实 Codex 发出 5 次模型请求、完成 13 次 Serper 搜索，写入 6 条 Graph turn，2 次请求实际裁剪旧轮次并注入 Graph，没有 fallback；Graph 的 4 次模型调用均记录为 Qwen3.8 Flash / none，返回 reasoning tokens 为 0。Agent 最终答案为 `1985–1986`，与数据集参考答案 `1988–96` 不符：这是接入流程通过，不是答题正确或性能提升的证据。

本次实测的限制：Codex Code Mode 将并行搜索包装为 `exec`，代理保留该原始名称、代码与 call ID，不推测内部 MCP 调用映射。因此当前 Graph 中这些 `exec` 工具结果使用原有通用规则摘要，尚未逐条拆成搜索结果的语义 beliefs。最后两轮和最终答案仍保留在原始上下文，不做 final flush。测试还记录了 `model_supports_reasoning_summaries` 被当前 Codex 版本忽略的非致命配置警告；实际请求中的 `summary: auto` 由代理保留。未为使答案正确而重跑本题或修改检索、抽点、建边策略。

### 原有 API key 路径

[`scripts/codex_browsecomp_smoke.py`](scripts/codex_browsecomp_smoke.py) 为人工授权之后的单题实测入口，不自动运行。它使用真实 `codex exec`、独立 `/tmp` 下的 HOME/CODEX_HOME/工作目录、独立 Graph Server 和本代理。只提供只读 Serper MCP 搜索工具，关闭 shell 与多 Agent 工具，问题答案不会进入模型 prompt。结束或出错会关闭它自己启动的服务，不动现有 tmux 服务。选用的 key 只在进程环境和认证 header 中流转，不写入配置。

例如（会产生真实 API 费用）：

```bash
PYTHONPATH=. "$PYTHON" -m scripts.codex_browsecomp_smoke \
  --provider openrouter \
  --preflight-only \
  --output results/codex-browsecomp-live/preflight

PYTHONPATH=. "$PYTHON" -m scripts.codex_browsecomp_smoke \
  --provider openrouter \
  --dataset /path/to/browse_comp/data.json \
  --task-id browsecomp-0000 \
  --output results/codex-browsecomp-live/attempt-01
```

Agent 为 GPT-5.6-Luna/low，Graph 为同型号/none，最近两轮、focused/6600。也支持 `--provider litellm`，分别读取 `--env-file`（默认 `~/.bcg/.env`）中的 `OPENROUTER_OPENAI_API_KEY` 或 `LITELLM_OPENAI_API_KEY`。两种路由都需要 `SERPER_API_KEY`，并使用该文件的 `SERPER_ENDPOINT`。单题最多 40 个模型请求、30 次搜索，默认运行上限 1800 秒；不会以正确率为目标反复重跑同题。

2026-10-06 实测准备状态：Serper 返回 200；OpenRouter Responses 返回 401 `User not found`；两个现有 LiteLLM key 都返回 429 `budget_exceeded`。因此尚未启动 BrowseComp 模型求解，也未获得正常完成的实测结果。需更新凭据/额度后继续，不能将离线测试或 preflight 错误当作真实测试成功。
