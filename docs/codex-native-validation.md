# 原生 Codex + BCG：三种接入方案的实测

验证日期：2026-10-07 至 2026-10-08。代码全部位于隔离副本 `belief-context-graph-codex`，没有修改原项目，也没有 push、全局安装或替换用户现有 Codex。

## 实测设置

- 原生 Codex CLI 0.160.1，真正的交互式 TUI；不是旧 BCG Agent，也不是每一轮重新执行 `codex exec`。
- Agent：`gpt-6-luna`，`reasoning_effort=low`，使用本机 ChatGPT 登录的临时私有副本。没有把登录凭据交给 OpenRouter。
- Graph：OpenRouter `qwen/qwen3.8-flash`，`reasoning_effort=none`，key 来自 `~/.bcg/.env`。实测 Graph LLM 返回的 reasoning token 数为 0。
- 使用现有 unified Graph 引擎、本地 CPU embedding；未改抽点/建边 Prompt 和 confidence 传播公式。
- 所有测试使用独立 `CODEX_HOME` 和工作目录；只在隔离测试中允许已审查的 Hooks 无需重复信任确认。普通用户安装仍须在 `/hooks` 审查并信任。

## 已验证的结果

| 方案 | 原生持续会话测试 | Context 实际行为 | 结果 |
| --- | --- | --- | --- |
| `assist` | BrowseComp 一题 + 两次追问；同一会话共 3 个用户轮次，中间执行一次 `/compact` | 18 个事件自动入图；12 次真实 Serper 工具结果被捕获；7 条 Graph developer context；模型实际调用 `recall` 和 `inspect` | 通过：压缩后仍能检索原问题、belief、confidence 和原始来源 |
| `compact` | 两个用户轮次 + 一次 `/compact` | 初始约束保存入图，压缩后通过 Hooks 回注；4 个事件完成建图；模型检索并查看证据 | 通过：找回 `eu-west-1`、`ORCHID-27` 和不得删除审计凭据的约束 |
| `gateway` | 同一个 TUI 连续 6 个用户轮次 + 一次 `/compact` | 第 4 次主请求：14 → 13 个 input item；第 5 次：16 → 14；保留最近两步并把 Graph 写入 `instructions` | 通过：原用户/指令消息未改写；连续回答和压缩后继续对话均正常；没有 409 或上游 HTTP 错误 |

`gateway` 的逐请求审计验证了：裁剪后的 input 与 `partition(original_input, recent_turns=2).retained` 完全一致；原来的 user/system/developer 消息没有被改写；Graph 只追加到原 `instructions` 后面。保留/配对工具调用和结果的边界还由回归测试覆盖。

本机 TUI 的标题生成虽然标记为 `request_kind=turn`，但 `thread_source=thread_title`。首次实测发现它与主对话争用会话，产生 409；现在按原生线程 ID 隔离，并明确让标题请求绕过建图。最终实测中，标题与 compaction 共两次辅助请求均原样通过，未污染主图。

代理下本次 `/compact` 走的是原生 `request_kind=compaction` 的普通 Responses 请求，而非 `/responses/compact`。代理原样透传；压缩后当前历史仍可解析，随后启动新图段继续工作，没有触发 opaque-item 回退。单独的 `/responses/compact` 路由与未知/加密 item 的保守回退由本地 mock 测试覆盖，不能把它们说成已在本次真实请求中出现。

## 一个实际的记忆使用片段

BrowseComp 后续追问中，Codex 检索并检查了 graph `codex_3b5e80abb0a7_d514475219cd` 的 belief `B7`。它报告该节点的日期判断和 confidence `0.961`，同时明确指出证据来自搜索摘要和质量存疑的网站，置信度不能替代独立核实。

这表明 Agent 能在原生会话中使用 BCG 的节点和证据接口；**不表明这道题答对了**。这次最终回答是 `1985–1986`，与数据集参考答案 `1988–96` 不一致。流程验证不能当作准确率或节省 Token 的实验结论。

## 费用与测试记录

| 通过的会话 | Graph LLM input tokens | Graph LLM output tokens | Graph LLM total tokens | OpenRouter 返回的 Graph 费用 |
| --- | ---: | ---: | ---: | ---: |
| `assist-01` | 35,301 | 7,540 | 42,841 | $0.008565 |
| `compact-01` | 7,394 | 1,693 | 9,087 | $0.001630 |
| `gateway-03` | 27,754 | 3,501 | 31,255 | $0.004848 |

以上只列通过的三组会话，不包括排查代理并发问题时的短暂启动测试；只计算 Graph LLM，不把本地 embedding token 加进来。Agent 使用 Codex 登录，无法从这些日志得出其美元账单，也不应套用 OpenRouter Agent 单价。

本地记录：

```text
results/codex-native-live-20261007/assist-01/
results/codex-native-live-20261007/compact-01/
results/codex-native-live-20261008/gateway-03/
```

每组保留 `run.json`、`exit.json`、原生 session 记录和 `verification.json`。Graph 完整记录在各自的 `workspace/.bcg-codex/state/graphs/`；代理的真实请求前后对照在 `state/gateway/model_io/`。这些私有记录未加入 Git，分享前应检查其中的任务内容和加密 reasoning 数据。

所有测试正常退出，指纹检查确认用户原始 `auth.json`、`config.toml`、`hooks.json` 完全未改。临时认证副本已删除，测试启动的 Codex/Graph/MCP 进程已停止。

## 回归和安装包验证

- 177 项针对性的本地回归测试通过：原生 Hooks/MCP、事件存储与恢复、数据脱敏、请求裁剪、工具配对、辅助请求隔离、compaction 透传，以及现有 CLI、HTTP 和 Graph 配置契约。
- Ruff 检查和 `git diff --check` 通过。
- 成功构建 wheel，在 `/tmp` 单独安装该 wheel 后验证了 `bcg.integrations.codex.cli --help`。这使用了已有依赖环境，不等于全新机器上的完整依赖安装验证；没有全局安装或发布包。
- `scripts/check_codex_native_smoke.py` 对上述三组实测记录的机械检查全部通过。检查涵盖持续会话、压缩后记忆回注、请求前后对照和全局配置不变，不代替任务正确性评估。

## 选择建议和边界

- 想先让普通 Codex 用户稳定用上 BCG：选 `assist`。原生工具和交互不变，但额外记忆会增加上下文，不能声称已经省 Token。
- 想尽量利用原生压缩能力调整 Context：选 `compact`。确定实测的是压缩流程和图回注；配置的自定义 handoff 模板是否被特定原生 provider 路径采纳，仍需逐版本验证，不能保证永久保留用户原文或精确两轮窗口。
- 想测试严格请求侧窗口：选 `gateway`。它确实改变模型输入，但同步建图会增加延迟，本次压缩后的下一次回答约 50 秒；Code Mode 内层工具的捕获也不如 Hooks 完整。native compaction、加密历史、不同 transport 和多 agent 场景仍需额外兼容性测试。
- Hooks 当前不直接提供中间 Thinking；不会冒充读取到隐藏 reasoning。托管工具、某些错误返回和特殊工具路径可能没有 `PostToolUse`，因此事件记忆不等于完整模型 I/O 记录。
- 同一会话恢复时可以检索已经保存的图；新工作另建图段，不自动虚构跨段关系。Linux 之外尚未验证。

安装、模式切换和停用命令见 [接入说明](codex-native-integration.md)。
