# 原生 Codex + BCG Gateway 验证记录

本文只记录 gateway。当前目录的 Codex 接入只保留 gateway，原生 Codex 自己的 compaction 功能不受这次源码整理影响。

## 持续会话验证

2026-10-08 使用 Codex CLI 0.160.1 验证了同一个原生 TUI 中的连续 6 个用户轮次和一次原生 `/compact`。Agent 为 `gpt-6-luna`，`reasoning_effort=low`，使用本机 ChatGPT 登录的临时私有副本；Graph 为 OpenRouter `qwen/qwen3.8-flash`，`reasoning_effort=none`，使用独立 API key。

实际请求记录显示：

- 第 4 次主请求从 14 个 input item 变为 13 个，第 5 次从 16 个变为 14 个。
- 裁剪后的 input 与 `partition(original_input, recent_turns=2).retained` 一致。
- 原 user/system/developer 消息保持不变，Graph 只追加到原 `instructions` 后面。
- 连续回答和压缩后的继续对话均正常。标题生成和 compaction 辅助请求原样通过，不污染主图。
- 该轮验证没有 409 或上游 HTTP 错误。

这次原生压缩走的是标记为 `request_kind=compaction` 的普通 Responses 请求，而不是 `/responses/compact` 路由。独立压缩路由与未知/加密 item 的回退由 mock 测试覆盖，不应描述成已在这一真实会话中出现。

Graph LLM 用量为 27,754 input tokens、3,501 output tokens，共 31,255 tokens，OpenRouter 返回费用为 $0.004848。这不是 Agent 的账单，也不包括此前排错测试。Agent 使用 Codex 登录，不能据此换算其美元费用。

本地记录位于 `results/codex-native-live-20261008/gateway-03/`，包含运行设置、原生 session 记录、Graph 输出和请求前后对照。它们属于私有测试数据，不应直接上传 GitHub。

## 通用工具输出与连续请求复测

2026-10-15 在 Codex CLI 0.162.1 下，使用 Agent `gpt-6-luna low` 和 LiteLLM Graph `gpt-6-luna none` 做了真实复测。Agent 在同一流程内执行三次工具调用，依次得到 Project Orion 的端口、环境和状态，再输出 `ORION_READY`。

四次模型请求均正常完成。最后一次输入从 13 个 item 变成 11 个，移出一个完整步骤，注入 542 字符的 Graph 内容。长工具输出中的端口事实被提取成 29 字符的 belief：`Project Orion uses port 8860.` 没有出现并发冲突或 Graph 选择回退。

这验证了通用工具输出会进入有界事实抽取，以及流式响应终止后下一次请求可以继续。它不是 benchmark 准确率或节省 Token 的结论。

## Gateway-only 源码整理

独立 Graph runtime 保留原来的 unified SDK、抽点/建边 Prompt、confidence 传播和运行参数，不依赖 hooks、事件 SQLite 队列或后台 worker：

- 抽点上下文：12,000 字符。
- 抽点历史：最近两轮 Graph 节点。
- 建边候选：最多三层。
- Agent 原始上下文：最近两个已完成步骤。
- Graph 选择：focused，节点预算 6,600 字符。

这次移出其他接入代码只做离线回归，不重新调用付费模型。相关测试文件是 `tests/test_codex_gateway.py`、`tests/test_codex_gateway_init.py`、`tests/test_codex_proxy.py`、`tests/test_codex_proxy_verification.py` 和 `tests/test_codex_proxy_live_setup.py`。

## 复现工具与边界

`scripts/codex_gateway_smoke.py` 用于在隔离工作目录和临时 Codex 认证副本中人工驱动持续会话，至少完成两次用户请求、一次原生 `/compact`，并继续到发生 Graph 裁剪；`scripts/check_codex_gateway_smoke.py` 审查运行记录、请求裁剪、压缩透传和配置指纹。测试脚本不代表默认安装必须注册搜索 MCP，搜索适配器只用于实验。

Gateway 不保证所有历史都能被准确恢复。同步建图会增加延迟，原生 compaction 或不可解释的历史会触发保守回退。Code Mode 内层工具、远程环境和其他 transport 仍需逐场景验证。加密 reasoning 不会被解密，也不会冒充被读取。

安装、配置和记录查看方式见 [Gateway 接入说明](codex-native-integration.md)。
