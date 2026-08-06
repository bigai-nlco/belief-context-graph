# BCG 跨组件契约（contracts/）

本目录是 BCG 跨组件边界的**单一规范来源**。契约按边界分开定义，不合成
"万能 Graph 类型"：

| 文件 | 边界 | 生产者 | 消费者 |
|---|---|---|---|
| `http.schema.json` | 图构造服务的 HTTP 请求/响应（snapshot、/turns、/release、/health、错误信封） | `bcg.apps.online_server` | `agent-cli`（`BcgContextManager`） |
| `memory-document.schema.json` | 持久化 memory document（`memory.json`，`schema: bcg.memory.v2`） | `bcg.core.runner` | dashboard、benchmark、回放工具 |
| `stream.schema.json` | stream JSONL（`belief_graph.jsonl` 行、`trajectory_stream.jsonl`、`result.json`） | `bcg.construct._shared.session` | `dashboard/bcg_viewer`、`build_stream_manifest.py` |
| `defaults.json` | 跨语言共享默认值（server host/port） | — | Python `bcg/config/defaults.yaml`、`agent-cli` settings |
| `fixtures/` | 确定性请求/响应基准，Python 与 TypeScript 测试共用 | — | Python/TS contract tests |
| `generate_ts_types.py` | 从 `http.schema.json` 生成 TypeScript 类型 | — | `agent-cli/src/core/context/bcg-contract.types.ts` |

## 版本与兼容规则

- 每个 schema 文件带 `schema_version` 整数。HTTP 服务在 `GET /health` 暴露
  当前 `schema_version`；消费者（如 Agent 健康检查）可校验并据此降级或报错。
- **向后兼容只允许：新增 optional 字段。** 删除字段、重命名字段、改变类型、
  改变默认值或收紧枚举，必须递增 `schema_version` 并提供 migration 说明。
- 服务不能在同一 refactor-only 提交中停止接受旧 payload；先发布新字段，
  再在下一个版本协商窗口淘汰旧字段。
- artifact 版本沿用各自标记：`bcg.memory.v2`、`bcg.segments.v2`；HTTP 层
  版本见 `health.schema_version`。

## 生成与校验

```bash
# 从 http.schema.json 重新生成 TypeScript 类型（--check 用于 CI）
uv run python contracts/generate_ts_types.py [--check]

# 全仓契约校验（make check 的一部分）
make check-contracts
```

- Python producer contract tests（`tests/test_http_contract.py`）用
  `jsonschema` 校验真实 handler 响应必须匹配 `http.schema.json`。
- TypeScript 侧使用生成类型（`bcg-contract.types.ts`），不再手写断言；
  `generate_ts_types.py --check` 保证生成文件与 schema 同步。
- 跨语言 fixture（`fixtures/`）由 Python 与 TypeScript 测试共同消费：
  同一个请求/响应基准两边各验证一次。

## BCG 配置 → Agent 设置映射

Agent 的 `contextManagement.bcg`（`~/.bcg/config.json` 或 settings）字段与
BCG 配置的映射：

| Agent 设置字段 | 来源 | 优先级 | 敏感 |
|---|---|---|---|
| `url` | `BELIEF_GRAPH_URL` 环境变量 → `defaults.json` 的 `server.host:port` | env > settings > 默认 | 否 |
| `recentTurns` | Agent settings（`context.recentTurns`，setup 写入） | settings | 否 |
| `timeoutMs` | Agent settings 默认 60000 | settings 默认 | 否 |
| `maxTurns` | Agent settings 默认 40 | settings 默认 | 否 |
| `includeRelations` | Agent settings 默认 true | settings 默认 | 否 |

- URL 默认值的唯一规范来源是 `contracts/defaults.json`；Python
  `bcg/config/defaults.yaml` 的 `server` 域必须与其一致（CI 校验），
  Agent 侧不再硬编码 host/port。
- 敏感边界：API key 只通过 `api_key_env` 引用 `.env`；settings/契约/日志/
  fixture 中不得出现内联密钥。

## 错误语义

- 所有非 2xx 响应统一为 `{"error": "<消息>"}`（见 `errorEnvelope`）：
  - 400：请求体非法（JSON 解析失败、缺字段）
  - 404：未知路径、未知 `problem_id`
  - 409：trajectory 已 finalize 后再 push（`TrajectoryClosedError`）
  - 500：未预期异常
- 消费者应把状态码 + 错误文本一起呈现；`/release` 对未知 `problem_id`
  返回 `200 {"released": false}`（幂等语义，不是错误）。
