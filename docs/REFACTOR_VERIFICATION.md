# BCG 重构验证指南（人工测试部分）

> 给负责对重构后代码进行充分测试的开发者。
> 目标：确认旧版功能**没有丢失**、功能效果**没有变质**。
> 修订：2026-08-06 · 分支：`jun/refactor`（待合并 main 前的验证用）
> 本文件为本地文档，不进入版本库。

---

## 0. 自动化已覆盖（无需人工，仅供参考）

以下验证已全部自动化并通过，**不在本清单内**：

- `make check` 全链：Python 174、Agent 121、Dashboard 15、shell、scripts
  golden（15 断言）、contracts、release manifest、install smoke、hygiene
- 双后端 golden（`tests/fixtures/refactor/construct_{api_based,light}.json`，
  重构前捕获的逐字段产物基线）+ entrypoint 默认值 golden
- 契约一致性（http/artifact schema + TS 生成类型 freshness）
- **Agent 确定性 A/B**：`bash scripts/ab_agent_deterministic.sh`——同一
  fixture 驱动新旧版（`main` vs 当前分支）的 `formatBcgMarkdown` 与
  `BcgContextManager.transform/augmentSystemPrompt`，逐字节一致（无需 LLM）
- wheel 内容、安装冒烟、脚本 dry-run

**本清单只列必须人工执行的内容**（需要真实 LLM、真实浏览器、真实旧数据或
本机配置）。

---

## 1. 前提与准备

1. 干净环境：用临时 HOME（`HOME=$(mktemp -d)`）避免本机旧配置干扰。
2. 真实模型可用（API key 已配；`bcg setup` 或 `~/.bcg/config.yaml`）。
3. 旧版对比需要两个工作树：
   ```bash
   git worktree add /tmp/bcg-old main      # 旧版
   # 当前分支即新版（jun/refactor）
   ```
   各自按 README 源码安装（`make install`），配**同一 API key / 同一模型**。
4. 本机旧数据备用：旧 `~/.bcg/` 配置、旧 session 文件、旧构造产物
   （`memory.json`/`belief_graph.jsonl`）。

---

## 2. 人工测试清单

### 2.1 真实构造管线（最高优先级）

| 项 | 步骤 | 预期 |
|---|---|---|
| api_based 真实构造 | 按 README 示例，用真实 LLM 构造一个短对话（≥3 轮，含工具调用更好） | 产出 graph.json/memory.json，字段结构符合契约；置信度为确定性计算值 |
| light 真实构造 | 同上，切到 light 后端（需本地 vLLM/SGLang + 嵌入模型） | 同上；两后端产物结构一致（数值允许模型差异） |
| 双后端产物对照 | 同一对话两后端各跑一次 | n_beliefs/关系数/置信度分布合理，无字段缺失 |

判定：产物结构与 golden/契约一致；无报错、无空图。

### 2.2 真实 Agent 对话 A/B（效果一致的核心）

同一输入在旧版（`/tmp/bcg-old`）和新版各跑一遍，对比：

| 对比维度 | 一致 = |
|---|---|
| 模型消息序列（role/content/顺序） | 一致（除时间戳） |
| `/graph` 注入的系统提示图 markdown | 文本一致 |
| 模型调用参数（model/max_tokens 等） | 一致 |
| 构造服务调用（/turns 触发时机、问题 ID） | 一致 |
| 响应轨迹（保存两版完整对话记录） | 结构/要点/工具调用序列一致；措辞差异可接受（LLM 随机性） |

建议场景 3-5 个：普通问答、多轮追问、工具调用、长上下文、图信息注入场景。
判定：任何**结构性**差异 → 检查是否属于有意变更清单（`bcg --help` 默认值、
错误消息含信封文本等），否则判定回归。

### 2.3 旧 session 恢复

把重构前的 `~/.bcg/agent/` 会话文件复制到临时 HOME，启动新版 `bcg-agent`：
预期能恢复会话（v1/v2 → v3 迁移），历史消息完整、顺序正确、`/session`
统计数字与旧版一致。

### 2.4 旧配置迁移

若本机仍有旧 `model_config.json` / `~/.bcg/config.json`（重构后不再读取）：
对照 `bcg/config/config.example.yaml` 手工迁移到 `~/.bcg/config.yaml`，
`bcg config show` 确认模型/管道参数正确。

### 2.5 旧产物只读消费

把重构前生成的 `memory.json` / `belief_graph.jsonl` 喂给：
- Dashboard artifact replay（`loadArtifactReplay` 路径）
- benchmark（`bcg benchmark` 读取）
预期正常解析渲染，无字段丢失。

### 2.6 Dashboard 真实连接

1. 起真实服务：`python -m bcg.apps.online_server api_based`（需模型配置）
2. `npm --prefix dashboard run dev` → 页面加载**真实图**（非 sample）
3. 关掉服务刷新 → 应明确报错（不是伪装成 sample 成功）
4. 对比重构前截图：节点/边渲染、metrics、trajectory 一致

### 2.7 交互命令冒烟（真实 TUI 会话）

在新版 Agent 会话中逐项操作：`/login`、`/model`、`/graph`、`/session`、
`/tree`、`/fork`、`/name`、`/export`、`/import`、`/logout`。
预期：行为与旧版一致（这些 flow 全部被抽取重构，是行为差异高发区）。

### 2.8 干净环境安装

按 `install.sh` 全流程在干净环境走一遍（需 curl/tar/node）：下载 → 安装 →
`bcg --version`、`bcg-agent --version` 可用 → 首次启动创建 `~/.bcg/`。

---

## 3. 结果记录模板

```
功能域：____（如 真实构造 / Agent A/B / 旧 session 恢复）
验证步骤：____
预期：____
实际：____
结论：✅ 符合预期  /  ⚠️ 差异
若 ⚠️：属于"有意变更清单"（`bcg --help` 默认值、错误消息含信封文本、
        import bcg 不加载 .env、旧路径删除）？是/否
   - 否 → 判定回归，附最小复现（命令 + 输入 + 输出 diff）
```

最终交付：覆盖全部 2.x 项的结果表 + 任何回归的最小复现。

---

## 4. 关键参考

- 重构执行记录（含全部有意变更清单）：`docs/REFACTOR_PLAN.md`
- 行为契约：`contracts/`（schema、fixtures、生成类型）
- 双后端产物基线：`tests/fixtures/refactor/construct_{api_based,light}.json`
- 配置规范：`bcg/config/defaults.yaml` + `config.example.yaml`
- 确定性 A/B 脚本：`scripts/ab_agent_deterministic.sh`（已自动化，无需人工）
