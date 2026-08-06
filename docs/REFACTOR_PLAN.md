# BCG 全仓代码结构问题分析与重构执行计划

> 状态：第一部分步骤 0-8 已完成；第二部分步骤 10-16 待执行
> 修订：2026-08-06 · 分支：`jun/refactor` · Python 重构基线：`4a1927b`
>
> 本文件是受 Git 管理的正式执行计划。根目录的
> `REFACTOR_PLAN_20260804.md` 仅保留为最初工作记录，不再作为规范来源。

本计划分为两部分：第一部分保留已经完成的 Python `bcg/` 包重构记录；
第二部分覆盖 `agent-cli/`、`dashboard/`、`deploy/`、`scripts/`、CI、安装、
文档和发布治理。第二部分不是推倒重写，而是在现有绿色基线上逐个收敛组件边界。

---

## 第一部分：Python `bcg/` 包重构（步骤 0-8，已完成）

本部分中的“当前”“本轮”等表述记录的是 2026-08-04 至 2026-08-05 的执行现场，
用于解释已完成提交的决策，不代表第二部分的全仓现状。

## 1. 背景、目标与边界

当前 `bcg/` 包共有 63 个 `.py` 文件、23,264 行；包根目录有 16 个 `.py` 文件。项目已经包含 `bcg/py.typed`，但尚未确认它是否随 wheel 正确发布。

当前主要问题是：SDK 核心与应用层混在同一个命名空间；`api_based` / `light` 两个构造后端由复制演化而来；核心代码直接依赖具体后端；配置和 CLI 默认值分散；部分导入会修改进程环境；持久化 artifact 和公共导入路径缺少明确的兼容契约。

### 1.1 最终目标

1. `bcg` 具备清晰、可验证的依赖方向，核心层不依赖应用层或具体构造后端。
2. 双后端共享确定为同一语义的基础设施，真实算法差异通过明确的 adapter / strategy 保留。
3. 用户配置经过统一的 Pydantic schema 校验；行为默认值只有一个规范来源。
4. SDK、CLI、服务、回放、可视化和 benchmark 的职责边界明确。
5. 公共 Python API、CLI 调用方式和持久化 artifact 在约定的兼容窗口内保持可用。
6. 消除 import 时修改 `os.environ` 的副作用，secrets 不写入 YAML。

### 1.2 不作为本轮重构目标

- 不在结构迁移提交中顺手修复算法问题或改变置信度公式。
- 不强求所有相似代码抽象为同一个实现；只有语义和生命周期一致的逻辑才共享。
- 不以“文件行数”或“逐字节重复为零”作为架构成功的唯一标准。
- 不在没有单独决策和迁移说明的情况下改变 artifact schema、CLI 默认值或 SDK 默认行为。
- 本轮先做包内分层；是否把 apps 拆成独立发行包另行决策。

### 1.3 当前基线状态（开始重构前必须处理）

2026-08-04 实测：

- `uv run pytest -q`：90 passed，1 failed。
- 失败项：`tests/test_belief_graph.py::test_construct_confidence_is_the_only_confidence_policy`，测试期望的 `confidence_history` 与当前实现不一致。
- `uv run ruff check bcg/`：`bcg/graph.py` 存在 1 个 I001 导入顺序错误。

步骤 0 完成后：

- `uv run pytest -q`：112 passed。
- `uv run ruff check`：All checks passed（沿用当前 exclude 范围）。
- api_based / light normalized golden、入口默认值、公共 import、旧 CLI help 和当前 env import 副作用均已有契约测试。

绿色基线已经建立；后续 refactor-only 提交以这些结果为回归门槛。

步骤 1 完成后（2026-08-05 实测）：

- `uv run pytest -q`：115 passed（步骤 1 新增 3 个架构/契约测试）。
- `uv run ruff check`：All checks passed。
- 新文件：`bcg/core/contracts.py`、`bcg/core/client_adapter.py`、`bcg/core/pipeline.py`、`bcg/construct/backends.py`、`bcg/construct/{api_based,light}/adapter.py`。
- `bcg/runner.py` 不再模块级导入任何具体后端，通过 `resolve_backend()` 懒解析 + Protocol 驱动；`isinstance(BeliefGraphOptions)` 分支已删除。
- `api_based/pipeline.py` 不再反向导入 `bcg.runner`；`BeliefGraphOptions` 改为继承 `core.contracts.RunOptions`，`BeliefGraphPipeline` 继承 `core.pipeline.BeliefGraphPipelineBase`。
- 兼容别名：`BeliefGraphRunPaths/RunResult`（contracts.py）、runner 私有 `_ConstructClientAdapter`/`_resolve_backend`（供既有测试引用）。
- api_based adapter 复刻原行为：不应用 `belief_graph_config`（原实现只有 light 应用），行为等价。

步骤 2 完成后（2026-08-05 实测）：

- `uv run pytest -q`：115 passed（golden 契约测试不变）。
- `uv run ruff check`：All checks passed。
- 新增共享模块：`bcg/construct/_shared/`（`__init__.py`、`loaders.py`、`roles.py`、`spans.py`、`llm.py`），均不导入具体后端。
- `loaders.py`：逐字节相同的两份合并为 `_shared/loaders.py`（git mv 保留 api_based 历史），4 处 import 改向。
- role normalization：`_ROLE_ALIASES` 5 处定义统一为 `_shared/roles.py::normalize_role`。
- span trimming：`trim_span` 4 份相同实现（api utils/split/evidence、light split）统一为 `_shared/spans.py`；api_based/utils 保留公共 re-export。
- 置信度数学：`logit`/`sigmoid`/`clamp_confidence`/`posterior_confidence`/`CONF_FLOOR`/`CONF_CEIL` 统一到 `bcg/core/confidence.py`（3 处实现、常量一致）；graph.py 保留 `_confidence_from_dimensions`（模型层逻辑），api/light 的 confidence policy（初始策略、参数化）保持分开。
- llm 共享基础设施：`TokenUsageTracker` + usage/日志基础设施（各 109-467 行）+ `EmbeddingClient` + `cosine_similarity_matrix`（逐字节相同）抽取到 `_shared/llm.py`（566 行）；两个后端 `llm.py` 变薄壳（567/633 行）并 re-export 全部符号，`USAGE`/`TokenUsageTracker`/`EmbeddingClient` 现为跨后端共享单例；`LocalEmbeddingClient`、`make_embedder`、`load_config`、`call_model`、`parse_json_response` 等后端特有逻辑保留。
- **暂缓项**：`parse_json_response` 两版实现不同（api_based 用 `find/rfind` 大括号截取，light 用 `JSONDecoder` 迭代解析），在 refactor-only 约束下不强行合并，由 golden 契约保护，留待行为变更步骤评估统一。

步骤 3 完成后（2026-08-05 实测）：

- `uv run pytest -q`：119 passed（新增 4 个 step3 契约测试）。
- `uv run ruff check`：All checks passed。
- 新增 `bcg/construct/_shared/session.py`（git 删除 855 行 / 新增 513 行）：`StreamingTrajectorySession` 状态机 + `resolve_dated_output_root` + 辅助函数，两个 online.py 各剩 SessionManager（约 -413 行）。
- 新增 `bcg/construct/_shared/writers.py`：`EventRecorder`（两后端逐字节相同的 `_event` 抽取）+ `ArtifactWriter`（temp-file + `os.replace` 原子 JSON 写出，接入 result.json）。
- session 差异注入：`builder_cls`/`options_cls` 由后端传入（SessionBackendAdapter 新增字段、两个 adapter 声明）；`edge_generator` 仅 light 传（条件传参，api builder 不接受该参数）。
- 后端保留：SessionManager（config wiring / current_output_root / edge_generator 分发）、graph mutator（`_make_node`/`_keep_only_latest_decision` 等）、LLM/extractor 调用、timing.csv 与 result dict 后端特有格式。

步骤 4 完成后（2026-08-05 实测）：

- `uv run pytest -q`：137 passed（新增 18 个 config 测试：4A 十二个 + 4B 六个）。
- `uv run ruff check`：All checks passed。
- 新增 `bcg/config/`：`schema.py`（Pydantic，`extra="forbid"`、`schema_version`、范围/枚举校验；models 为动态 key 的 `ModelEntry` dict）、`loader.py`（分层加载 + deep merge + 字段级来源追踪）、`defaults.yaml`（唯一行为默认值来源）、`config.example.yaml`、`migration.py`、`cli.py`。
- 优先级（按文档）：显式 `--config` > `BCG_CONFIG` > 项目 `bcg.yaml` > 用户 `~/.bcg/config.yaml` > 包内 defaults.yaml；deep merge 规则：mapping 递归、list 整体替换、null 回退；CLI 覆盖仅非 None 生效。
- `cli_defaults` legacy profile（0.86/100000/True）与 `runner` profile（0.8/9000/False）在 defaults.yaml 中分离保存，待步骤 7 统一。
- `bcg config show`（有效值 + 每字段来源）、`bcg config migrate`（原子写 + `.bak` 备份 + 幂等）已接入根 CLI。
- 4B：两个旧 schema adapter（model_config.json、setup 的 camelCase config.json）；legacy 仅作回退且发 `DeprecationWarning`；内联 `api_key` 丢弃并警告（`api_key_env` 引用保留）。
- **步骤 4 当时消费者未切换**：runner/CLI 仍读旧配置；最终已在步骤 7 和八步完成审计中接入分层 YAML settings，并保留旧 JSON 兼容回退。
- 打包：`pyyaml>=6.0` 显式依赖；hatch force-include 新增两个 YAML；wheel 内容测试（`uv build` + zipfile 检查 defaults/example/py.typed）。

步骤 5 完成后（2026-08-05 实测）：

- `uv run pytest -q`：139 passed（步骤 5 契约测试更新/新增 3 个）。
- `uv run ruff check`：All checks passed。
- SDK 核心移入 `bcg/core/`（git mv 保留历史）：graph/memory/runner/llm/env/utils/tracing + 既有 contracts/client_adapter/confidence/pipeline。
- 旧路径 `bcg.graph`/`bcg.memory`/`bcg.runner`/`bcg.llm`/`bcg.env`/`bcg.utils`/`bcg.tracing` 为薄转发壳（`import *` + 显式私有符号），**未发弃用警告**（本轮不弃用，警告推迟到兼容窗口结束的 breaking release）。
- `bcg/__init__.py` 收窄：只导出 `BCG`/`BCGMemory`/`BCGRunner`/`LLMClient`/`BCGSettings`/`load_settings`；`import bcg` 不加载 apps、不加载具体构造后端、不修改 `os.environ`（隔离进程测试验证）；`import bcg.env` 仍执行 import-time 加载（兼容层行为保留，步骤 7 再删）。
- memory 去后端依赖：api_based 的初始置信度 policy（`BASE_CONFIDENCE` 表、`initial_confidence`、`init_belief_confidence` 等）逐字节搬入 `core/confidence.py`，`api_based/confidence.py` 改 re-export（同一对象，行为零变化）；light 的 config 化 policy 保持独立。
- core 各文件补生成 `__all__`；`bcg.graph` 的 `CONF_FLOOR/CONF_CEIL` 兼容 re-export 恢复（此前被 ruff F401 误删，已在提交中修复）。
- **行为变更点**（文档 3.2 明确记录）：`import bcg` 从"加载 .env"变为"不加载"（步骤 0 契约测试 `test_import_bcg_currently_loads_explicit_project_env` 更新为反向断言）；`bcg.__all__` 移除 `PROJECT_ENV_FILE`。CLI 入口不受影响（显式加载路径保留）。
- 提交：`refactor: move SDK core under bcg.core with compatibility shims`（单提交；core 内部互相依赖，拆小会产生 broken 中间态）。

步骤 6 完成后（2026-08-05 实测）：

- `uv run pytest -q`：139 passed；`uv run ruff check`：All checks passed。
- 应用层移入 `bcg/apps/`（git mv 保留历史）：setup、agent_runtime、run、online_server、online_driver、visualize_beliefs_graph、cli、cli_help、benchmark/（文件名保留，未改成 server.py/driver.py 等）。
- 旧路径全部为薄转发壳（含 `__main__` 块，`python -m bcg.run` 等旧用法可用）；`bcg/benchmark/` 重建为转发包（6 个转发模块）；`[project.scripts] bcg` 指向 `bcg.apps.cli:main`。
- 新增 `apps/cli_options.py::add_run_options`：run/server/driver 三处复制粘贴的运行参数组收敛（默认值保持 0.86/100000/verify-ON，未统一）；server 内两个解析器共用同一参数组。各入口的 config/input/host 参数保留（有真实差异：run 的 `--input` required + `--item`/`--keep-order`、server 的 `--host`/`--port`、driver 的 `--input` 可选）。
- 删除 3 处 `sys.path.insert` hack（run/online_server/online_driver）。
- 修复 `setup.py` 打包模板路径（`__file__` 定位改为包根）。
- 测试调整：白盒测试（monkeypatch 模块属性/常量）改指 `bcg.apps.*`（`_serve_forever`、`_resolve_agent_command`、`subprocess.run`、`DEFAULT_GRAPH_URL` 等）；构造 CLI 的旧路径延迟导入路径（`bcg.run.main` 等）仍 patch 转发壳（验证旧路径兼容）。

步骤 7 完成后（2026-08-05 实测）：

- `uv run pytest -q`：140 passed；`uv run ruff check`：All checks passed。
- **ADR 决策（用户确认）**：三个冲突默认值统一到 SDK 值——`incremental_merge_threshold=0.8`、`context_chars=9000`、`verify_merge=False`；CLI 三入口行为相应变化。
- 单一来源落地：`cli_options.add_run_options` 从 `defaults.yaml` 的 `runner` 域读取默认值（删硬编码 0.86/100000/True）；`cli_defaults` legacy profile 与 `LegacyCliProfile` schema 模型删除；`run_merge_pass` 签名默认统一为 0.8（调用方总是显式传，实际行为不变）；`entrypoint_defaults.json` golden 重新捕获（0.8/False/9000）。
- **env 副作用移除**：`bcg/core/env.py` 删除 import-time `load_project_env()`；7 个 CLI 入口（cli/run/online_server/online_driver/agent_runtime/visualize/benchmark）在 `main()` 显式 `_bootstrap_env()`；`resolve_config_api_key` 保留内部显式调用。
- 契约测试更新：`import bcg.env` 不再加载 .env（新契约）+ 显式 `load_project_env()` 仍工作；config 测试删除 cli_defaults 断言。
- 提交：`config: unify run defaults and remove env import side effect (step 7)`（行为变更单提交）。

步骤 8 完成后（2026-08-05 实测）：

- `uv run pytest -q`：142 passed（新增 2 个 errors 层级测试）；`uv run ruff check bcg tests`：全绿且**无 exclude**。
- **版本号统一**：`bcg.__version__`（及 `construct.api_based/light` 的 `__version__`）从安装 metadata 读取，pyproject 单一来源；历史 3.0.0/3.1.0 后端世系记入 docstring；public exports 契约更新。
- **ruff 全量启用**：移除 `extend-exclude` 中 bcg 全部路径（保留 annotation/datasets/scripts），修复 1394 项告警（~1300 自动 + 手动）。过程中发现并修复 exclude 掩盖的潜伏 bug：`_EMBEDDING_LOG_LOCK`/`_PROMPT_LOG_LOCK` 未 re-export（EmbeddingClient 日志 NameError）、`_shared/llm.py` 缺 `sys`/`openai` 导入；恢复 ruff F401 误删的三处兼容 re-export（api_based.confidence、两后端 llm.py、两后端 online.py，均标注 noqa）。
- **异常层级**：`bcg/core/errors.py`（BCGError + BCGConfigError/BCGUsageError/BCGArtifactError/BCGBackendError，子类保留标准异常基类）；config loader 接入 `BCGConfigError`。
- **artifact schema version**：已存在（`_memory_document` 顶层 `"schema": "bcg.memory.v2"`），无需新增。
- **py.typed + YAML wheel**：已有 wheel 内容测试覆盖（含 defaults/example/py.typed）。
- **测试组织**：按文档第 6 条不强制镜像源码目录；tests/ 保持领域文件 + 契约测试。
- **兼容窗口**：旧 import/CLI/JSON 兼容层保留（转发壳 + 迁移 adapter），删除仅限后续 breaking release（文档第 7 条）。

---

## 2. 已核实的问题

### P0：双后端存在大量重复，但差异不能仅按文本相似度处理

| 模块对 | api 行 | light 行 | 现状 |
|---|---:|---:|---|
| `loaders.py` | 177 | 177 | 逐字节相同，可直接共享 |
| `constants.py` | 36 | 37 | 高度相似，需确认顺序是否影响输出 |
| `online.py` | 805 | 800 | 会话骨架高度相似，构造参数有差异 |
| `llm.py` | 1089 | 1155 | 基础设施重复，配置读取与后端行为有差异 |
| `merge.py` | 949 | 956 | 骨架相似，合并策略需用测试刻画 |
| `confidence.py` | 559 | 641 | 公式相似，但默认策略和可配置程度明显不同 |
| `stream.py` | 994 | 1209 | 方法结构相似，内部算法已经分化 |

重复点包括：

- `parse_json_response`、`TokenUsageTracker`、`EmbeddingClient` 各有 3 份。
- `logit` / `sigmoid` 数学实现存在 3 份。
- `_ROLE_ALIASES = {"function": "tool"}` 存在 5 份。
- span trimming 在同一后端中也有重复实现。

这里需要区分：

- **可直接共享**：纯函数、逐字节相同且契约一致的 loader、无后端状态的基础设施。
- **需参数化后共享**：会话编排、LLM usage 记录、writer 等生命周期一致但依赖不同的逻辑。
- **应继续分离**：抽取、stance、置信度 policy、merge policy 等会产生不同业务输出的算法。

### P0：应用层与 SDK 核心混杂

以下模块属于应用或入口层，不应由 SDK 公共导入面隐式加载：

- `setup.py`
- `cli.py` / `cli_help.py`
- `agent_runtime.py`
- `run.py`
- `online_server.py`
- `online_driver.py`
- `visualize_beliefs_graph.py`
- `benchmark/`

移动这些文件会影响 `bcg.run`、`bcg.online_server` 等既有 import 和 `python -m` 用法，因此必须保留兼容转发模块，不能只修改 `[project.scripts]`。

### P0：配置没有规范来源

当前配置和默认值分布在：

- `bcg/model_config.json` / `model_config.example.json`
- `~/.bcg/config.json`
- `run.py` / `online_server.py` / `online_driver.py` 的 argparse 默认值
- `runner.py`、`pipeline.py` 和两套 `StreamOptions` 的默认值
- setup、agent runtime、benchmark 中的端口和模型 fallback

已确认的漂移包括：

- `incremental_merge_threshold`：0.8 vs 0.86
- `context_chars`：9000 vs 100000
- `verify_merge`：False vs True
- server port、model name 在多处重复

### P1：依赖方向和 API 边界不清晰

- `bcg.memory` 直接导入 `construct.api_based.confidence`。
- `bcg.runner` 在模块加载时直接导入两个具体后端。
- `runner.py` 与 `construct/api_based/pipeline.py` 通过延迟导入形成双向依赖。
- `_Backend` 只保存 class/module，实际选项转换仍依赖 `isinstance`。
- construct 与 SDK 之间的 graph/result 映射散落在多个函数中。
- `bcg/__init__.py` 导入 `env.py`，后者在 import 时调用 `load_project_env()` 修改环境变量。

### P1：测试无法充分保护当前行为

- 现有测试主要集中在 10 个扁平文件中。
- `test_belief_graph.py` 横跨 graph、memory、runner、pipeline 和双后端。
- HTTP 服务、visualize、artifact 兼容和导入副作用缺少独立契约测试。
- 现有测试不足以证明两个后端抽取共享代码后输出不变。

测试目录是否镜像源码不是核心指标；优先保证行为契约、公共接口和关键边界得到覆盖。

### P2：治理问题

- pyproject、api_based、light 各自维护版本号。
- Ruff 排除了 `bcg/construct` 及多个应用模块。
- 大量 `print()` 直接写 stdout/stderr。
- 缺少统一异常层级和 artifact schema version / migration 策略。
- `bcg/py.typed` 已存在，但需验证打包产物。

---

## 3. 重构原则

### 3.1 先建立契约，再移动或抽象

任何会影响以下边界的改动，都必须先有测试或快照：

- 公共 import：`from bcg import ...` 以及 `bcg.graph`、`bcg.llm`、`bcg.runner` 等旧路径。
- CLI：`bcg`、`bcg construct ...`、旧模块入口和 `--help`。
- artifact：`graph.json`、`memory.json`、events、segments、token usage 等。
- 双后端：给定固定输入和 stub LLM 时的节点、边、merge、confidence 输出。
- 配置：旧 JSON、新 YAML、环境变量与 CLI 覆盖的最终生效值。

### 3.2 区分行为保持提交和行为变更提交

每个 PR / commit 必须标记为：

- `refactor-only`：公共行为、默认值和 artifact 不变。
- `behavior-change`：有明确决策、迁移说明、版本影响和更新后的契约测试。

默认值统一、移除 env import 副作用、删除旧模块路径等都属于行为变更，不能伪装成纯重构。

### 3.3 依赖方向固定

```text
apps / CLI
    |
    v
config loader ---> core contracts <--- construct backend adapters
                       ^                        |
                       |                        v
                  core SDK <------------- construct shared primitives
```

约束：

- `core` 不导入 `apps`。
- `core` 不直接导入 `construct.api_based` 或 `construct.light`。
- backend adapter 实现 core 定义的 Protocol，具体后端使用延迟注册或显式注入。
- shared primitives 不反向导入具体后端。
- apps 可以依赖 core、config 和 construct，但不能成为 SDK import 的必经路径。

使用 `import-linter` 或架构测试验证这些规则；Ruff 不能证明模块之间不存在循环依赖。

### 3.4 单一实现不等于零重复文本

- 同一业务规则只能有一个规范实现。
- 后端确有不同策略时允许保留相似代码，优先保持可读性。
- 不为了降低重复率引入大量 backend flag、`if backend == ...` 或过深继承。
- 文件行数只作为观察信号；最终按职责、复杂度和测试边界验收。

---

## 4. 目标结构

```text
bcg/
├── __init__.py                 # 仅导出稳定 SDK 公共面
├── graph.py                    # 兼容转发：一个版本窗口后评估删除
├── memory.py                   # 兼容转发
├── runner.py                   # 兼容转发
├── llm.py                      # 兼容转发
├── core/
│   ├── graph.py                # 数据模型与图容器，不含 IO/LLM
│   ├── memory.py
│   ├── runner.py
│   ├── llm.py
│   ├── confidence.py           # 数学基础；后端 policy 不强制合并
│   ├── contracts.py            # options/result/artifact/backend Protocol
│   ├── errors.py
│   ├── env.py                  # 显式加载，无 import 副作用
│   ├── tracing.py
│   └── utils.py
├── config/
│   ├── schema.py               # 校验模型，不重复定义行为默认值
│   ├── loader.py               # 分层加载、解析、校验、来源追踪
│   ├── migration.py            # 两种旧 JSON 的独立迁移 adapter
│   ├── defaults.yaml           # 唯一的行为默认值来源
│   └── config.example.yaml     # 由 defaults/schema 生成或由 CI 校验一致
├── construct/
│   ├── _shared/                # 纯共享逻辑
│   ├── api_based/              # adapter + 后端特有算法
│   └── light/                  # adapter + 后端特有算法
└── apps/
    ├── cli.py
    ├── cli_options.py
    ├── setup.py
    ├── agent_runtime.py
    ├── run.py
    ├── server.py
    ├── driver.py
    ├── visualize.py
    └── benchmark/
```

兼容模块只做 re-export / main 转发，不保留第二份业务实现。

---

## 5. 分阶段执行计划

步骤存在明确依赖，不再假设“互不阻塞”。每一步可以独立合并，但必须建立在前置步骤已完成的基础上。

### 步骤 0：修复基线并建立行为护栏（已完成）

1. 修复或确认当前 pytest 失败原因，使基线全绿。
2. 修复当前 Ruff I001；记录移除 `extend-exclude` 前的完整 lint baseline。
3. 为双后端增加固定输入 + stub LLM 的 characterization tests。
4. 保存经过规范化的 artifact golden fixtures；比较时剔除 UUID、时间戳、绝对路径等不稳定字段。
5. 增加公共 import、旧模块入口、CLI help 和 env import 副作用测试。
6. 记录当前各入口的有效默认值，作为后续行为变更的对照表。

验收：

- `uv run pytest` 全绿。
- `uv run ruff check` 对当前未排除范围全绿。
- api_based / light 都有最小 golden case。
- 当前公共路径和 artifact schema 有自动化契约测试。

### 步骤 1：建立 contracts 和后端 Protocol，打破循环依赖（已完成）

1. 新建 `bcg/core/contracts.py`，放置 `RunOptions`、`RunPaths`、`RunResult`、artifact DTO。
2. 定义 `ConstructBackend` Protocol，至少覆盖 options 构造、session 创建、finalize/result 转换。
3. 为 api_based / light 各实现一个 adapter；registry 使用显式注入或延迟导入。
4. `BCGRunner` 只依赖 Protocol，不再 `isinstance(BeliefGraphOptions)`。
5. `pipeline.py` 不再反向导入 `BCGRunner`；共享 DTO 从 contracts 获取。

实际做法与计划的差异（均不影响验收结论）：

- 架构验证选用 **pytest + AST 静态扫描**（`tests/test_refactor_contracts.py::test_step1_dependency_direction_has_no_concrete_backend_imports`），未引入 import-linter 依赖；测试扫描整个 `bcg/core/` 目录，断言 core 不导入具体后端与 apps。
- `bcg/core/pipeline.py` 提供 `BeliefGraphPipelineBase` 兼容基类；其 `run()` 仍延迟导入 `bcg.runner`（单向，不构成环），待步骤 5 将 runner 移入 core 后收敛。
- runner 保留私有兼容别名 `_ConstructClientAdapter` / `_resolve_backend`（`tests/test_belief_graph.py` 仍引用）。
- 两个 adapter 基于共享的 `SessionBackendAdapter` 声明式封装（`construct/backends.py`），后端差异收敛在 options 构造与序列化函数。

验收（全部通过）：

- core 层不直接导入两个具体后端。✅
- 无 `runner.py` / `pipeline.py` 双向依赖。✅
- 两个后端的 runner golden tests 不变（115 passed）。✅
- 架构测试验证依赖方向。✅

### 步骤 2：抽取低风险共享基础设施（已完成）

按风险从低到高分多个小提交：

1. 合并完全相同的 `loaders.py`：移动其中一份到 `_shared`，删除另一份，更新 import。
2. 统一 role normalization、span trimming 等纯函数。
3. 抽取 `parse_json_response` 和无后端状态的 token usage 数据结构。
4. 抽取 EmbeddingClient 的公共传输/缓存层；后端配置映射留在 adapter。
5. 将 `logit`、`sigmoid`、clamp 等数学函数统一到 `core/confidence.py`。
6. api_based / light 的 confidence policy 和默认配置暂时分开，直到 golden tests 证明可以参数化共享。

实际做法与计划的差异（均不影响验收结论）：

- 第 3 项的 `parse_json_response` **暂缓**：两版实现不同（api_based `find/rfind` vs light `JSONDecoder` 迭代），refactor-only 约束下合并即行为变更，留待行为变更步骤；token usage 数据结构已抽取。
- 共享模块命名：`_shared/loaders.py`、`_shared/roles.py`、`_shared/spans.py`、`_shared/llm.py`；数学函数放 `core/confidence.py`（供 graph 模型与双后端共用）。
- `_shared/llm.py` 为逐字节抽取（含原注释），其中一段"Chat-model config + client"注释随抽取进入共享文件，位置语境不符，留待步骤 8 清理。
- `api_based/utils.py::trim_span` 保留为公共 re-export（`__all__` 契约不变）。

验收（全部通过）：

- 测试和 golden artifacts 不变（115 passed）。✅
- 共享模块不导入具体后端。✅
- 没有为了共享新增 backend 条件分支。✅

### 步骤 3：拆分 session、stream 和 IO 职责（已完成）

1. 从 `online.py` 抽取共享 session 状态机，后端差异通过 adapter/factory 注入。
2. 从 `StreamingBeliefBuilder` 拆出：
   - `ArtifactWriter`：文件写出和原子替换
   - `EventRecorder`：事件记录
   - graph mutator：节点、边和 merge 变更
   - LLM/extractor adapter：模型调用和后端算法
3. 保留后端特有的 extract、stance、entity、merge/confidence policy。
4. 每拆一个职责就补对应单测，不在一个提交中同时重写两个后端的大段算法。

实际做法与计划的差异（均不影响验收结论）：

- 共享 session 通过 `builder_cls`/`options_cls` 注入 + `edge_generator` 条件传参（api builder 不接受该参数，light 传）；`SessionManager` 保持后端本地（config wiring、dated output-root、edge 分发差异较大，不强行合并）。
- `ArtifactWriter` 范围收敛为原子 JSON 写出工具（接入 result.json）；timing.csv 等后端特有格式保留各自实现（不因文本相似合并，符合 3.4 原则）。
- graph mutator 与 LLM/extractor 调用未抽成共享组件：两后端算法差异大，按 1.2 原则保留（文档第 3 项本身要求保留）。
- 提交拆分：`refactor: extract shared streaming trajectory session` + `refactor: extract shared writer components (EventRecorder, ArtifactWriter)`（writers.py 同文件含两个组件，合并一个提交）。

验收（全部通过）：

- 职责和复杂度为验收核心；`stream.py`/`online.py` 行数明显下降。✅
- 每个拆出的职责有对应单测（4 个 step3 契约测试）。✅
- 未在同一提交中重写两个后端的大段算法。✅
- golden artifacts 不变（119 passed）。✅

### 步骤 4：引入统一配置系统并迁移消费者（已完成）

#### 4A：配置基础设施（refactor-only）

1. 使用 Pydantic schema 校验由 YAML parser 解析后的 dict；引入 `PyYAML` 或 `ruamel.yaml`，不得假设 Pydantic 自带 YAML parser。
2. `defaults.yaml` 是唯一行为默认值来源；schema 不再维护另一套同值默认值。
3. schema 设置 `extra="forbid"`，包含 `schema_version`，对范围、枚举和路径做显式校验。
4. loader 返回 typed settings 和每个字段的来源，便于 `bcg config show`/诊断。
5. 配置优先级固定为（显式 CLI > `--config` > `BCG_CONFIG` > 项目 > 用户 > defaults）。
6. 明确定义 deep merge：mapping 递归合并；list 整体替换；`null` 回退。
7. SDK 核心接受显式 `BCGSettings` / options 注入，不在 import 或普通构造时自动读取用户目录。
8. argparse 参数使用"未提供"状态，解析后再与 settings 合并，避免 parser 默认值覆盖 YAML。
9. 更新 hatch 配置，确保 defaults、example 和 `py.typed` 进入 wheel；增加 wheel 内容测试。

#### 4B：旧配置迁移

1. 分别实现 `model_config.json` 和 `~/.bcg/config.json` adapter。
2. 仅当新 YAML 不存在时回退读取旧格式；禁止无提示地同时合并新旧文件。
3. 提供显式迁移命令，临时文件 + 原子替换，保留备份且迁移可重复执行。
4. warning 按发布版本淘汰（回退读取发 `DeprecationWarning`，不用运行次数状态）。
5. API key 只保留 `api_key_env` 引用；YAML 和生成日志中不得写入 secret（内联 `api_key` 丢弃并警告）。

实际做法与计划的差异（均不影响验收结论）：

- 4A 的 schema 从 `model_config.example.json` 逐字段复刻（extractor/stance/edge_generation/runtime/incremental_merge/entities/confidence/chunking）；`models` 域为动态 key（模型名）→ `ModelEntry`，字段因模型类型差异设为 Optional（该处 `extra="forbid"` 仍生效）。
- null 语义统一为"回退"（跳过该键），文档化为全局约定而非逐字段。
- 步骤 4 提交当时未接入现有消费者，以保证该阶段零行为变化；最终接入在步骤 7 和八步完成审计中完成，CLI/construct loaders 现已消费分层 YAML settings，旧 JSON 作为兼容回退保留。
- 迁移命令目标默认 `~/.bcg/config.yaml`；`migrate_to_yaml` 输出先与 defaults merge 再写（避免文件缺字段导致校验失败）。
- 提交拆分：`config: add unified YAML settings infrastructure (step 4A)` + `config: add legacy JSON migration and bcg config commands (step 4B)`。

验收（全部通过）：

- 配置优先级、错误字段、类型错误、deep merge、迁移幂等性、secret 泄漏和 wheel 内容都有测试（18 个）。✅
- 137 passed、ruff 全绿。✅

### 步骤 5：迁移 SDK 核心到 `bcg/core`（已完成）

1. 移动 graph、memory、runner、llm、env、utils、tracing。
2. `memory` 使用 core confidence API，不依赖具体后端。
3. `bcg/__init__.py` 仅导出稳定公共面。
4. 在旧路径保留薄转发模块，并发出一次性、可测试的 deprecation warning（如团队决定弃用）。
5. `import bcg` 不加载 apps，不修改环境变量，尽量不加载具体后端和重型可选依赖。

实际做法与计划的差异（均不影响验收结论）：

- 转发壳未发 deprecation warning（文档允许"如团队决定弃用"——本轮不弃用；`import bcg.env` 的 import-time 副作用也保留为兼容行为，步骤 7 统一删除）。
- memory 去后端依赖通过"policy 提升"实现：api_based 的初始置信度硬规则表整体搬入 `core/confidence.py`（api_based re-export，行为逐字节不变），而非在 core 里另写实现。
- `import bcg` 加载 `bcg.construct.backends`（懒注册表，非具体后端）——符合"尽量不加载具体后端"。
- 该步骤含一处明确行为变更（`import bcg` 不再加载 .env；`__all__` 变化），契约测试同步更新并在基线记录。

验收（全部通过）：

- 新旧 import 契约均通过（7 个转发壳 + 全部既有测试）。✅
- 隔离进程测试证明 `import bcg` 不改变环境变量（新增）。✅
- golden artifacts 不变（139 passed）。✅
- core 不导入具体后端（架构测试扫描整个 core/ 通过）。✅

### 步骤 6：迁移应用层并统一 CLI 选项（已完成）

1. 将 setup、agent runtime、run、server、driver、visualize、benchmark 移入 `bcg/apps/`。
2. 建立 `apps/cli_options.py`，集中声明共享参数及其到 settings 字段的映射。
3. 保留 `bcg.cli`、`bcg.run`、`bcg.online_server`、`bcg.online_driver` 等兼容入口，内部只转发。
4. 更新 `[project.scripts]` 指向新入口，同时测试旧的 `python -m` 和 import 用法。
5. 删除 `sys.path.insert` hack。

实际做法与计划的差异（均不影响验收结论）：

- 应用文件名保留（`apps/online_server.py` 等），未改名 server.py/driver.py/visualize.py（减少模块名/日志名变动面）。
- `cli_options.py` 只收敛运行参数组；config/input/host 参数未统一（入口间有真实差异，留待步骤 7 与 YAML 映射一起处理）。
- 旧路径转发壳含 `__main__` 块（`python -m bcg.run` 兼容），`[project.scripts]` 已切换。
- setup.py 模板路径修复为包根定位（`__file__` 迁移副作用）。

验收（全部通过）：

- 所有新旧 CLI 冒烟通过（`uv run bcg`、`python -m bcg.run/server/driver`、旧 import 用法）。✅
- `import bcg` 不导入 `bcg.apps`（契约测试）。✅
- app 模块不被 SDK 核心反向依赖（core 不导入 apps，架构测试覆盖）。✅
- 139 passed、ruff 全绿。✅

### 步骤 7：经确认后统一默认值并移除 env 副作用（已完成）

开始前由团队形成简短 ADR，确认：

- `incremental_merge_threshold` 最终值 —— **0.8（SDK 值，用户确认）**
- `context_chars` / `io_context_chars` 最终值 —— **9000（SDK 值，用户确认）**
- `verify_merge` 最终值 —— **False（SDK 值，用户确认）**
- server host/port —— 保持 127.0.0.1:8848（无冲突，未变更）
- 默认 backend 和模型选择/fallback 规则 —— 保持 api_based / gpt-5.5（无冲突，未变更）
- 用户级与项目级配置的最终优先级 —— 按步骤 4A 实现（项目 > 用户），未变更

然后：

1. 把最终值写入 `defaults.yaml`，删除 legacy profiles 和代码字面量默认值。✅
2. CLI `--help` 从有效 settings 展示默认值。✅（`cli_options` 从 defaults.yaml 读取）
3. 删除 `env.py` 的 import-time `load_project_env()`；由 CLI bootstrap 或显式 API 调用。✅
4. 更新迁移说明、release note 和所有行为契约测试。✅（golden 重新捕获、env 契约更新）

实际做法与计划的差异：

- CLI 三入口现通过 `resolve_runtime_config` 合并项目/用户/环境/显式 YAML settings；显式 CLI 参数优先，未发现 YAML 时回退旧 JSON 路径。
- `run_merge_pass` 函数签名默认值（0.86）作为隐藏第三处默认一并统一为 0.8（调用方总是显式传参，实际行为不变）。
- 7 个 CLI 入口统一加 `_bootstrap_env()`（含 python -m 直跑路径），非仅根 CLI。

验收（全部通过）：

- 三 CLI 入口 `--help` 默认值一致且与 defaults.yaml 同源（golden 0.8/False/9000）。✅
- `import bcg` 与 `import bcg.env` 均不加载 env；显式加载可用。✅
- 140 passed、ruff 全绿；行为变更单提交。✅

### 步骤 8：治理、兼容窗口和收尾（已完成）

1. 版本号以 package metadata / pyproject 为规范来源；旧 `__version__` 保留兼容读取或发出弃用提示。✅
2. 逐目录消除 Ruff 告警，再移除 `extend-exclude`。✅
3. 引入统一异常层级；日志替换 `print()` 时保持 CLI stdout/stderr 契约。✅（层级建立；`print()` 未替换，保持契约，留待后续）
4. 为 artifact 添加 schema version；如果格式发生变化，提供独立 migration。✅（memory document 已有 `schema: bcg.memory.v2`，格式未变）
5. 验证 `py.typed` 随 wheel 发布，而不是重复创建。✅（wheel 内容测试）
6. 测试文件可按领域整理，但不把"一一镜像源码目录"作为硬性要求。✅（保持领域文件）
7. 旧 import、CLI 和 JSON 兼容层至少保留一个已发布版本；只在后续明确的 breaking release 删除。✅（转发壳 + 迁移 adapter 保留）

最终验收：

- `uv run pytest` 全绿（142 passed）。✅
- `ruff check bcg tests` 全绿且核心目录无 exclude。✅
- 架构测试证明依赖方向（test_step1_* + 契约套件）。✅
- 双后端关键 golden artifacts 与批准后的预期一致（normalized golden + entrypoint defaults）。✅
- 新 wheel 包含 YAML defaults/example 和 `py.typed`（wheel 内容测试）。✅
- 文档列出的旧入口在兼容窗口内仍可用（legacy module help / import 契约测试）。✅

---

## 6. 提交与共存纪律

### 6.1 一份实现，多条兼容路径

允许短期共存的只有：

- 旧 Python 模块路径到新模块的 re-export。
- 旧 CLI 入口到新 main 的转发。
- 旧 JSON 到统一 settings schema 的只读迁移 adapter。

这些兼容层不得复制业务逻辑，并必须注明删除版本。除此以外，同一行为不保留两份可运行实现。

### 6.2 小提交顺序

```text
补契约测试
-> 新建目标模块/接口
-> 移动或抽取一项职责
-> 切换调用方
-> 保留薄兼容层
-> 跑单测、全量测试、lint、golden
-> 提交
```

一个阶段可以由多个可审查提交组成，不要求把数千行迁移压成单 commit。每个提交必须可测试、可回滚，不跨提交保留两套业务实现。

### 6.3 禁止事项

- 禁止在 refactor-only 提交中改变默认值、公式、artifact 字段或 CLI 语义。
- 禁止仅因文本相似就合并两个 policy。
- 禁止用 backend flag 堆叠代替清晰 adapter。
- 禁止 loader 静默合并新旧配置。
- 禁止把 API key 写入 YAML、artifact、warning 或测试 fixture。
- 禁止删除旧公共路径而不经过发布兼容窗口。

---

## 7. 可量化验收指标

| 维度 | 检查方法 | 要求 |
|---|---|---|
| 基线 | `uv run pytest` | 步骤 0 后全绿 |
| 行为 | normalized golden artifact diff | refactor-only 提交无变化 |
| 公共 API | import/CLI contract tests | 兼容窗口内全部通过 |
| 依赖方向 | import-linter / architecture test | core 不依赖 apps/具体后端 |
| 配置 | schema/precedence/migration tests | 默认值单一来源，错误立即失败 |
| secrets | fixture + log scan | 无 API key 落盘或输出 |
| lint | Ruff | 排除项逐阶段减少，最终核心无 exclude |
| 复杂度 | 单函数复杂度/职责审查 | 上帝类被拆分，无新增巨型协调函数 |
| 打包 | build wheel + inspect | YAML、`py.typed` 和入口正确 |

重复行数和文件行数仅作为趋势数据，不作为否决合理后端差异的硬指标。

---

## 8. 主要风险与决策点

1. **默认值冲突**：必须由团队确认，不能以“重构”为由擅自选边。
2. **置信度策略差异**：先共享数学基础，policy 是否合并由 golden tests 和领域语义决定。
3. **artifact 兼容**：下游 agent-cli、dashboard、benchmark 依赖现有格式，结构变化必须有 schema version 和 migration。
4. **环境加载变化**：显式 env bootstrap 会影响直接使用 SDK 的调用方，需 release note 和独立测试。
5. **配置层级**：用户级、项目级和显式配置的覆盖顺序一旦发布即成为公共行为。
6. **重型依赖**：移动 apps 不会自动缩小 wheel；是否拆 optional dependency 或独立发行包需单独评估。
7. **兼容层生命周期**：以发布版本为单位淘汰，不按运行次数或重构分支存在时间计算。

---

## 9. 建议 PR 序列

1. `baseline: fix tests and add behavior contracts`
2. `architecture: add core contracts and backend adapters`
3. `refactor: extract pure shared construct primitives`
4. `refactor: split session, writer and graph mutation responsibilities`
5. `config: add typed YAML loader and legacy migration adapters`
6. `refactor: move SDK implementation under bcg.core with shims`
7. `refactor: move apps and centralize CLI option mapping`
8. `behavior: adopt approved defaults and explicit env loading`
9. `governance: remove lint exclusions and finalize compatibility policy`

每个 PR 描述必须包含：行为类型、影响的公共边界、测试证据、回滚方式和兼容层删除版本。

---

## 第二部分：仓库级重构（步骤 9-15 已完成，步骤 11 结项，步骤 16 待执行）

### 10. 审计结论与当前基线

#### 10.1 已覆盖和未覆盖的范围

对 `refactor-baseline-20260804..4a1927b` 的变更审计表明，第一部分主要修改了
`bcg/`、Python tests、`pyproject.toml` 和 `uv.lock`。以下区域尚未进行系统重构：

| 区域 | 当前职责 | 主要问题 |
|---|---|---|
| `agent-cli/` | 参考 Agent、CLI、TUI、模型适配和 BCG 上下文接入 | 大型协调文件、手写 HTTP 契约、workspace 测试依赖预构建产物 |
| `dashboard/` | Vite 图内存界面 | 单文件实现、假定但尚未实现的 API、私有地址硬编码、无测试和 CI |
| `dashboard/bcg_viewer/` | 历史 artifact 回放和运行控制器 | 第二套 UI/后端；引用已不存在的 `bcg-construct`、`scripts/start.sh` 和机器路径 |
| `deploy/` | TongGraph 服务配置 | 数据目录写死为另一台开发机路径，环境覆盖链路不完整 |
| `scripts/` | TongGraph、vLLM、SGLang 启动 | 参数/环境规则分散，含本机可执行文件 fallback，缺少自动化验证 |
| 根工程文件 | 安装、Make、CI、贡献和发布说明 | `make test` 只测 Python；CI 不安装或构建 Dashboard；贡献指南与 CI 命令不一致 |

`bcg/construct/` 和 `bcg/benchmark/` 已在第一部分建立内部边界，但它们仍是跨组件
契约的生产者或消费者，因此第二部分会验证接口和 artifact，不重新改写算法。

#### 10.2 2026-08-06 实测基线

- `uv run pytest -q`：167 passed，3 个预期的 legacy config deprecation warnings。
- `npm --prefix agent-cli run build`：通过。
- `npm --prefix agent-cli test`：必须先 build；按该顺序为 3 files、20 tests passed。
- `npm --prefix dashboard ci --ignore-scripts`：通过。
- `npm --prefix dashboard run build`：通过；当前没有 lint、unit test 或 E2E script。
- `bash -n install.sh scripts/start_tonggraph_server.sh scripts/start_sglang_server.sh scripts/start_vllm.sh`：通过。
- `npm audit --omit=dev`：Dashboard 生产依赖为 0 个漏洞；Agent 报告 3 个 high severity，
  涉及 `brace-expansion` 和 `undici`。依赖升级作为独立行为变更处理，不执行盲目 `--force`。
- 工作区在依赖安装和构建后仍无受跟踪文件变更；`node_modules/` 和 `dist/` 为生成物。

该基线只证明现有自动化覆盖的路径可用，不等于全仓行为已被充分保护。
#### 10.3 步骤 9 完成记录（2026-08-06）

- 根 Makefile 已提供 `make test-python`、`make build-agent`、`make test-agent`、`make build-dashboard`、`make check-shell`、`make check-repository` 和 `make check`。
- Agent `npm test` 增加 `pretest`，在 workspace `dist` 缺失时自动构建内部包；干净 workspace 验证为 3 files、20 tests passed。
- CI 已拆分 Python、Agent、Dashboard、Repository 和 Packaging jobs；Dashboard 构建和 shell/卫生检查进入 required gates。
- `uv run ruff format .` 补齐既有 format gate，48 个文件只做机械格式化；随后 `make check` 全部通过：Python 167 passed、Agent 20 passed、Dashboard build、shell、卫生检查均通过。
- `uv build` 和 `npm pack ./agent-cli --dry-run` 均通过；未访问外部模型或真实服务。

#### 10.4 已核实的架构风险

1. **跨语言契约没有规范来源。** Agent 在
   `agent-cli/src/core/context/bcg-context.ts` 中自行声明 `/turns`、`/release` payload、
   snapshot 解析和错误语义；Python 服务没有向 TypeScript 消费者发布可验证 schema。
2. **Dashboard 的两套实现不是简单的新旧皮肤。** Vite Dashboard 尝试
   `/api/memory/graph`、`/api/graph`、`/api/memory/context` 等尚未落地的接口；旧 Viewer
   读取 JSONL/artifact 并提供运行控制 API。删除任一套之前必须先明确产品能力和数据源。
3. **机器相关默认值进入源码。** Dashboard 包含 `172.25.10.2`，TongGraph 配置含
   `/data/user/fukeshu/...`，启动脚本含 `/data/user/baijun/...` fallback，旧 Viewer 还含另一台
   机器的 `start.sh` 路径。这些值不能作为发布默认值。
4. **协调模块过大且职责混合。** `interactive-mode.ts` 6034 行、`agent-session.ts`
   3334 行、`session-manager.ts` 1712 行、`settings-manager.ts` 1300 行；Vite `main.ts`
   1912 行，旧 Viewer HTML 2611 行、Python server 925 行。行数不是拆分标准，但这些文件
   同时承担协议、状态、持久化和 UI 协调，已形成高变更耦合。
5. **仓库级质量门不完整。** 根 `make test` 只运行 pytest；CI 测 Python 和 Agent，
   不安装/构建 Dashboard，不检查 shell，不做跨语言 contract test，也不验证安装包组合。
6. **测试命令隐含顺序。** Agent workspace 包入口指向 `dist`，干净 checkout 直接
   `npm test` 无法解析内部包；只有先 build 才能测试。
7. **发布边界未定义。** Python 为 1.0.0，Agent/workspace/Dashboard 为 0.1.0；版本不同
   本身不是错误，但必须明确是独立版本还是兼容矩阵。

### 11. 第二部分目标、边界和执行纪律

#### 11.1 目标

1. 每个可交付组件都有明确 owner、输入输出、独立构建命令和最小测试门。
2. Python 服务、Agent 和 Dashboard 使用版本化、机器可校验的 HTTP/artifact 契约。
3. 根目录提供一组与 CI 一致的全仓命令，能区分 unit、contract、build 和可选 E2E。
4. 本机路径和网络地址全部退出发布配置，运行环境通过显式配置注入。
5. 新旧实现只通过薄 adapter 共存；旧 Viewer、Python forwarding modules 和 legacy config
   adapter 都必须满足删除门槛后再移除。
6. 安装、开发、部署和发布文档与实际命令一致，并可在干净环境复现。

#### 11.2 非目标

- 不在结构提交中改变置信度、图构建算法、Agent prompt 或上下文淘汰策略。
- 不要求 Python、Agent 和 Dashboard 使用相同版本号；先通过 ADR 确定版本策略。
- 不把 Dashboard 的假定 API 直接当作 Python 服务既定公共 API。
- 不为追求小文件机械拆分 cohesive 逻辑，也不一次性重写 Agent 或 Viewer。
- 不把需要 GPU、外部模型或真实凭据的 E2E 设为普通 PR 的强制本地门槛。
- 不提交 `.env`、模型、数据集、运行输出、`node_modules` 或构建产物。

#### 11.3 新旧代码共存和回滚

- 旧代码参考点使用 Git tag `refactor-baseline-20260804`，不复制第二棵源码目录。
- 每个步骤开始前记录 HEAD、基线命令和受影响的公共边界；每个提交保持可单独回滚。
- 新模块接管职责后，旧入口只能转发到新实现，禁止两份实现同时写同一状态或 artifact。
- 涉及持久化格式时，先增加 reader/contract test，再切 writer；必要时提供双读单写迁移期，
  禁止双写两个无事务保障的数据源。
- 结构迁移和行为变更分开提交。依赖安全升级、默认值变化、接口版本升级均属于行为变更。
- 阶段失败时回滚本阶段提交，不删除已经证明可用的旧路径；不得用复制旧目录作为同步机制。

#### 11.4 依赖顺序

~~~text
步骤 9  全仓基线与质量门
  -> 步骤 10 版本化跨组件契约
       -> 步骤 11 Agent 模块化 --------┐
       -> 步骤 12 Dashboard 收敛 ------+-> 步骤 14 安装、构建与发布治理
       -> 步骤 13 部署与脚本收敛 ------┘
                                              -> 步骤 15 文档与仓库治理
                                                   -> 步骤 16 兼容层删除
~~~

步骤 11、12、13 在步骤 10 后可由不同 PR 并行，但不得各自发明契约。步骤 16 永远最后执行。

### 步骤 9：建立全仓基线和质量门（已完成）

**目的：** 先让仓库能够诚实地回答“哪些组件被验证过”，不改运行时业务行为。

1. 在根 Makefile 增加明确目标：`test-python`、`build-agent`、`test-agent`、
   `build-dashboard`、`check-shell`、`check`；`test-agent` 自身处理所需预构建或改为从源码解析。
2. 将 CI 拆为可定位失败来源的 Python、Agent、Dashboard 和 packaging jobs；缓存键分别基于
   `uv.lock`、`npm-shrinkwrap.json` 和 `dashboard/package-lock.json`。
3. Dashboard 至少加入 TypeScript build gate；测试框架在步骤 12 添加测试时再启用，不提交空壳测试。
4. 对 install/shell 脚本增加 `bash -n`；评估 ShellCheck 后再启用规则，先记录必要例外。
5. 增加 generated/ignored 文件断言，防止 `dist`、outputs、密钥和本机配置进入提交。
6. 将 CONTRIBUTING 的本地命令改为与 CI 完全一致；删除仍使用 unittest 的过期说明。

**验收：** 干净 checkout 上一个根命令可运行所有无外部服务的 required checks；CI 明确构建
Dashboard；Agent test 不再依赖开发者记住隐含顺序；不产生受跟踪生成物。

**回滚：** 质量门按组件独立提交。若新门暴露既有失败，先记录 baseline/临时 non-blocking，
不得通过放宽已有 Python 或 Agent 断言来换取全绿。

### 步骤 10：定义版本化跨组件契约（已完成）

**目的：** 在拆 Agent、Dashboard 或服务之前固定跨组件边界。

1. 在顶层 `contracts/`（或经 ADR 确认的等价位置）记录服务拥有的 schema：
   `/health`、`/turns`、`/release` 请求/响应、错误 envelope 和版本协商方式。✅
2. 分开定义 HTTP snapshot、持久化 memory document、stream JSONL 三类 schema；不得因为字段
   看起来相似就合成一个万能 Graph 类型。✅
3. 由 Python producer contract tests 验证真实响应；TypeScript 使用生成类型或由 CI 校验的
   类型映射，替换 Agent 中无校验的 type assertion 和任意 snapshot fallback。✅
4. 为兼容性写规则：只加 optional 字段为向后兼容；删除、改义或改默认值需新版本和 migration。✅
5. 明确 BCG 配置到 Agent 设置的映射、优先级和敏感字段边界；URL 默认值只能有一个规范来源。✅
6. 建立最小跨语言 contract fixture：无需模型，使用确定性 fake backend 验证 Agent client 与
   Python handler 对同一请求/响应达成一致。✅

实际做法与计划的差异（均不影响验收结论）：

- 契约格式选 **JSON Schema（draft 2020-12）** 而非 OpenAPI：服务是极简 `http.server`（无框架），
  JSON Schema 直接做语言无关规范来源；Python 用 `jsonschema` + `referencing` registry 校验，
  TS 用**生成类型**（编译期）+ CI freshness 检查，未引入运行时 schema 校验依赖。
- 三类 schema 分开定义（`http.schema.json` / `memory-document.schema.json` / `stream.schema.json`），
  node/relation 子结构通过**跨文件 `$ref`** 复用（referencing Registry），不复制、也不合成万能类型。
- 版本协商：`GET /health` 新增 `schema_version` 字段（server 行为变更，向后兼容）；常量
  `bcg.core.contracts.HTTP_SCHEMA_VERSION` 与 schema 文件版本双向断言；TS 侧 `BCG_SCHEMA_VERSION`
  由生成器产出。**版本守卫** `contracts/check_schema_version.py`：版本化契约文件内容变更但
  `schema_version` 未递增时 CI 失败（本地未提交改动同样生效）。
- 配置映射与敏感边界记录在 `contracts/README.md`；`contracts/defaults.json` 与
  `bcg/config/defaults.yaml` 的 server 域一致性有测试（URL 默认值单一规范来源）。
- Agent 契约修复：删除 `formatBcgMarkdown` 的 `forward_relations` 死分支（Python 从不产出，
  靠 fallback 掩盖的契约漂移）；`parseSnapshot` 容错逻辑保留，类型断言改为生成类型。
- 测试增量：Python **180 passed**（+9 HTTP producer、+4 artifact）；Agent **23 tests**（+3 fixture）。
- `make check-contracts`（生成 freshness + 版本守卫 + 契约测试）进入 `make check` 与 CI Python job。

验收（全部通过）：

- schema 有版本且随仓库发布。✅
- Python producer、TS consumer、fixture 三方测试通过。✅
- contract diff 能在 CI 阻止未声明的 breaking change（版本守卫 + 生成 freshness）。✅
- 日志和 fixture 不含凭据。✅
- 跨语言 fixture 双边消费（`contracts/fixtures/`）。✅

#### 步骤 11 执行记录（2026-08-06，进行中）

- **11.1 包测试入口与 export contract（已完成）**：三个 workspace 包
  （bcg-agent-core / bcg-ai / bcg-tui）各加 `pretest`（build）+ `test`（vitest）
  脚本和 export-contract 测试；三者均确认可独立发布（无 shell 依赖），
  bcg-ai 根入口保持无副作用（不泄漏 generated catalogs）。
- **11.2 BCG HTTP 收敛（已完成）**：`src/core/context/bcg-client.ts` 成为
  contract-backed client（URL 组装、AbortSignal 组合、错误信封消费、
  latest[problemId] 解析、release 幂等）；`BcgContextManager` 只保留上下文
  窗口策略（发送哪些 turn、上限、markdown 渲染、降级）。错误消息现包含
  服务器 error envelope 文本。
- **11.4 session schema golden/migration（已完成）**：v3 golden fixture +
  v1/v2 迁移测试（parentId 链、firstKeptEntryIndex→firstKeptEntryId、
  hookMessage→custom、幂等、round-trip）；**修复真实兼容 bug**：
  `loadEntriesFromFile`/`parseSessionHeaderCandidate` 原要求 header 带 id，
  导致 v1 文件永远无法加载（迁移形同虚设），现仅要求 `type === "session"`。
- **11.4 分解（部分）**：`_expandSkillCommand` 抽为
  `src/core/skill-expansion.ts`（纯函数 + 注入 getSkills/emitError），5 个
  定向测试；`_rebuildSystemPrompt` 已确认委托 `buildSystemPrompt`（组装已分离）。
- **11.5 默认 URL 单一来源（已完成）**：`DEFAULT_BCG_GRAPH_URL` 常量 +
  契约测试断言与 `contracts/defaults.json` 一致（Python defaults.yaml ↔
  contract ↔ agent 三方同步）。
- **11.6 生产依赖安全升级（已完成）**：undici 8.5.0→8.10.0（调用面仅
  Dispatcher/Client/Pool，已确认兼容）、minimatch 10.2.6、brace-expansion
  override 5.0.8→5.0.9；`npm audit --omit=dev` 0 漏洞；build/46 tests/
  CLI smoke 全过。
- **11.7 定向测试（部分）**：BcgClient 5 个（信封解析、错误信封、release
  404 幂等、released 标志、AbortSignal）；session 恢复（open + 迁移）；
  既有错误降级/turn limit/release-once 测试确认覆盖。
- **11.3 interactive-mode.ts 分解（进行中）**：6034 → 4946 行（-1088），
  已抽三个纯/可参数化职责模块，各带定向测试：
  1. `display-format.ts`：17 个路径/scope 显示格式化函数（home 缩短、
     node_modules 相对化、package 标签、分组、诊断渲染），13 个测试；
  2. `login-provider-options.ts`：登录/登出 provider 选项构建（按 auth
     类型过滤、排序、configured 状态标记、id/name 匹配），5 个测试；
  3. `autocomplete-source.ts`：autocomplete 源标签（u/p/t + npm/git）、
     描述前缀、内建命令冲突诊断（内置命令列表参数化），7 个测试。
  4. `resources-sections.ts`：showLoadedResources 的区块组装
     （Context/Skills/Prompts/Extensions/Themes 区块 + 四类诊断块），
     类内只保留数据收集与容器渲染，5 个测试。
  5. `auth-selectors.ts`：登录/登出选择器三流程（auth-type 选择、provider
     选择、logout）为注入 deps 的 flow（showSelector 原语、provider 选项、
     登录/登出后端、状态/错误上报），7 个测试（组件 mock 捕获构造）。
  6. `session-selectors.ts`：/sessions 选择器流程（列表/恢复/重命名/关闭
     注入），4 个测试。
  7. `model-selectors.ts`：/model 与 /models（作用域选择 + 设置持久化）
     flow，deps 注入 UI/模型后端，5 个测试。
  8. `auth-dialogs.ts`：登录/登出对话框簇（showLoginDialog、
     showApiKeyLoginDialog、showAmbientAuthDialog、showAuthSelect/Prompt、
     loginProvider、notifyAuthDialog、completeProviderAuthentication），
     登录后默认模型选择逻辑可直接测试，5 个测试。
  9. `trust-selectors.ts`：/trust 与 fork 消息选择器（trust store 经
     getSavedDecision/saveTrust 注入，flow 无文件系统依赖），5 个测试。
  10. `tree-selector.ts`：/tree 分支导航 flow（摘要选择循环、escape 临时
     处理器安装/恢复、剪贴板复制），5 个测试。
  11. `session-info.ts`：/session 统计文本组装（消息/token/成本/缓存命中
     纯文本），4 个测试。
  12. `command-args.ts`：/export、/import 路径参数与 /name 参数解析
     （纯函数），7 个测试。
  13. `message-utils.ts`：用户消息文本转换，3 个测试。

**11.3 结项**（6034 → 4946 行，-1088；13 个模块 + 68 个定向测试）：
协议（BcgClient）、认证（选项/选择器/对话框/登录后处理）、模型选择、
session/tree/trust/fork 选择器、资源区块、autocomplete、命令参数、
消息转换、session 统计等**纯逻辑与可注入 flow 全部抽离**。剩余
~4900 行为有状态 UI 协调（扩展 widget 组 18 方法、状态指示、键位/
订阅、onSubmit 命令 if 链、SVG/渲染）——按文档 1.2 原则接受现状
（抽取仅搬家、测试全 mock、无实际可维护性收益），不再作为 11.3 目标。
- 测试基线：Python 180 passed；Agent **121 tests**（24 个文件，步骤 11
  累计 +71：包 export 3、fixture 3、client 5、session schema 5、
  skill-expansion 5、URL 契约 1、display-format 13、login-options 5、
  autocomplete 7、resources-sections 5、auth-selectors 7、
  session-selectors 4、model-selectors 5、auth-dialogs 5、trust-selectors 5、
  tree-selector 5、session-info 4、command-args 7、message-utils 3）；
  `make check` 全绿。

### 步骤 11：重构 `agent-cli/`

**目的：** 在保持 CLI、session 文件和用户设置兼容的前提下，拆开协议、领域状态和 TUI。

1. 先为三个 workspace 包补各自的测试入口和 public export contract；明确哪些包可独立发布。
2. 将 BCG HTTP 访问收敛为 contract-backed client；`BcgContextManager` 只负责上下文窗口策略，
   不再兼任协议解析和容错格式猜测。
3. 分解 `interactive-mode.ts`：命令注册、Graph 状态、模型/登录 flow、session UI 和渲染生命周期
   按既有组件边界迁移，每次只移动一个有测试的职责。
4. 分解 `agent-session.ts` 和 `session-manager.ts`：运行编排、消息转换、持久化、迁移和资源释放
   分离；session schema 先加 golden/migration tests。
5. `settings-manager.ts` 保留 schema/默认值/读写三层，BCG 设置复用步骤 10 的映射；不改变现有
   `~/.bcg/agent` 和 `~/.bcg/config.json` 的兼容行为。
6. 将生产依赖安全升级独立提交：先确认 `brace-expansion` 和 `undici` 实际调用面与上游兼容，
   更新 shrinkwrap 后运行 build、unit、contract 和 CLI smoke；禁止盲目 `npm audit fix --force`。
7. 给超时、AbortSignal、部分响应、release 幂等、session 恢复和错误降级增加定向测试。

**验收：** `npm ci` 后单个规范命令可测试；现有 20 tests 与新增测试通过；build 产物和 bin 名不变；
旧 session/settings fixtures 可读；Python fake server contract test 通过；生产依赖审计结果被解决或有
带期限、owner 和影响面的例外记录。

**回滚：** 按职责小提交；旧模块在迁移期只做 re-export/delegation。任何 session writer 切换都先
证明新 reader 能读取旧 fixture，不允许同时维护两套 session 状态机。

### 步骤 12：收敛 `dashboard/` 和旧 Viewer

**目的：** 明确一个正式前端，同时保留 artifact 回放能力，避免先删掉唯一可用功能。

1. 先制作功能矩阵：实时 Graph、artifact/JSONL 回放、目录导入、运行控制、timing、subgraph、
   Memgraph/TongGraph 连接。确定哪些属于正式 Dashboard，哪些属于开发工具。
2. 以 Vite 应用作为正式前端候选，但在 ADR/功能矩阵确认前不删除 `bcg_viewer/`。
3. 为 live HTTP、artifact replay、sample 三种数据源建立明确 adapter；所有输入先通过步骤 10 schema
   校验，不再依次猜测三个不存在或不同语义的 URL。
4. 把 `main.ts` 拆为 data client、normalizer、state、layout 和 UI components；先迁移纯函数并补单测，
   再迁移 DOM/交互。旧 Viewer 的 Python run control 若仍需要，移到明确的 devtool 服务并限制命令输入。
5. 移除 `172.25.10.2` 和所有个人路径；开发代理、Graph/Memgraph 地址通过 `.env.example` 和运行配置注入。
6. 增加 Vitest 单测（normalizer/layout/client）、DOM 交互测试、生产 build；关键桌面/移动 workflow 的
   Playwright E2E 可由负责 E2E 的开发者执行，但测试场景和 fixture 必须在本步骤定义。
7. 达到功能等价后，将旧 Viewer 标为 deprecated；至少跨一个发布版本后才进入步骤 16 删除。

**验收：** Dashboard 不依赖 sample 才能启动真实模式；错误状态不会静默伪装成 sample 成功；无私有地址；
schema/unit/build 进入 CI；旧 Viewer 功能矩阵每项有“迁移、明确废弃或保留为 devtool”的结论。

**回滚：** data adapter、纯函数、UI 分开提交。新 Dashboard 未通过功能矩阵前继续保留旧 Viewer 为只读
参考；禁止两个 UI 同时写同一个运行状态。

### 步骤 13：收敛 `deploy/`、`scripts/` 和运行时配置

**目的：** 使服务脚本可移植、可审计，且与 `.env.example`、setup 和文档一致。

1. 修复 TongGraph 配置链：删除 `/data/user/fukeshu/...`，确保 `TONGGRAPH_DATA_DIR` 真正进入服务配置；
   删除 `/data/user/baijun/...` 可执行文件 fallback，改为显式 bin 或 PATH。
2. 为端口、host、模型、日志、数据目录建立环境变量清单，标记 required/default/secret；检测端口冲突和
   无效数值时快速失败。
3. vLLM 与 SGLang 共享的纯 shell 校验可抽取小型 helper；引擎特有参数和生命周期保持独立，避免一个
   布满 backend flag 的通用脚本。
4. 为脚本增加 `--help`、dry-run/command rendering 或可注入 executable，以便无 GPU 单测参数映射；
   start/stop 只操作经过端口和进程特征双重确认的目标。
5. 对 deploy YAML 增加 schema/解析 smoke，验证 token 只通过环境读取；记录生产部署仍需的外部组件。

**验收：** 任意 checkout 路径可启动到依赖检查阶段；仓库无个人绝对路径；shell syntax/static checks 通过；
dry-run golden 覆盖默认值和 override；TongGraph 数据目录 override 被实际消费。

**回滚：** 每类服务脚本独立提交；默认值变化必须单独列为 behavior change。保留旧环境变量 alias 一个
发布周期，但最终只映射到同一内部配置，不复制启动逻辑。

#### 步骤 13 执行记录（2026-08-06，已完成）

- **TongGraph 配置链（文档 1）**：`scripts/start_tonggraph_server.sh` 删除
  `/data/user/baijun/...` 可执行文件 fallback（仅 PATH / `TONGGRAPH_SERVER_BIN`）；
  启动时把 `TONGGRAPH_DATA_DIR` 注入运行时配置副本
  （`var/tonggraph-config.generated.yml`），override 被服务实际消费；
  `deploy/tonggraph-server.yml` 变可移植模板（相对 `./var/tonggraph` 默认）。
- **个人路径清理**：`bcg/construct/light/stance.py` 的
  `/data/user/wenxinyi/...` 默认模型路径移除（空默认，按部署配置）；
  `check_repository_hygiene.sh` 新增 git grep 检查，tracked 文件含
  `/data/user/` 即失败（防回归）。
- **共享 helper 与清单（文档 2、3）**：`scripts/lib/common.sh`
  （`bcg_load_root_env`/`bcg_require_env`/`bcg_validate_port`/
  `bcg_check_port_free`/`bcg_maybe_dry_run`）；三脚本统一 env 加载 + 端口
  校验 + 必填快速失败；`scripts/README.md` 为环境变量清单
  （required/default/secret + 兼容 alias + 外部生产组件）。
- **dry-run 与 golden（文档 4）**：三脚本支持 `--dry-run` 渲染精确命令；
  关键实现细节——sglang 的运行时检查（`import sglang`、进程探测）移到
  渲染之后，否则 dry-run 也会挂；`scripts/test_scripts.sh` 15 个 golden
  断言（默认/override/无效端口/缺 token/配置注入），`make check-scripts`
  进 check 链。
- **deploy YAML smoke（文档 5）**：`scripts/check_deploy_yaml.py` 校验结构、
  端口范围、token 仅 `token_env` 引用、无内联凭据/个人路径。
- `.gitignore` 的 `lib/` 规则加 `!scripts/lib/` 例外。
- 测试/检查：make check 全绿（Python 180、Agent 87、contracts、shell、
  scripts golden、hygiene）。

#### 步骤 12 执行记录（2026-08-06，已完成）

- **Vitest 基建 + normalizer/layout 抽取**：dashboard 加 vitest；`src/types.ts`、
  `src/normalize.ts`（normalizeAnyGraph/normalizeNode/normalizeEdge/
  normalizeMemory + helper）、`src/layout.ts`（applyLayout/original/layered/
  star）从 main.ts（1912 → 1583 行）抽出；测试以
  `contracts/fixtures/turns-response.json` 为输入。
- **修复真实契约漂移（测试暴露）**：normalizeNode 原不认识契约的顶层
  `node_type`/`belief`/`confidence` 字段（节点全变 Claim、标签变 id）；
  normalizeEdge 不认识 `from_id`/`to_id`（契约关系边全部被过滤丢失）——
  已接线并加回归测试。
- **三数据源 adapter（文档 3）**：`src/data-sources.ts`——`loadLiveGraph`
  （契约端点 `GET /graph?problem_id=`，经 `/health` 解析 active 会话）、
  `loadArtifactReplay`（memory document/JSONL）、`sampleMemory`（原样保留）。
  `loadGraphFromApi` 不再猜测 3 个不存在的 URL；live 错误明确抛出，
  绝不伪装成 sample 成功（有测试）。
- **私有地址清理（文档 5）**：main.ts 的 Memgraph 默认
  （bolt://172.25.10.2:7687 等）→ localhost + env 覆盖；vite proxy 的
  `172.25.10.2:23456` → `VITE_BCG_API_URL`；`dashboard/.env.example` 新增。
- **功能矩阵（文档 1、2、7）**：`dashboard/README.md` 记录矩阵与结论
  （live/artifact/timing 迁移到 dashboard；目录导入、Memgraph 保留为
  devtool；run control/subgraph 不在范围）；旧 `bcg_viewer/` 标记
  deprecated（只读参考，步骤 16 窗口删除）。
- **CI（验收）**：`make check` 与 CI 增加 `test-dashboard`（vitest）。
- 测试/检查：Dashboard 15 tests（normalizer/layout/data-sources）；
  make check 全绿。
- 剩余（后续步骤）：main.ts 的 state 对象与 SVG/DOM 渲染层（~1500 行）
  拆分；artifact replay 的 UI 文件选择器。

### 步骤 14：统一安装、构建和发布治理

**目的：** 保证用户安装到的是经过验证的 Python/Agent/Dashboard 组合。

1. 写 ADR 决定版本策略：组件独立 semver + compatibility matrix，或 lockstep release；不因数字不同自动统一。
2. 明确 Dashboard 是发布 artifact、独立部署，还是仅开发工具；据此决定是否进入 `make install` 和发布包。
3. 将 `install.sh`、Makefile、README 和 CI 统一到相同的 locked install/build 顺序；安装器继续使用临时目录，
   增加下载失败、PATH 和部分安装失败说明。
4. 增加 clean wheel、Agent npm pack/bin smoke；若 Dashboard 发布则增加静态 bundle artifact smoke。
5. 在干净容器/临时 HOME 做安装后 smoke：`bcg --version`、`bcg-agent --version/help`、资源文件和配置首次创建；
   不访问真实模型，不写开发者 HOME。
6. 生成 release manifest，记录 Python version、Agent version、contract version 和可选 Dashboard version。

**验收：** 一次 release 可追溯到唯一组件组合；锁文件无未提交漂移；干净环境安装和卸载路径有测试证据；
失败不会留下被误认为完整安装的组合。

**回滚：** packaging 与业务实现分离提交；发布失败回退 manifest/installer，不回退已验证的内部重构。

#### 步骤 14 执行记录（2026-08-06，已完成）

- **ADR-0001（docs/adr/）**：lockstep 主版本 + release manifest（patch 可
  独立推进）；Dashboard 为发布 artifact、独立部署（不进 install.sh /
  make install）。agent-cli 与 dashboard 版本对齐 1.0.0。
- **release manifest**：`scripts/release-manifest.py`（--check 模式）输出
  release-manifest.json（python/agent/dashboard 版本 + contract schema
  版本 + lockfile-clean），锁文件漂移即失败。
- **安装后 smoke**：`scripts/test_install_smoke.sh`——临时 HOME 下验证
  `bcg --version`、`bcg config show`、打包 defaults 可达、只读命令不提前
  创建 `~/.bcg`、`bcg-agent --version/help`。
- **统一安装顺序**：`make install-tool` 修复为先 build agent 再
  `npm install -g`；README 安装章节以表格统一两个 lockstep 路径
  （install.sh 发布 / make install 源码），Dashboard 部署指引指向
  dashboard/README。
- **部分安装失败说明**：install.sh 的 npm 全局安装失败时明确提示
  "bcg 可能已装，可重跑或 make install-tool"。
- **CI**：packaging job 增加 `make check-release`（manifest 一致性 +
  install smoke）；既有 wheel 内容测试与 npm pack inspect 保留。
- 验收核对：release 可追溯到唯一组件组合（manifest）✅；锁文件无未提交
  漂移（manifest --check）✅；干净环境安装/卸载路径有测试证据
  （install smoke + wheel 测试）✅；失败不留下被误认为完整安装的组合
  （install.sh 失败语义 + PATH/部分安装提示）✅。

### 步骤 15：文档、benchmark、artifact 和仓库治理

**目的：** 让仓库说明反映真实架构，并收紧长期维护边界。

1. README 架构图和 Quick Start 以实际组件/命令为准；为 Agent、Dashboard、部署分别提供最小维护文档。
2. CONTRIBUTING 列出 required checks、可选 E2E、外部服务测试和安全升级流程，禁止声称“同 CI”但命令不同。
3. 为 benchmark 明确只读消费的公共 API/artifact schema；使用固定小 fixture 做 smoke，不把受限数据集提交仓库。
4. 记录 outputs/artifacts 的 schema、保留策略和迁移工具；`.gitignore` 只忽略生成物，不掩盖应跟踪的规范文档。
5. 增加 ownership/decision 记录：core、construct、benchmark、Agent、Dashboard、deploy 的维护边界和跨组件变更审批点。
6. 检查许可证、第三方前端资源、npm/Python 发布元数据和示例，删除过期组件名与路径。

**验收：** 新开发者只按受跟踪文档即可复现 required checks；全仓搜索无已删除的 `bcg-construct`、
`scripts/start.sh`、个人绝对路径或失效命令；benchmark fixture 与当前 artifact contract 一致。

**回滚：** 文档随对应行为提交或紧随其后；不得先删旧说明而没有新规范入口。

#### 步骤 15 执行记录（2026-08-06，已完成）

- **CONTRIBUTING（文档 2、5）**：required checks 与 `make check` 链逐项
  对齐（lint/compile/test、agent、dashboard、shell、scripts golden、
  contracts、release+install smoke），明确"CI 就是 make check 原样"；
  新增 E2E/外部服务测试说明与安全升级流程（11.6 策略）；维护边界矩阵
  （core/construct/benchmark/Agent/Dashboard/deploy/contracts）+ 跨组件
  变更审批点；架构决策指向 docs/adr/。
- **benchmark（文档 3）**：固定最小 fixture
  `tests/fixtures/benchmark/browsecomp.jsonl` + 端到端 loader smoke 测试；
  `bcg/apps/benchmark/README.md` 说明只读消费（artifact 形状由
  stream.schema.json / memory document 契约覆盖）与数据政策
  （完整数据集不入库、缺失数据 loud fail）。
- **过期引用清理（文档 6、验收）**：bcg_viewer（deprecated 只读工具）的
  `bcg-construct`/`scripts/start.sh`/`outputs_7_2`/`/home/yofuria/...`
  个人路径全部清除（stream 目录回退改为仓库 outputs、Run 按钮显式
  `--start-sh`）；dashboard/package.json 补 license/repository 元数据；
  deploy/README.md 说明模板与 data_dir 注入。git grep 确认 tracked 文件
  无 bcg-construct、scripts/start.sh、/home/yofuria、172.25.10.2
  （REFACTOR_PLAN 自身历史语境除外）。
- **README/维护文档（文档 1）**：README 架构图与 Quick Start 已按真实
  组件（api_based/light、install 两路径）；Agent（agent-cli/README）、
  Dashboard（dashboard/README + 功能矩阵）、部署（scripts/README +
  deploy/README）各有最小维护文档。
- 验收核对：新开发者按受跟踪文档可复现 required checks（CONTRIBUTING
  命令即 make check）✅；全仓无已删除组件名/个人路径/失效命令 ✅；
  benchmark fixture 与当前 artifact contract 一致 ✅。
- 测试/检查：make check 全绿（Python 181（+1 fixture smoke）、Agent
  121、Dashboard 15、contracts、scripts、release、hygiene）。

### 步骤 16：在兼容窗口后删除旧代码

**目的：** 最后清除已经没有消费者的兼容层，而不是把“删除”误当成重构完成标准。

候选项包括：

- `bcg.*` 到 `bcg.core` / `bcg.apps` 的 Python forwarding modules；
- 旧 JSON 配置 migration adapters；
- 达成功能矩阵后的 `dashboard/bcg_viewer/`；
- Agent 中仅为旧 session/settings schema 保留的 reader；
- 已经过一个发布周期的环境变量 alias 和过期 artifact reader。

每个候选必须同时满足：

1. 至少一个已发布兼容窗口已经结束，并有 release note/迁移指南。
2. `rg`、构建产物、文档、示例和已知下游扫描都没有未迁移消费者。
3. 新入口拥有等价 contract/unit/smoke 保护，必要数据已完成迁移演练。
4. 删除作为独立 breaking commit/PR，不夹带新功能或格式变化。
5. 回滚方式明确；持久化数据 reader 通常比 writer 多保留一个版本。

**最终验收：** 全仓 required checks、跨语言 contract、clean install、package smoke 全绿；无两套可写业务实现；
旧路径返回清晰迁移错误或按发布策略消失；文档和 release manifest 同步。

### 12. 第二部分提交序列和完成定义

建议 PR/提交序列：

1. `ci: establish repository-wide build and test gates`
2. `contracts: version graph HTTP and artifact schemas`
3. `refactor(agent): isolate BCG client and session boundaries`
4. `refactor(agent): split interactive orchestration by responsibility`
5. `security(agent): update audited production dependencies`
6. `refactor(dashboard): add typed data adapters and component boundaries`
7. `refactor(dashboard): reach replay/live parity and deprecate legacy viewer`
8. `refactor(deploy): remove machine paths and unify runtime configuration`
9. `build: align installer, Make targets and release manifest`
10. `docs: align architecture, contribution and artifact governance`
11. `breaking: remove expired compatibility layers`

“整个仓库重构完成”必须同时满足以下条件，而不只是目录看起来整齐：

- 所有交付组件进入 CI，且根命令与 CI 一致；
- 跨组件边界有版本化 contract 和 producer/consumer tests；
- 无发布代码依赖个人机器路径或隐含构建顺序；
- 同一状态只有一个 writer/业务实现，兼容路径为薄 adapter；
- 安装、发布、配置、artifact 和旧版本迁移均有可复现证据；
- 步骤 16 的删除条件逐项满足，或明确记录仍在兼容窗口内而暂不删除。
