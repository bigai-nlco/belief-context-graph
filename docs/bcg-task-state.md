# BCG 任务状态层

这是一版可选的、基于来源的任务状态维护。它复用 Unified 原有的 User/Assistant 抽点调用做分类，再由代码维护任务状态骨架，不另起 Planner，不增加一次 Agent 调用。默认关闭，便于与原版 BCG 对照。

## 两层表示

```text
工具返回 / 用户输入 / Agent 回复
                 │
         原有 Belief 抽取与建边
                 │
    Evidence ──真实关系── Intermediate conclusion
                              │ 显式 task_ids
                              ▼
                 Task-state skeleton
          goal → subgoal → plan / next_action
                   ↑ progress 记录
                              │
             与真实来源和关系链一起注入 system
```

图中的箭头是流程示意，不会据此创建事实关系。原图的 `depends_on`、`supplements`、`contradicts` 保持原来的方向和推导含义。任务层的归属、状态更新与事实推理是不同的关系。

| 表示 | 生成方式 | 含义 |
|---|---|---|
| `task_role: evidence` | Tool Result 由代码标记；User/Assistant 由原有抽点模型分类 | 观察或有来源的信息，不等于已核实事实 |
| `intermediate_conclusion` | 原有 Assistant 抽点调用 | 对证据的判断、推导或假设，保留原来的 stance 和 confidence |
| `goal` / `subgoal` | 原有 User/Assistant 抽点调用 | 明确提出的目标或工作项，不凭连通分量臆造子任务 |
| `plan` / `next_action` | 原有 Assistant 抽点调用 | 明确表达的方法或下一步，不是代码给 Agent 新下达的指令 |
| `progress` | 原有 User/Assistant 抽点调用 | 明确报告的进度、阻塞、完成或取消 |
| `constraint` / `other` | 原有抽点调用 | 约束或不属于上述分类的信息 |
| `tool_action` / `decision` | 原有工具调用规则 / decision 类型 | 已发出的调用 / 最终选择，不自动代表子任务完成 |

这些是原有 belief 的分类字段，不替换原来的 `node_type`。程序同时生成独立的 `snapshot.task_state`，其中任务记录 ID 为 `T<anchor belief ID>`，原始图 ID 仍为整数，展示为 `B<id>`。

## 每次更新

1. 原有 User/Assistant 抽点请求增加 `task_role`，必要时增加 `task_ids` 和 `task_status`。工具调用仍按规则生成；多个 Tool Result 仍使用现有批量抽点策略，不增加独立的分类请求。
2. 模型只能引用这次 Prompt 实际展示的历史任务 anchor ID。代码拒绝不在窗口中、不是任务 anchor 或没有可靠原文定位的引用/分类。不能引用本次输出尚未分配的 ID；同一次抽取发现的子目标暂时独立保留，后续明确关联时再接入。
3. 抽点历史优先保留最近 12 个任务 anchor，再加入原来的近期节点，整个历史部分仍受 `context_chars` 限制（当前配置为 12,000）。因此较早的目标在最近原文已淘汰后仍可被后续结论引用。
4. 按 anchor 建立任务记录。它引用原始文本、父任务、明确关联的 belief、历史状态更新和最近一次状态来源。anchor、进度记录和明确关联任务的 belief 不参与增量合并，避免任务 ID 或状态时间线被去重吞掉；其他节点仍走原有合并逻辑。
5. 从任务 anchor 和显式关联的结论/进度出发，在真实 `depends_on`、`supplements` 边上最多寻找 3 跳相关信息，不跨过另一个任务 anchor。检索时可以双向走边，但展示始终保留原始方向。相关不代表逻辑蕴含，更不代表任务已完成。
6. 相关区域存在 `contradicts` 时标记 `needs_review`。原来的 confidence 公式、数值和传播实现不变；任务层不生成另一套置信度，也不把 workflow 链接送入 confidence propagation。
7. 每次 acknowledged snapshot 和最终图都附带 `task_state`。SDK 的 memory 导出保留分类和任务层。状态可随图重建，不需要另一个数据库或模型维护隐藏状态。

## 进度的含义

状态包括 `open`、`in_progress`、`blocked`、`reported_complete`、`cancelled`。没有明确报告时显示 `open`，并保存 `status_provenance: not_reported`；这不是对实际完成率的判断。

`reported_complete` 只表示 User/Assistant 声称完成，必须有对应原文和任务引用。搜索有结果、执行过工具、高 confidence 或写出了最终报告都不会自动触发完成。后续报告可以重新打开或阻塞同一任务，旧的更新 ID 仍保留。找不到可靠目标的进度记录进入 `unassigned_progress_ids`，不会据此关闭任何任务。

任务层有 `described_by`、`part_of`、`updated_by`、`addresses`、`related_via_graph` 这几种链接，只存在于 `task_state.relations`。它们用于工作流追溯，不进入原图的事实关系集合。

## Agent 如何使用

用户输入、最近两轮、淘汰后建图时机、工具与 Agent 调用方式不变。原本应该注入 Graph 时，在 `<belief_graph>` 内加入一个 `## Task state` 小节。它优先保留未完成任务和近期更新，连同来源文本，尽量附上一条真实的 Evidence–Conclusion 关系链。关系链不完整时不会补造边。

普通 BCG 中任务小节默认最多占 1,800 字符，从原有 6,600 字符 Graph 总预算中扣除，剩余部分继续使用原有 retrieval。按完整任务包取舍，放不下时跳过该包，不截断节点正文。所有实际注入的源节点/边 ID 都进入 `requests.jsonl` 的选择记录。

BCG w/o Retrieval 中不设上述字符上限，仍放入完整原图，附上完整任务层。Default 不启用这个扩展，也不更改其 Prompt 或 context 行为。

示意（非实验结果）：

```text
## Task state
- [T1] subgoal | blocked
  [B1] subgoal: Verify the operating license.
  [B6] progress: License verification is blocked until the notice is available.
  [B3] intermediate_conclusion: Pilot status does not establish licensed operation.
  [B2] evidence: The source describes a pilot, not an operating license.
  [B3] addresses [T1] (explicit task reference, not a factual edge)
  [B3] depends_on [B2]: distinguish pilot from operation
- [T4] next_action | open
  [B4] next_action: Fetch the licensing authority's original notice.
```

目标是让 Agent 看到“还缺什么、已有判断依赖什么、下一步先前打算核实什么”。它仍自行决定下一次调用，不强制执行 task node，也不通过额外 Agent 回合制定计划。

## 启用

Studio 的「启动测试」中选择 BCG 或 BCG w/o Retrieval，勾选「维护任务状态（实验功能）」。这会为该 Run 同时设置 Graph 和 Agent 两边的开关，并记录在该 Run 的配置中。已有运行与结果不变。

手动运行时，两边都要配置。Graph YAML：

```yaml
backend: unified
pipeline:
  task_state:
    enabled: true
```

Agent profile：

```yaml
agent:
  context_mode: bcg
  bcg:
    task_state_enabled: true
    task_state_chars: 1800
```

其余模型、URL、最近轮数和 retrieval 配置沿用现有 profile。暂只支持 Unified，Hybrid 配置开启该功能会在启动校验阶段报错。关掉两边开关即可恢复原版，不需要删除模块。

若 Agent 开启而连接的旧 Graph 服务未开启，普通 Graph 注入仍继续，日志中的 `task_state.available` 为 `false`；这不代表任务状态已经生效。

## 验证与边界

离线测试覆盖分类与 ID 校验、状态来源、恢复/冲突、真实边追溯、预算、snapshot/SDK 导出、原有模式兼容以及 Studio 启动配置。`tools/task_state_replay.py` 还能对已有图做显式标注回放：

```bash
uv run python -m tools.task_state_replay \
  --state path/to/agent/bcg/state.json \
  --labels path/to/reviewed-labels.json \
  --out results/task-state-replay-demo
```

`reviewed-labels.json` 是以真实 belief ID 为键的对象，例如 `{"13": {"task_role": "plan", "task_ids": [2]}}`；被引用的 B2 也需要在同一文件中标为任务 anchor。该回放不调用模型，不修改原结果，只验证状态投影与来源链。输出明确标为 `offline_explicit_label_replay`，不能作为模型分类质量或 Agent 性能提升的证据。

这版没有增加独立的 Planner/分类调用，但抽点 Prompt 和输出会增加字段；状态保留和后续 Agent 决策也可能改变总 Token、图大小和调用次数。需要后续同题对照实验衡量收益，不能由离线测试推断准确率、耗时或费用已经改善。对于 Agent 从未表达的计划、模糊完成宣称或原图缺失的依赖边，本版会保留未知，而不是自动补齐。
