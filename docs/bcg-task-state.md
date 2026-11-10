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
             task → task（父子任务）
              ↑ plan / progress 等关联记录
                              │
             与真实来源和关系链一起注入 system
```

图中的箭头是流程示意，不会据此创建事实关系。原图的 `depends_on`、`supplements`、`contradicts` 保持原来的方向和推导含义。任务层的归属、状态更新与事实推理是不同的关系。

| 表示 | 生成方式 | 含义 |
|---|---|---|
| `task_role: evidence` | Tool Result 由代码标记；User/Assistant 由原有抽点模型分类 | 观察或有来源的信息，不等于已核实事实 |
| `intermediate_conclusion` | 原有 Assistant 抽点调用 | 对证据的判断、推导或假设，保留原来的 stance 和 confidence |
| `task` | 原有 User/Assistant 抽点调用 | 明确提出、可以跟踪完成情况的目标或工作项。总目标和子任务使用同一类别，通过 `task_ids` 表示父任务 |
| `plan` | 原有 User/Assistant 抽点调用 | 明确表达的方法或具体下一步，通过 `task_ids` 关联任务，不生成独立任务记录 |
| `progress` | 原有 User/Assistant 抽点调用 | 明确报告的进度、阻塞、完成或取消 |
| `constraint` / `other` | 原有抽点调用 | 约束或不属于上述分类的信息 |
| `tool_action` / `decision` | 原有工具调用规则 / decision 类型 | 已发出的调用 / 最终选择，不自动代表子任务完成 |

这些是原有 belief 的分类字段，不替换原来的 `node_type`。模型使用 7 个语义类别，`tool_action` 和 `decision` 仍由代码识别。程序同时生成独立的 `snapshot.task_state`（版本 2），只有 `task` 生成任务记录。其 ID 为 `T<task belief ID>`，原始图 ID 仍为整数，展示为 `B<id>`。例如 B42 表示待完成的工作，T42 跟踪它的状态；B45 可以作为它的计划保留，但不再建立 T45。

## 每次更新

1. 原有 User/Assistant 抽点请求增加 `task_role`，必要时增加 `task_ids` 和 `task_status`。工具调用仍按规则生成；多个 Tool Result 仍使用现有批量抽点策略，不增加独立的分类请求。
2. 模型只能引用这次 Prompt 实际展示的历史 `task` 节点 ID，不能把计划当作任务引用。对 `task`，`task_ids` 表示父任务；对 `plan`，表示该计划服务的任务；对 `progress`，表示被更新的任务。代码拒绝不在窗口中、不是任务或没有可靠原文定位的引用/分类。不能引用本次输出尚未分配的 ID；同一次抽取发现的任务暂时独立保留，尚无可靠引用的计划也保持未关联，不猜测归属。
3. 抽点历史优先保留最近 12 个 `task` 节点，再加入原来的近期节点，计划不占这 12 个任务名额。整个历史部分仍受 `context_chars` 限制（当前配置为 12,000）。
4. 只按 `task` 建立任务记录。它引用原始文本、父任务、明确关联的 belief、`plan_ids`、历史状态更新和最近一次状态来源。任务、计划、进度记录和明确关联任务的 belief 不参与增量合并，避免来源 ID 或状态时间线被去重吞掉；其他节点仍走原有合并逻辑。
5. 从任务和显式关联的计划、结论、进度等 belief 出发，在真实 `depends_on`、`supplements` 边上最多寻找 3 跳相关信息，不跨过另一个任务。检索时可以双向走边，但展示始终保留原始方向。相关不代表逻辑蕴含，更不代表任务已完成。
6. 相关区域存在 `contradicts` 时标记 `needs_review`。原来的 confidence 公式、数值和传播实现不变；任务层不生成另一套置信度，也不把 workflow 链接送入 confidence propagation。
7. 每次 acknowledged snapshot 和最终图都附带 `task_state`。SDK 的 memory 导出保留分类和任务层。状态可随图重建，不需要另一个数据库或模型维护隐藏状态。

## 进度的含义

状态包括 `open`、`in_progress`、`blocked`、`reported_complete`、`cancelled`。没有明确报告时显示 `open`，并保存 `status_provenance: not_reported`；这不是对实际完成率的判断。

`reported_complete` 只表示 User/Assistant 声称完成，必须有对应原文和任务引用。搜索有结果、执行过工具、高 confidence 或写出了最终报告都不会自动触发完成。后续报告可以重新打开或阻塞同一任务，旧的更新 ID 仍保留。找不到可靠目标的进度记录进入 `unassigned_progress_ids`，不会据此关闭任何任务。

计划不再拥有独立任务状态。没有明确任务引用的计划进入 `unassigned_plan_ids`，仍保存在原图中，不根据内容相似度强行挂到某个任务。

任务层有 `described_by`、`part_of`、`updated_by`、`addresses`、`related_via_graph` 这几种链接，只存在于 `task_state.relations`。它们用于工作流追溯，不进入原图的事实关系集合。

## Agent 如何使用

用户输入、最近两轮、淘汰后建图时机、工具与 Agent 调用方式不变。原本应该注入 Graph 时，在 `<belief_graph>` 内加入一个 `## Task state` 小节。它优先保留未完成任务和近期更新，必带任务原文及最新状态来源。在预算允许时，加入该任务最新一条明确关联的计划，再尽量附上一条真实的 Evidence–Conclusion 关系链。计划显示在任务下面，不单独占据一个 T 记录。关系链不完整时不会补造边。

单 Agent BCG 的任务状态始终由全部已入图的历史节点和关系维护，与近期窗口无关。注入时任务小节保留原有 1,800 字符预算和完整任务包选择规则，放不下时跳过该包，不截断节点正文。维护全局状态不等于每次显示所有任务记录。

Task state 之外的 Graph 内容不再 retrieval，而是选取最近两轮完整原始交互之前的三个完整交互轮的全部节点，以及两端均在这些节点内的原始关系。一轮包括一次 Assistant 输出（Thinking、全部 Tool Call）和其全部并行 Tool Result；没有工具的 Assistant 输出也算一轮，抽点为空的轮次不会用更老的轮次补位。例如第 11 次调用前，9–10 轮保留原文，6–8 轮作为近期 Graph，Task state 基于全部已入图历史维护，而非仅 1–5 轮。初始用户输入持续保留，不额外重复到近期 Graph。合并节点按归并前来源追踪，使用当前 canonical ID 和原有关系。

`<belief_graph>`、Task state / Completed actions / Findings / Relations 的层级、节点和边写法均不变，只替换筛选内容。近期三轮不再受旧 6,600 字符阈值裁剪。原图和最终导出保留全部历史。`requests.jsonl` 在 `selector_result.agent_step_numbers`、`graph_turn_ids` 中记录完整交互轮与建图层的对应关系，`task_state.scope: full_graph` 区分全局状态来源。Agent Team 的旧 retrieval 行为不变。

BCG w/o Retrieval 中不设上述字符上限，仍放入完整原图，附上所有有效任务记录；每项仍采用任务、状态、最新关联计划和一条关系路径的展示方式，不展开所有计划和状态历史。Default 不启用这个扩展，也不更改其 Prompt 或 context 行为。

注入时保留紧凑结构，用 `source` 明示原始任务 belief，用自然语言解释 `status`，不再要求 Agent 解码 `task | in_progress`，也不重复整句 “is a task which is …”。`[Tn]` 跟踪原始任务 belief `[Bn]`，缩进的 `[B…]` 行保留原始节点和边的内容。JSON 中的状态枚举及状态计算不变，仅改变给 Agent 的展示。

| 内部状态 | 注入中的状态句 |
|---|---|
| `open`，没有进度来源 | `status: unknown (no progress reported).`，不等于尚未开始 |
| `open`，有明确状态来源 | `status: reported open.` |
| `in_progress` | `status: reported in progress.` |
| `blocked` | `status: reported blocked.`，具体阻塞保留在 progress 原文中 |
| `reported_complete` | `status: reported complete (not verified).` |
| `cancelled` | `status: reported cancelled.` |

示意（非实验结果）：

```text
## Task state
[Tn] tracks the task recorded in [Bn]. Status reflects reported progress, not verified completion. Unknown means no progress was recorded.
- [T1] source: [B1] | status: reported blocked.
  [B1] task: Verify the operating license.
  [B6] progress: License verification is blocked until the notice is available.
  [B4] plan: Fetch the licensing authority's original notice.
  [B3] intermediate_conclusion: Pilot status does not establish licensed operation.
  [B2] evidence: The source describes a pilot, not an operating license.
  [B3] addresses [T1] (explicit task reference, not a factual edge)
  [B3] depends_on [B2]: distinguish pilot from operation
```

目标是让 Agent 看到“还缺什么、已有判断依赖什么、下一步先前打算核实什么”。它仍自行决定下一次调用，不强制执行 task node，也不通过额外 Agent 回合制定计划。

### 格式检查

2026-11-05 使用 RUN-0018 的 drb2:96 做固定输入检查，保留同一批历史节点、边和近期消息，用当前本地 Qwen3.5-9B 的 tokenizer 计数。20 次历史注入中，原版 task-state 小节平均 415.8 tokens，保留枚举并补完整图例为 466.8 tokens，完整自然语言句子版本为 430.3 tokens，最终的 `source` / `status` 短字段版本为 429.3 tokens。这里是小节的输入 Token，不是整题总 Token；最终版本相对原版增加 13.5 tokens，相对完整图例减少约 8%。

最终 renderer 在 1,800 字符预算内保留了这 20 次注入的全部原有任务、节点文本和关系，最长 1,772 字符。另对没有进度记录、已有明确进度的两个真实输入做状态理解检查，并对其中一次输入检查下一步工具调用。最终格式的任务 ID、来源 ID、进度来源及计划/完成边界检查通过；两种格式都提出了 4 次搜索，没有验证搜索次数减少。

这是单任务上的探索性格式检查，不是完整任务重跑，没有独立盲评或最终报告评分。早期措辞在同一模型上仍出现过状态误读，不能据此声称模型在所有任务中都会正确理解，也不能把格式 Token 的变化解释为整题准确率或总成本的改善。历史结果、正在运行的服务和模型配置均未改动。

## 旧记录兼容

读取旧标签时，`goal` 和 `subgoal` 映射为 `task`，`next_action` 映射为 `plan`。版本 1 的任务层由原始节点和边重新投影为版本 2，旧计划不再生成 T 记录。旧计划之间的已有 `task_ids` 链可以追溯到真实任务；无法追溯的计划保持未关联。仅引用旧计划的进度不会转为父任务的状态更新，避免把“某次搜索完成”误当作“整个任务完成”。

兼容处理不修改磁盘上的历史图、节点、边或 confidence。文档中历史实验的实际注入原文仍保留旧标签，并标明为旧版记录；新版展示示意或离线投影不冒充新的实际运行结果。

## 启用

Studio 的「启动测试」中选择 BCG，默认开启全局任务状态并自动配置 Graph 服务。BCG w/o Retrieval 仍可选是否启用任务状态。已有运行与结果不改写。

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
    recent_turns: 2
    graph_recent_turns: 3
    task_state_chars: 1800
```

其余模型、URL 配置沿用现有 profile。暂只支持 Unified，Hybrid 配置开启该功能会在启动校验阶段报错。单 Agent BCG 强制使用 `recent_steps` 和 Task state，旧 `BCG_RETRIEVAL` 环境变量不会重新开启 retrieval。`graph_recent_turns` 或 `BCG_GRAPH_RECENT_TURNS` 可将近期图窗口调整为 3、4、5 等正整数。

若单 Agent BCG 连接的旧 Graph 服务未开启任务状态，保留完整原始上下文并记录 `graph_server_task_state_required`，不将缺少全局状态的近期窗口冒充完整方案。需更新 Graph 配置并重启服务。

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
