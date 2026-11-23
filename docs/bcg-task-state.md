# BCG 任务状态层

这是一版可选的、基于来源的任务状态维护。它复用 Unified 原有的 User/Assistant 抽点调用做分类，再由代码维护任务状态骨架，不另起 Planner，不增加一次 Agent 调用。默认关闭，便于与原版 BCG 对照。

首个非空 User 输入是总任务，本轮只抽取普通 belief、原文证据、stance 和实体，不请求 Task State 标签。即使题面分成多问或列出多个交付物，也不会因此生成子任务。原始问题照常保留在 Agent 上下文和 TSG 展示的根位置，不额外创建一个 T 节点。从后续 Assistant 回复（或真实人类追问 / `steer`）中才可以记录子任务；运行时框架与各类 Observer 注入的临时控制提示（如漏工具重试、重复输出提醒、收尾提示等 `role="user"` 催促）统一带 `bcg_origin: "runtime_nudge"` 来源标记：仅在刚注入的当轮请求可见，模型响应后自动随历史淘汰且**永不推入图抽取服务**，不会误生成虚假的 `[Tn]` 子任务，也不依赖文案正则匹配。已有子任务的进度更新方式不变。

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
             与真实来源一起注入 Agent 上下文（具体位置取决于模式）
```

图中的箭头是流程示意，不会据此创建事实关系。原图的 `depends_on`、`supplements`、`contradicts` 保持原来的方向和推导含义。任务层的归属、状态更新与事实推理是不同的关系。

| 表示 | 生成方式 | 含义 |
|---|---|---|
| `task_role: evidence` | Tool Result 由代码标记；User/Assistant 由原有抽点模型分类 | 观察或有来源的信息，不等于已核实事实 |
| `intermediate_conclusion` | 原有 Assistant 抽点调用 | 对证据的判断、推导或假设，保留原来的 stance 和 confidence |
| `task` | 首条用户输入之后的语义抽点 | 明确提出、可以跟踪完成情况的工作项。嵌套子任务通过 `task_ids` 引用父任务；原始 user query 本身是总任务，不通过这个字段创建 |
| `plan` | 原有 User/Assistant 抽点调用 | 明确表达的方法或具体下一步，通过 `task_ids` 关联任务，不生成独立任务记录 |
| `progress` | 原有 User/Assistant 抽点调用 | 明确报告的进度、阻塞、完成或取消 |
| `constraint` / `other` | 原有抽点调用 | 约束或不属于上述分类的信息 |
| `tool_action` / `decision` | 原有工具调用规则 / decision 类型 | 已发出的调用 / 最终选择，不自动代表子任务完成 |

这些是原有 belief 的分类字段，不替换 `node_type`。只有 `task` 生成 `T<belief ID>` 任务记录。版本 4 的 `snapshot.task_state.root` 仅汇总明确针对整体任务的记录，指向原始 User 输入，不是另一条 T 或 B；无明确归属但有原句依据的缺口单独记录在 `unassigned_gap_ids`。旧版快照（1、2、3）仍可读，不追补它们当时没有记录的缺口和归属。

## 每次更新

1. 首个非空 User turn 不请求任务分类，也不采纳模型自行返回的任务字段。它产生的 belief 使用中性标签 `other`。之后的 User/Assistant 语义抽点照常增加 `task_role`，必要时增加 `task_ids` 和 `task_status`。工具调用仍按规则生成；多个 Tool Result 仍使用现有批量抽点策略。
2. 模型只能引用这次 Prompt 实际展示的历史 `task` 节点 ID，不能把计划当作任务引用。对 `task`，`task_ids` 提出候选父任务；对子任务的校验见下文。对 `plan`，它表示该计划服务的任务；对 `progress`，表示被更新的任务。代码拒绝不在窗口中、不是任务或没有可靠原文定位的引用/分类。不能引用本次输出尚未分配的 ID；同一次抽取发现的任务暂时独立保留，尚无可靠引用的计划也保持未关联，不猜测归属。
3. 在原始建图模型路径上，Assistant 抽点在同一历史预算内看到只读的原始问题（`ROOT`，预算的三分之一且不超过 4,000 字符）、最近 12 个任务、最多 8 个尚未明确回应的缺口，再加近期节点。超长问题仅展示有界的原文首尾；Agent 自身的原始 User 输入不截断。首轮 User 不启用这些字段，也没有新增一次模型调用。
4. 只按 `task` 建立任务记录。它引用原始文本、父任务、明确关联的 belief、`plan_ids`、历史状态更新和最近一次状态来源。任务、计划、进度记录和明确关联任务的 belief 不参与增量合并，避免来源 ID 或状态时间线被去重吞掉；其他节点仍走原有合并逻辑。
5. 从任务和显式关联的计划、结论、进度等 belief 出发，在真实 `depends_on`、`supplements` 边上最多寻找 3 跳相关信息，不跨过另一个任务。检索时可以双向走边，但展示始终保留原始方向。相关不代表逻辑蕴含，更不代表任务已完成。
6. 相关区域存在 `contradicts` 时标记 `needs_review`。原来的 confidence 公式、数值和传播实现不变；任务层不生成另一套置信度，也不把 workflow 链接送入 confidence propagation。
7. 每次 acknowledged snapshot 和最终图都附带 `task_state`。SDK 的 memory 导出保留分类和任务层。状态可随图重建，不需要另一个数据库或模型维护隐藏状态。

Assistant 直接建立一级子任务时也要提交 `root_refinement`：引用本轮原话和可见的 ROOT 原文，说明选中的较小范围或新交付物。不仅仅因为重复题面、重列各国各项就建任务。复述保留为普通 belief。计划/进度通过 `task_ids` 指向可见子任务，或用 `task_scope: root` 明确指向整体任务。两者都不明确时保持未归属，不能影响状态。

同一 belief 可同时保留事实和原文明确提出的 `gap_quote`；`next_action_quote` 是具体下一步的原文。对于已有原句证明、但未给出明确任务归属的缺口，记录在 `unassigned_gap_ids` 并在整体任务下以 `open gap (unassigned)` 展示带对象的原句，不补造子任务归属，也不将其误判为 `blocked`。一个被 Agent 明确回应的缺口通过 `closes_gaps` 引用以前已展示的 B ID 和本轮回应原文。代码校验原文、先后顺序和所属任务；即使有搜索结果或已经交报告，也不会据此自动关闭缺口或把任务标成完成。Decision Model 对照组不使用这些新抽点字段。

### 如何区分子任务与复述

以 RUN-0028 第 2 题为例：B2 要求识别印度、巴基斯坦、孟加拉国和尼泊尔的社会保障项目并说明其失败。B234 把同一要求改写为“报告必须识别这四国的项目并描述其失败”。交付物和覆盖范围都没变，不应生成 `T234 part_of T2`。如果这次复述被抽成 Belief，可以通过 `B234 addresses T2` 留下来源，但不会更新 T2 的状态。

当本轮看得到历史任务时，抽点模型在同一次响应中为每个候选父任务填写一条 `task_refinements`。它有两种依据：`narrower_scope` 指本次工作只覆盖父任务的一部分；`new_deliverable` 指本轮首次指定了一个服务于父任务、可单独验收的产出。下面是**构造的正例**：如果新一句明确说 “Investigate program failures in Nepal only.”，下面这条引用可以支持新的子任务；完成尼泊尔部分并没有完成巴基斯坦部分。

```json
{"task_role":"task","task_ids":[2],"task_refinements":[{"parent_id":2,"kind":"narrower_scope","parent_quote":"Pakistan","child_quote":"Nepal only"}]}
```

模型负责判断语义，代码逐父级核对引用：父引文必须出现在本次实际展示的父任务正文，子引文必须来自**当前这条 Belief 自己的原文证据**。通过后可在快照中查看 `part_of.basis` 的两段原文。相同任务文本、只把相同对象再次列出、把父任务已有的产出原文当作全新产出，都不能作为对应的拆分依据。都未通过时，保留已抽出的 Belief 和有效的任务引用，不把它另挂成一个顶层任务。在原始模型的 Assistant 路径中，没有历史父 T 的一级任务也必须对照 ROOT 给出依据；ROOT 不是新的事实节点。这个检查不另调模型，也不按相似度聚类或改写历史快照。

## 进度的含义

状态包括 `open`、`in_progress`、`blocked`、`reported_complete`、`cancelled`。没有明确报告时显示 `open`，并保存 `status_provenance: not_reported`；这不是对实际完成率的判断。

`reported_complete` 只表示 User/Assistant 声称完成，必须有对应原文和任务引用。搜索有结果、执行过工具、高 confidence 或写出了最终报告都不会自动触发完成。后续报告可以重新打开或阻塞同一任务，旧的更新 ID 仍保留。找不到可靠目标的进度记录进入 `unassigned_progress_ids`，不会据此关闭任何任务。

计划不再拥有独立任务状态。没有明确任务引用的计划进入 `unassigned_plan_ids`，仍保存在原图中，不根据内容相似度强行挂到某个任务。

任务层有 `described_by`、`part_of`、`updated_by`、`addresses`、`related_via_graph` 这几种链接，只存在于 `task_state.relations`。它们用于工作流追溯，不进入原图的事实关系集合。

## Agent 如何使用

BCG Compaction 在原始 User 输入之后以 user role 放置 `<task_state>`；无内容时不发送空消息。最近 5 轮和全部轮次策略使用相同的全局 Task State（默认不设字符上限），这个选择只改变逐轮 `<belief_graph>`。两者均可正交开启 `compaction_task_state_filter`（环境变量 `BCG_COMPACTION_TASK_STATE_FILTER`、CLI `--compaction-task-state-filter`，默认关闭）：以当前活跃 Task State 节点（未完成的 T 任务、整体任务最新进展/未解决缺口/活跃下一步及未归属缺口，排除已报告完成/取消或已回应关闭的条目）为种子，在全图真实关系上做无额外模型调用的 k 跳扩展（`compaction_task_state_hops`，默认 2），按轮次交集筛选每轮保留节点并自动补带对应工具调用；最近 1 个已结束轮次无条件完整保留，筛选后为空的历史轮次跳过不发空块，关系仍要求两端均入选且所有节点保持原本所属轮次与 `v1` 格式不变。前端根任务和子任务的详情也可查看缺口、下一步和进展来源。

普通 BCG（非 Compaction）维持既有的原始交互保留策略，将全局 Task State 放在 `<belief_graph>` 中；Compaction 使用上文所述的独立 user 消息。两者共享同一任务状态投影。

单 Agent BCG 的任务状态始终由全部已入图的历史节点和关系维护，与近期窗口无关。`task_state_chars` 默认不设（无字符上限）；显式设置为 ≥256 的整数时，按“整体状态 → 约束 → 子任务 → 最新计划 → 缺口 → 受阻尝试 → 已有发现（新者优先）”的优先级整条保留，放不下的条目整条跳过，不截断句子。

Task state 之外的 Graph 内容不再 retrieval，而是选取最近两轮完整原始交互之前的三个完整交互轮的全部节点，以及两端均在这些节点内的原始关系。一轮包括一次 Assistant 输出（Thinking、全部 Tool Call）和其全部并行 Tool Result；没有工具的 Assistant 输出也算一轮，抽点为空的轮次不会用更老的轮次补位。例如第 11 次调用前，9–10 轮保留原文，6–8 轮作为近期 Graph，Task state 基于全部已入图历史维护，而非仅 1–5 轮。初始用户输入持续保留，不额外重复到近期 Graph。合并节点按归并前来源追踪，使用当前 canonical ID 和原有关系。

`<belief_graph>`、Task state / Completed actions / Findings / Relations 的层级、节点和边写法均不变，只替换筛选内容。近期三轮不再受旧 6,600 字符阈值裁剪。原图和最终导出保留全部历史。`requests.jsonl` 在 `selector_result.agent_step_numbers`、`graph_turn_ids` 中记录完整交互轮与建图层的对应关系，`task_state.scope: full_graph` 区分全局状态来源。Agent Team 的旧 retrieval 行为不变。

兼容的 BCG w/o Retrieval 配置使用完整原图，不额外截断任务层的总字符数，仍只选取与各任务相关的工作和证据。Default 不启用 Task State，不修改其 Prompt 或 context。

`source` 指向原始任务 belief，`status` 是报告状态而非完成验证。任务状态枚举不变；新增的主任务归属、缺口和来源只在有记录时显示。下表为状态字段的读法。

| 内部状态 | 注入中的状态词 |
|---|---|
| `open`，没有进度来源 | `no progress reported`，不等于尚未开始 |
| `open`，有明确状态来源 | `open (reported)` |
| `in_progress` | `in progress (reported)` |
| `blocked` | `blocked (reported)`，具体阻塞保留在 progress 原文中 |
| `reported_complete` | `complete (reported, not verified)` |
| `cancelled` | `cancelled (reported)` |

### 注入格式（v5 渲染）

`<task_state>` 固定为 5 个小节，空小节不显示；有子任务时以缩进树嵌在第一节，其余条目用 `(for [Tn])` 标注归属，同一个 B 的正文只出现一次：

1. **Task status and constraints**：整体任务的最新报告状态与进展原文、按核心实体/主题去重后的 `constraint` 节点、子任务树（`[Tn]` 为 Agent 在 `[Bn]` 中定义的子任务，按 `part_of` 缩进，已报告完成/取消的排在同级末尾，并给出最新状态来源）。
2. **Established findings**：保留 `intermediate_conclusion` 及信息密度达标的历史 `evidence` 归纳（要求包含多实体对比、量化指标或结构化分句，剔除沙盒环境/代码执行报错等过程噪音）；已在下方近期 `<belief_graph>` 窗口（最近 3 个已结束 Assistant 轮次）中展示的 `evidence` 不重复打印，也不含原始工具结果；同时展示相关的 `contradicts` 关系（未展示端点内联原文）。
3. **Open gaps**：展示带主语的完整缺口原句而非 `gap_quote` 片段。全局/未归属缺口与建图抽点侧 `task_history` 的可关闭窗口（最多 8 个活跃缺口）对齐，避免早已滚出抽点窗口的旧缺口永久滞留；若后续实质性发现已覆盖该缺口的核心实体与指标词，也在展示层自动退役。
4. **Blocked or failed attempts**：从图中按 `tool_call_id` 配对出被拦截或失败的工具调用（工具名 + 原查询/URL），对高度相似的重复失败查询自动聚类折叠为单行并标注 `(+K similar)`；已有结构化工具失败记录时省略冗余的无归属自述抱怨。
5. **Latest plan**：只保留最近一次记录了计划的 Assistant 轮次中的计划；更早轮次的计划自动退役；同轮有具体计划时过滤空泛套话。

所有条目按语义近似去重（新的表述优先），展示层去掉抽取器的第三人称前缀（如 `The assistant reports that`），不改动存储的 belief。

示意（非实验结果）：

```text
## Task state
Your own research state from earlier turns, rebuilt from the full graph. Statuses, findings and plans are self-reported, not independently verified.
### Task status and constraints
- Subtasks:
  - [T0] no progress reported — Compare two licensing schemes. [B0]
    - [T1] blocked (reported) — Verify the operating license. [B1]
      latest update [B6]: License verification is blocked until the notice is available.
### Established findings (your earlier syntheses)
- [B3] (for [T1]) Pilot status does not establish licensed operation.
### Latest plan (intended next steps, not verified as done)
- [B4] (for [T1]) Fetch the licensing authority's original notice.
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
    # task_state_chars: 4000   # 可选；默认不设上限
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
