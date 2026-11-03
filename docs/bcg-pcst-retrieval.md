# BCG 的 PCST retrieval

BCG 现在提供三个可选策略：`focused`、`confidence_pcst`、`query_pcst`。
默认使用 **`query_pcst`**，相关性向量来自配置的远程 embedding 服务。PCST 在 Agent 内对已确认的图快照运行，
无需修改独立 Graph 服务；单 Agent 和 Team 每个成员的本地图都支持。

这是受 G-Retriever 启发的 BCG 适配，不是论文实现的完整复现：本版只使用节点奖励，
不做边文本 embedding、不设边奖励，也不引入 GNN 或训练步骤。

## 启用

普通安装已包含 PCST 求解器和兼容的 NumPy；默认远程路径无需下载模型或安装 PyTorch：

```bash
uv sync
```

```dotenv
BCG_RETRIEVAL=query_pcst
BCG_PCST_EDGE_COST=0.2
BCG_PCST_CONFIDENCE_FLOOR=0.2
BCG_RETRIEVAL_EMBEDDING_PROVIDER=openai
BCG_RETRIEVAL_EMBEDDING_URL=http://10.1.100.76:8016/v1/embeddings
BCG_RETRIEVAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BCG_RETRIEVAL_EMBEDDING_API_KEY=
BCG_RETRIEVAL_EMBEDDING_TIMEOUT=60
BCG_RETRIEVAL_EMBEDDING_CHUNK_BYTES=240
BCG_GRAPH_CHARS=6600
```

然后沿用原来的 `frontier-agent --native --mode react` 或 `--mode agent_team` 启动命令。
这里配置的是 **Agent 端检索模型**，独立于 Graph 服务的构图 embedding。`openai` 表示接口协议，
不是调用 OpenAI 公网。2026-11-03 实测该地址返回 `sentence-transformers/all-MiniLM-L6-v2` 的
384 维向量，健康检查显示服务运行在 CUDA 上。请求直接访问配置的完整 URL，不继承
`OPENAI_API_KEY` 或系统代理；认证仅使用可选的 `BCG_RETRIEVAL_EMBEDDING_API_KEY`。
MiniLM 不是针对中文任务优化的模型，后续可换用更适合任务语言的服务。

需要本地编码时执行 `uv sync --extra bcg-embeddings`，设置
`BCG_RETRIEVAL_EMBEDDING_PROVIDER=local`、`BCG_RETRIEVAL_EMBEDDING_DEVICE=cpu`，
以及模型名称或本地目录 `BCG_RETRIEVAL_EMBEDDING_MODEL`；离线运行必须提前准备权重。

只做 confidence 对照时，设置以下变量即可；此策略不调用 embedding：

```dotenv
BCG_RETRIEVAL=confidence_pcst
```

恢复旧策略使用 `BCG_RETRIEVAL=focused`。所有新参数也支持 profile 的 `agent.bcg` 配置：
`retrieval`、`pcst_edge_cost`、`pcst_confidence_floor`、`embedding_provider`、`embedding_url`、
`embedding_model`、`embedding_api_key`、`embedding_timeout`、`embedding_chunk_bytes`、`embedding_device`、
`embedding_store`、`embedding_revision`、`tonggraph_url`、`tonggraph_token`、`tonggraph_graph`、`tonggraph_timeout`；
显式 profile 值优先于环境变量。

`pcst-fast` 当前发布的 wheel 在 NumPy 2 下可能返回损坏的索引数组，因此基础安装
明确依赖 `numpy>=1.26,<2`，运行时也会检查。`bcg-pcst` extra 保留为兼容别名。
缺依赖、服务请求失败、向量格式错误、模型加载失败、求解失败或无有效选择，
均记录原因并沿用完整历史回退，不静默改用另一种检索算法。

## 向量持久化到 TongGraph

远程编码支持两层缓存：进程内向量缓存，以及 TongGraph 持久化向量。
通用代码默认 `memory`；需要跨进程复用时，可显式配置已有的 TongGraph 服务：

```dotenv
BCG_RETRIEVAL_EMBEDDING_STORE=tonggraph
BCG_RETRIEVAL_EMBEDDING_REVISION=1
TONGGRAPH_BASE_URL=http://127.0.0.1:8719
TONGGRAPH_GRAPH=frontier_retrieval
TONGGRAPH_TOKEN=your-writer-token
TONGGRAPH_TIMEOUT=30
```

部署时物理图需要预先创建并授予 writer 写权限；客户端会自动创建该图中的逻辑图与向量索引。

`TONGGRAPH_TOKEN` 为空时也接受 `TONGGRAPH_AGENT_WRITER_TOKEN`。这里只使用配置的 writer
凭据，不读取管理员凭据，也不在审计中保存 token。此功能目前要求 `embedding_provider=openai`。

每次 query-PCST 检索依次执行：

1. 从内存缓存取节点正文向量。
2. 对内存未命中的正文，按文本 SHA-256 和编码配置从 TongGraph 读取分块向量。
3. 只调用 embedding 服务编码仍缺失的文本；query/focus 使用进程内缓存。
4. 将新得到的节点正文向量写入 TongGraph，再计算当前 query 的 relevance 和 PCST。
5. 任务结束时，final graph flush 在构图请求得到确认后，会为最后几步的新节点编码并持久化；
   此时不运行 PCST。如果构图等待超时，本次不会进入末尾向量保存，必须检查失败审计。

TongGraph 的存储映射：

| 记录 | 内容 |
|---|---|
| `BCGRetrievalText` 节点 | 完整正文、文本哈希、编码配置、维度、有序分块节点 ID 清单 |
| `BCGRetrievalChunk` 节点 | 对应文本节点 ID、分块序号和文本哈希 |
| `bcg_pcst_<fingerprint>` 向量索引 | 每个分块节点绑定一个向量，保留原来的分块最大余弦匹配 |
| `bcg_pcst_embeddings_<fingerprint>` 逻辑图 | 同一编码配置的正文向量缓存，可跨 Agent 进程和任务复用 |

编码配置指纹包含服务 URL、模型、revision、分块算法及字节上限、归一化方式和缓存格式版本。
正文修改时使用新的文本键；仅 confidence 改变时仍可复用正文向量。更换同名模型背后的权重时，
需要增加 `BCG_RETRIEVAL_EMBEDDING_REVISION`，让新运行使用新的逻辑图和索引。

这里持久化的是**检索用的正文和向量缓存**。belief 的身份、confidence、来源和关系仍以
BCG 图快照为准；本文缓存不负责同步完整 BCG 图。相同正文可由多个 belief 共享，取向量时按
正文哈希查找。query/focus 向量不额外持久化；relevance 随当前问题重算，也不存作固定节点属性。

已有 `agent_embedding` 索引保持原样。新索引按编码配置隔离并核验模型、维度、metric 和版本，
避免将现有 1024 维 Jina 向量与当前 384 维 MiniLM 向量混用。稳定 external ID 支持幂等重试；
所有分块写入成功后才发布完整清单，部分写入或缺失/损坏向量会作为缓存未命中处理。

TongGraph 不可用时检索仍可使用内存/新编码的向量，审计 `embedding.persistent_cache.errors`
记录持久化失败，后续检索会重试未保存的正文。`loaded_texts`、`loaded_chunks`、
`written_texts`、`written_chunks` 和 `new_node_chunks_encoded` 用于确认实际复用情况。
这不改变 embedding 服务本身不可用时的完整历史回退行为。

## 节点奖励与连接

一个 active belief/decision 是一个节点，已有的 `depends_on`、`supplements`、`contradicts`
是可用于连接的边。初始用户输入已经保留在原文上下文中，其图节点奖励为 0，但仍可作为桥梁。
**没有 confidence 阈值过滤**。单个节点连同格式已大于总预算时才因不可行而被排除。

confidence 模式：

```text
prize(v) = clamp(confidence(v), 0, 1)
```

query 模式：

```text
relevance(v) = 0.5 × max_cosine(question_chunks, belief_chunks)
             + 0.5 × max_cosine(focus_chunks, belief_chunks)
prize(v) = relevance(v) × [ε + (1 − ε) × clamp(confidence(v), 0, 1)]
```

余弦相似度的负值截为 0。缺失或非有限的 confidence 按 0.5 处理。
这是上下文选择价值，不是重新估计真实性。`ε=0.2`、`edge_cost=0.2` 是初始参数，尚未调优。

`question` 来自用户输入；`focus` 来自最近一次助手的搜索参数与可读意图，不拼接工具结果正文。
最多取四个去重搜索参数，每个 400 字符，再取最近助手意图的末尾 1,200 字符。
新的用户输入会清空旧助手焦点，内部 Team 报告不会被误认成人类指令。
若没有 focus，就用 question。问题和焦点分别编码，避免长工具正文挤掉原问题。

远程模式将问题、焦点、节点正文按最多 **240 UTF-8 字节**分块，优先在空白处切分，
保留全部文本。这个保守上限适配当前 MiniLM 的 256-token WordPiece 窗口，无需在 Agent
安装 tokenizer；它不等于 240 tokens，也不是任意模型的 token 计数器。切换服务模型时应
根据部署模型的 tokenizer 和窗口重新确定 `embedding_chunk_bytes`。本地模式仍使用模型原生
tokenizer 分块并复核 token 长度。长 belief 的尾部也能参与匹配。

远程请求每批最多 32 个文本块，按响应的 `index` 恢复顺序，校验数量、维度、有限值与非零
范数，再做 L2 归一化和余弦相似度计算。相同文本复用向量；进程内按服务地址、模型、认证、
超时、分块参数隔离缓存，本地模式按模型和设备隔离，每个 encoder 最多缓存 4,096 个完整文本。
编码与缓存更新受锁保护。检索不调用生成模型，也不重新计算 belief confidence。

PCST 目标为：

```text
maximize  Σ prize(v) − edge_cost × |E_selected|
subject to  输出是一棵连通树，且完整渲染长度 ≤ graph_chars
```

连接搜索采用无向邻接，以允许从任意方向找到已有证据链；最终输出仍使用原始边 ID、类型、方向和 note。
同一对端点存在多条边时，选渲染最短的一条，长度相同则按边 ID；本版不对边的语义价值再排序。
未选中的其他关系不会被声明为不存在。没有生成虚构连接，也不传播或修改 confidence。

这可以找回零奖励的中间节点。例如奖励为 A=2、B=0、C=1，已有 A—B—C，边成本 0.2，
树收益是 2.6，高于单选 A 的 2；无奖励的额外叶子则不会增加收益。

## 硬预算的处理

普通 PCST 没有严格字符上限，不能只调一个边成本就宣称预算一定满足。
实现使用 `pcst_fast` 的非定根、单连通分量、strong-pruning 求解：

1. 按原始边成本，以及六种文本长度惩罚强度，各求一棵树。长度代理包含边文本和两端节点文本的一半。
2. 若某棵树超出预算，只移除叶子及其关联边，按损失奖励 / 释放字符数选择，直到满足上限。
   不会单独删除内部桥梁，留下断开的两端。
3. 加入最优可行单节点候选，按原始目标选择预算内收益最高的候选。
4. 使用与模型注入完全相同的 renderer，严格复核节点、关系、置信度、转义和格式的总长度。
   后续 renderer 不再自动删节点；不一致会完整历史回退。

这是**带硬预算的启发式 PCST 适配**，不保证带预算目标的全局最优，也不沿用普通 PCST 的近似比保证。
叶子裁剪不会保证所有初始命中都留下；节点太长、路径太贵时，相关内容仍可能被舍弃。
图有多个不连通分量时只选其中一个，不虚构连接。需要多条独立证据链的任务可能因此损失覆盖，
后续可以另做固定预算的 forest 变体对照。

Team 的默认本地图预算仍为 4,600 字符，跨图引用占另外 2,000 字符；PCST 的连通约束只作用于本地图。
跨图引用的填充规则和 `bcg_recover` 的来源恢复方式不属于本次变更。

## 审计与离线重放

`bcg/requests.jsonl` 现在记录 `selector_result`、`retrieval_strategy` 和 `graph_n_turns_ingested`。
PCST 结果包括奖励、求解调用次数、选择的长度惩罚、裁掉的叶子数、模型及分块统计、精确渲染长度。
远程模式还记录 provider、实际请求 URL、向量维度和新编码块数；不记录 embedding API key。
原有 `selected_node_ids` / `selected_relation_ids` 仍代表最终真正注入的集合。

新运行还会记录所有候选节点的 `node_scores`：参与奖励计算的 confidence、relevance、
confidence 系数、最终 prize、原始输入是否置零、单节点渲染长度及预算可行性。
`graph_chars_budget` 保存该次预算；这些字段只增加审计信息，不改变选择公式。
Case Study Studio 的「逐轮选图」会将这些值与对应历史快照、实际选中集合和 Agent 输出一起展示，
支持完整图高亮、仅选中子图、节点正文与关系详情。旧日志缺失的评分明确显示“未记录”，不重新编码补造。
启动和导入方式见 [Studio 使用说明](../tools/case_study_studio/README.md#inspect-retrieval-by-model-call)。

已有 `state.json` 的 `snapshot` 或独立 snapshot JSON 可用于重放：

```bash
uv run python -m scripts.replay_bcg_pcst \
  --snapshot /path/to/bcg/state.json \
  --question-file /path/to/question.txt \
  --focus '核对小林导师的所属机构' \
  --strategy confidence_pcst query_pcst \
  --budget 6600 \
  --out /path/to/pcst-replay.json
```

也可以用 `--messages` 传入原始模型输入的 JSON 消息数组，自动提取 focus；不要把图构建用的
序列化回合当作原始模型输入。输出保存输入哈希、选择结果和实际注入文本，不调用 Graph 构建或任务模型。
重放默认只运行 `query_pcst`，会访问 embedding 服务；完全离线时使用 `--embedding-provider local`
并准备本地模型。支持 `--embedding-url`、`--embedding-model`、`--embedding-timeout`、
`--embedding-chunk-bytes`、`--embedding-store`、`--embedding-revision`、`--tonggraph-url`、
`--tonggraph-graph`、`--tonggraph-timeout`。脚本加载项目 `.env`，不覆盖已有 shell 环境变量；
未指定的 embedding 参数读取环境变量，再回落到项目默认值。用 `--embedding-store memory` 可关闭本次重放的持久化。

结构测试使用真实 PCST 求解器，无需模型权重：

```bash
uv run --extra dev pytest -q \
  tests/test_bcg_pcst.py tests/test_bcg_embeddings.py tests/test_bcg_vector_store.py \
  tests/test_react_bcg.py tests/test_team_bcg.py
```

比较策略时应固定同一张在线图、问题、当前子目标和最终字符预算。先看桥接证据、覆盖、连通性、
节点与关系数量和检索耗时，再做端到端任务对照。A/B 节点比例、置信度总和或离线结构检查
不能单独证明回答质量改善。confidence-PCST 尤其可能偏向高置信但不切题的内容，应作为对照使用。

## 真实题目验证与已知限制

2026-11-03 从原始问题运行了 DeepResearch Bench II 第 35 题（截至 2021 年 5 月的四旋翼
控制综述），使用 30 轮、8 次搜索、8 次抓取的功能测试预算。该次运行使用本次 PCST 实现和
此前验证的独立 Graph 服务，早于目标分支后来合入的 confidence 改进；不代表合并后版本的质量对照。

- 27 次选图对应 28 个实际带图 HTTP 请求（含一次额外尝试）；图文本与 system 消息逐字匹配。
  所有子图连通且在 6600 字符内，研究阶段没有 BCG 回退。原始问题、系统指令和最近两步均保留。
- 检索内容仍有问题：最后子图的 14 个节点中有 10 个助手计划/操作节点，实质事实覆盖不足。
  81 次来源恢复中有 57 次重复区间；模型达到轮数上限，通过现有收尾机制生成 best-effort 报告。
- 收尾请求直接使用原始 LLM 客户端，不含 `<belief_graph>`。内部总结提示还会进入最终构图输入。
  因此不能把这份输出作为仅依靠 PCST 上下文自主完成的证据。
- 最后构图批次超过 300 秒，客户端记录 `final_graph_flush.status=failed`，服务端随后完成。
  当次剩余向量通过单独读取服务端快照补存；新进程读回 395 段文本、919 个向量，重编码为 0。
  这验证了持久化复用，但不代表末尾自动保存故障已修复。未知写入结果不能直接重发 `/turns`。
- relevance/confidence 奖励没有显式时间约束；该题输出引用了 2022 年资料。短预算、网页访问失败、
  中文 MiniLM 适配和过程节点奖励也影响结果，单题不能证明 PCST 提升答题质量。

后续需优先改善事实与过程记录的检索价值区分、减少重复来源恢复，并处理收尾上下文和末尾保存。
原始轨迹、请求体、凭据及数据库属于本地运行产物，不随代码提交。
