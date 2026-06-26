# Belief Memory PRD v0\.1

# **Belief Memory PRD v0\.1**



## **0\. 文档目的**



本文定义一个面向 AI agents 的 **Belief Memory** 系统：一个参考 Graphiti 的 episode\-first temporal graph 架构，但将其升级为 **belief\-native computation graph** 的长期记忆与工作记忆基础设施。



系统目标不是存储更多文本，也不是把知识图谱加上概率字段，而是让 agent 持续维护：



- 当前应该相信什么；

- 相信到什么程度；

- belief 来自哪些证据；

- 哪些 belief 互相冲突；

- 哪些 belief 已经过期；

- 当前不确定性是否足以阻止自动行动；

- 哪些新证据最能降低不确定性；

- 行动结果如何反向校准未来 belief。

一句话定位：



> Belief Memory is a probabilistic, temporal, evidence\-grounded memory substrate for autonomous agents\.
> 
> 



中文定位：



> Belief Memory 是 agent 的概率化世界模型与决策记忆层。它把 episode、entity、claim、factor、decision、outcome 组织成可计算的 belief graph，使 memory 不只是被检索，而是被持续更新、推断和校准。
> 
> 



**\-\-\-**



## **1\. 背景与问题**



### **1\.1 现有 agent memory 的不足**



当前 agent memory 主流形态包括：



1\. **Conversation memory**：保留历史消息或摘要。

2\. **Vector memory**：用 embedding 检索相关片段。

3\. **GraphRAG / KG memory**：抽取实体和关系，再做图检索。

4\. **Trace memory**：保存 agent runs、tool calls、intermediate steps。

5\. **Temporal KG memory**：记录事实随时间变化，支持 provenance 和 point\-in\-time 查询。



这些系统通常回答：



- 哪些文本相关？

- 哪些事实相关？

- 哪些实体相关？

- 过去发生过什么？

但 agent 在执行真实任务时还需要回答：



- 我是否真的应该相信这个事实？

- 这个事实是否仍然有效？

- 它来自可靠 source 吗？

- 它和其他证据冲突吗？

- 它是否足以支撑行动？

- 行动结果是否证明之前的 belief 是错的？

因此，下一代 agent memory 需要从 **retrieval memory** 升级为 **belief computation memory**。



### **1\.2 参考 Graphiti 的价值**



Graphiti 提供了非常好的系统骨架：



- episode\-first ingestion；

- raw source provenance；

- temporal validity；

- incremental graph update；

- entity / relationship extraction；

- hybrid retrieval；

- custom entity / edge types；

- 面向 agent 的动态 memory graph。

Belief Memory 借鉴这些设计，但核心差异是：



|维度|Graphiti\-style Temporal KG|Belief Memory|
|---|---|---|
|原始输入|Episode|EvidenceEpisode|
|抽取对象|Entity / relationship / fact|Mention / Concept / Entity / Claim|
|事实表示|Edge / fact triplet|Claim \+ TruthBelief|
|时间|fact validity|belief validity \+ posterior history|
|provenance|fact → episode|belief update → evidence \+ factors|
|retrieval|related facts|belief state bundle|
|agent 使用|recall context|infer, decide, ask for evidence, calibrate|
|计算关系|graph traversal / reranking|factor graph / active belief inference|



### **1\.3 设计原则**



1\. **Evidence\-first**：所有 belief 都必须能追溯到 evidence。

2\. **Typed belief**：不同对象上的 belief 语义不同，不能都叫 confidence。

3\. **Claim\-first, not entity\-only**：实体可以有 representation belief，但世界事实必须表示为 claim belief。

4\. **Semantic graph 与 computation graph 分离**：语义关系用于存储和检索；factor 关系用于概率计算。

5\. **Local inference, not global propagation**：每次只编译 active subgraph，避免全图震荡。

6\. **Temporal by default**：所有 evidence、claim、belief、factor weight 都有时间维度。

7\. **Calibration loop**：belief 必须被 outcome 反向校准，否则会退化为装饰性数字。

8\. **Agent\-action aware**：belief 必须影响 agent retrieval、planning、tool use、decision gate。

9\. **Explainable updates**：每次 posterior 变化都要产生 BeliefUpdateTrace。

10\. **Open spec first**：系统应从开放 schema 和 SDK 开始，而不是从 UI 开始。



**\-\-\-**



## **2\. 产品目标与非目标**



### **2\.1 产品目标**



Belief Memory 的目标是提供一个开源 runtime，使 agent 能够：



1. 持续观察来自用户、工具、文档、系统、human feedback 的 evidence。

2. 将 evidence 转换为 mention、concept、entity、claim、decision、outcome。

3. 给每类对象维护 typed belief。

4. 用 factor 显式建模 belief 之间的计算关系。

5. 针对当前任务编译 active belief subgraph。

6. 运行局部增量推断，更新 posterior。

7. 暴露 belief\-aware context 给 agent。

8. 根据 belief threshold gate agent actions。

9. 记录每次 belief update 的解释 trace。

10. 根据 action outcome 校准 source reliability 与 factor weights。

### **2\.2 非目标**



MVP 阶段不做：



1. 通用大规模精确贝叶斯网络推断。

2. 替代 Neo4j、Postgres、Qdrant 等底层存储。

3. 替代 LangGraph / agent orchestration framework。

4. 完全自动从所有业务系统抽取高质量 ontology。

5. 无人监督的因果发现。

6. 对所有领域通用的 factor library。

7. 端到端 closed\-loop 高风险自动执行。

### **2\.3 成功标准**



MVP 成功标准：



1. agent 能把 episode 转换为 typed beliefs。

2. belief update 可追溯、可解释、可回放。

3. 新 evidence 到来时，只更新局部相关 belief。

4. agent context 输出包含 belief、uncertainty、conflict、missing evidence，而不只是文本片段。

5. action decision 能根据 belief threshold 自动区分：auto\-execute、ask\-human、need\-more\-evidence、block。

6. 系统能用 outcome 更新 source reliability 或 factor weight。

7. 在 demo task 上，belief\-aware agent 比 vector\-memory baseline 更少犯 grounding error 和 stale fact error。

**\-\-\-**



## **3\. 用户与使用场景**



### **3\.1 目标用户**



1\. **Agent framework 开发者**：希望给 LangGraph / MCP / OpenAI Agents 加长期 memory。

2\. **AI infra 团队**：希望构建企业 agent memory substrate。

3\. **研究型个人用户**：希望 agent 记住长期研究目标、假设、证据和不确定性。

4\. **企业 workflow agent 团队**：希望 agent 在 deal desk、support escalation、incident response、compliance review 中可审计地行动。



### **3\.2 MVP 场景：Belief\-native Research Agent**



第一个 demo 选择 research agent，因为：



- 不依赖复杂企业权限；

- 用户可以快速验证 belief 是否有用；

- 天然存在假设、证据、冲突、更新、长期研究方向；

- 适合开源传播。

Research Agent 应维护：



- 用户长期研究目标；

- 项目核心假设；

- 文章、论文、repo、experiments 的可靠性；

- 设计判断的 belief；

- 技术路线之间的支持和冲突；

- 尚未验证的问题；

- 下一步最有价值的 evidence。

### **3\.3 第二场景：Incident / Engineering Memory**



第二个 demo 可选择 incident response：



- GitHub issue / PR；

- deploy event；

- PagerDuty incident；

- Slack escalation；

- root cause；

- rollback / hotfix decision；

- outcome。

Agent 可回答：



- 当前 incident 的 root cause belief 是多少？

- rollback 是否 justified？

- 哪些证据支持 hotfix？

- 是否需要 human approval？

- 类似 incident 以前如何处理，结果如何？

**\-\-\-**



## **4\. 概念模型**



### **4\.1 核心对象**



Belief Memory 中有 10 类核心对象：



1\. **EvidenceEpisode**：原始观测或输入。

2\. **MentionNode**：文本或结构化数据中的局部提及。

3\. **ConceptNode**：从 mention 抽象出来的候选概念。

4\. **EntityNode**：canonical entity。

5\. **ClaimNode**：关于世界、用户、任务、偏好、规则、未来结果的陈述。

6\. **SourceNode**：证据来源，如 user、tool、database、document、human、model。

7\. **DecisionNode**：agent 或 human 的行动候选或已执行决策。

8\. **OutcomeNode**：行动后观察到的结果。

9\. **BeliefVariable**：某个对象上的 typed probabilistic state。

10\. **Factor**：belief variables 之间的计算关系。



### **4\.2 关键区分：Graph Object vs Belief Variable**



一个图对象可以有多个 belief variable。



例如：



```Plain Text
EntityNode: account:acme
  - existence(account:acme)
  - type(account:acme = customer)

ConceptNode: concept:acme-from-slack
  - extraction(concept:acme-from-slack)
  - concept_validity(concept:acme-from-slack)

ResolutionEdge: concept:acme-from-slack -> account:acme
  - resolution(concept:acme-from-slack, account:acme)

ClaimNode: claim:acme_churn_risk_high
  - truth(claim:acme_churn_risk_high)
  - effective(claim:acme_churn_risk_high)
  - temporal_validity(claim:acme_churn_risk_high)

DecisionNode: decision:approve_discount
  - justification(decision:approve_discount)
  - safety(decision:approve_discount)
```



不要让 `node.belief = 0.82` 成为唯一抽象。正确抽象是：



```Plain Text
belief_variable(owner=node, kind=truth, posterior=0.82)
```



### **4\.3 Typed Belief 类型**



|Belief Kind|Owner|含义|
|---|---|---|
|extraction|Mention / Concept|是否正确抽取|
|concept\_validity|Concept|是否值得成为稳定概念|
|existence|Entity|实体是否存在|
|type|Entity / Concept|类型是否正确|
|resolution|Concept → Entity|是否正确归一化到 canonical entity|
|truth|Claim|陈述是否为真|
|temporal\_validity|Claim|当前时间是否仍适用|
|effective|Claim|truth × grounding × temporal validity 后的可用 belief|
|source\_reliability|Source|source 在某类 claim 上是否可靠|
|applicability|Policy / Claim|规则是否适用于当前 case|
|justification|Decision|当前证据是否支持该行动|
|safety|Decision|是否足以自动执行|
|outcome\_prediction|Outcome|某结果发生概率|
|calibration|Factor / Source|过去表现是否支持当前权重|



**\-\-\-**



## **5\. 系统架构**



### **5\.1 总体架构**



```Plain Text
External Events / Tool Results / User Messages / Documents
        ↓
Evidence Ingestion API
        ↓
Append-only Evidence Log
        ↓
Extraction Pipeline
  - mention extraction
  - concept extraction
  - claim extraction
  - source attribution
        ↓
Grounding Pipeline
  - entity resolution
  - type assignment
  - canonical mapping
        ↓
Semantic Graph Store
  - episodes
  - mentions
  - concepts
  - entities
  - claims
  - decisions
  - outcomes
        ↓
Belief Variable Store
        ↓
Factor Registry + Factor Instantiator
        ↓
Active Subgraph Compiler
        ↓
Inference Engine
  - log-odds updater
  - temporal decay
  - contradiction handling
  - factor propagation
  - calibration
        ↓
Belief State Store + Update Trace Store
        ↓
Agent Memory API
  - observe
  - believe
  - context
  - decide
  - next_best_evidence
  - calibrate
        ↓
Agent Runtime / LangGraph / MCP / Tools
```



### **5\.2 核心服务**



#### **5\.2\.1 Evidence Service**



职责：



- 接收外部事件；

- 标准化 episode；

- 记录 source、actor、time、provenance；

- 生成 evidence id；

- append\-only 存储。

#### **5\.2\.2 Extraction Service**



职责：



- 从 episode 中抽取 mention；

- 从 mention 中抽取 concept；

- 从文本或 JSON 中抽取 claim；

- 生成初始 extraction belief；

- 建立 evidence → mention / concept / claim 的 provenance。

#### **5\.2\.3 Grounding Service**



职责：



- 将 concept resolution 到 canonical entity；

- 计算 resolution belief；

- 判断 entity type；

- 将 claim subject/object grounding 到 entity；

- 生成 effective belief 的输入变量。

#### **5\.2\.4 Semantic Graph Service**



职责：



- 存储 mention、concept、entity、claim、source、decision、outcome；

- 存储 semantic edges，如 `MENTIONS`、`RESOLVED_TO`、`SUBJECT_OF`、`SUPPORTS`、`CONTRADICTS`、`PRECEDENT_OF`；

- 支持 temporal query 和 provenance query。

#### **5\.2\.5 Belief Store**



职责：



- 存储 BeliefVariable；

- 存储 posterior history；

- 支持 point\-in\-time belief query；

- 支持 active variable index；

- 支持 conflict index。

#### **5\.2\.6 Factor Registry**



职责：



- 管理 factor templates；

- 根据 ontology / claim type / domain 自动实例化 factors；

- 维护 factor weights；

- 维护 factor confidence；

- 支持 factor versioning。

#### **5\.2\.7 Active Subgraph Compiler**



职责：



- 根据 seed variables、task、focal entities、time window、factor whitelist 编译局部计算图；

- 限制 max hops、max variables、max factors；

- 去重 correlation group；

- 生成可被 inference engine 执行的 factor graph。

#### **5\.2\.8 Inference Engine**



职责：



- 执行 belief updates；

- 运行 log\-odds / rule / factor potential / temporal decay；

- 生成 posterior deltas；

- 处理 contradiction；

- 生成 BeliefUpdateTrace；

- 支持 bounded iterative inference。

#### **5\.2\.9 Retrieval / Context Assembler**



职责：



- 从 graph \+ vector \+ full\-text \+ belief store 检索相关对象；

- 根据 belief、uncertainty、recency、source reliability、decision impact rerank；

- 输出 agent 可消费的 belief context bundle。

#### **5\.2\.10 Decision Gate**



职责：



- 接收 action candidate；

- 查询相关 belief；

- 判断 action 是否 auto\-execute、ask\-human、need\-more\-evidence、block；

- 生成决策解释。

#### **5\.2\.11 Calibration Service**



职责：



- 接收 outcome；

- 比较 predicted belief 与 observed outcome；

- 更新 source reliability；

- 更新 factor weight；

- 输出 calibration metrics。

**\-\-\-**



## **6\. 数据存储设计**



### **6\.1 推荐存储组合**



MVP 推荐：



|存储|用途|
|---|---|
|Postgres|evidence log、belief variables、factor registry、update traces、outcomes|
|Neo4j / FalkorDB / Kuzu|semantic graph topology|
|pgvector / Qdrant|episode、claim、entity summary embeddings|
|Object Store|原始文件、长文本、tool artifacts|
|Redis|active inference cache、session working memory|



### **6\.2 逻辑存储层**



1\. **Evidence Log**：append\-only。

2\. **Semantic Graph**：可更新，但保留 historical edges。

3\. **Belief Store**：当前 posterior \+ posterior history。

4\. **Factor Store**：factor template \+ factor instance \+ version。

5\. **Trace Store**：每次 inference run 的解释记录。

6\. **Outcome Store**：行动结果与校准数据。



### **6\.3 Temporal 约定**



所有核心对象应支持：



```Plain Text
observed_at: 该事实在世界中发生或被观察到的时间
ingested_at: 系统接收该 evidence 的时间
valid_from: 该 claim / belief 开始有效的时间
valid_to: 该 claim / belief 结束有效的时间
updated_at: belief posterior 最近更新时间
superseded_by: 被哪个新 claim / belief 替代
invalidated_by: 被哪个 evidence 或 claim 否定
```



**\-\-\-**



## **7\. Spec：核心数据结构**



### **7\.1 EvidenceEpisode Spec**



```JSON
{
  "id": "ev_01J...",
  "schema_version": "0.1",
  "type": "EvidenceEpisode",
  "source": {
    "source_id": "src_slack",
    "source_type": "slack",
    "source_uri": "slack://channel/C123/message/456",
    "reliability_prior": 0.72
  },
  "actor": {
    "actor_id": "user_123",
    "actor_type": "human",
    "role": "account_executive"
  },
  "content": {
    "format": "text",
    "text": "Acme is threatening to churn after repeated outages."
  },
  "timestamps": {
    "observed_at": "2026-05-14T10:00:00Z",
    "ingested_at": "2026-05-14T10:01:00Z"
  },
  "metadata": {
    "conversation_id": "conv_123",
    "domain": "support_escalation",
    "tenant_id": "tenant_abc"
  }
}
```



### **7\.2 MentionNode Spec**



```JSON
{
  "id": "mention_01J...",
  "type": "MentionNode",
  "episode_id": "ev_01J...",
  "span": {
    "text": "Acme",
    "start": 0,
    "end": 4
  },
  "belief_variables": [
    "bv_extraction_mention_01J"
  ],
  "metadata": {
    "extractor": "llm_extractor_v1"
  }
}
```



### **7\.3 ConceptNode Spec**



```JSON
{
  "id": "concept_acme_from_slack_456",
  "type": "ConceptNode",
  "label": "Acme",
  "candidate_types": ["account", "customer", "project"],
  "source_mentions": ["mention_01J..."],
  "belief_variables": [
    "bv_concept_validity_acme",
    "bv_type_acme_account"
  ]
}
```



### **7\.4 EntityNode Spec**



```JSON
{
  "id": "entity_salesforce_account_001xx",
  "type": "EntityNode",
  "entity_type": "account",
  "canonical_source": "salesforce",
  "canonical_id": "001xx",
  "label": "Acme Corp",
  "belief_variables": [
    "bv_entity_exists_acme",
    "bv_entity_type_acme_account"
  ],
  "metadata": {
    "tenant_id": "tenant_abc"
  }
}
```



### **7\.5 ResolutionEdge Spec**



```JSON
{
  "id": "res_acme_concept_to_entity",
  "type": "ResolutionEdge",
  "from": "concept_acme_from_slack_456",
  "to": "entity_salesforce_account_001xx",
  "belief_variable": "bv_resolution_acme_to_sf_001xx",
  "evidence_ids": ["ev_01J..."],
  "features": {
    "name_similarity": 0.91,
    "same_account_owner": true,
    "same_channel_context": true
  }
}
```



### **7\.6 ClaimNode Spec**



```JSON
{
  "id": "claim_acme_churn_risk_high",
  "type": "ClaimNode",
  "claim_type": "hypothesis",
  "subject": "entity_salesforce_account_001xx",
  "predicate": "has_churn_risk",
  "object": "high",
  "natural_language": "Acme Corp has high churn risk.",
  "valid_time": {
    "valid_from": "2026-05-14T00:00:00Z",
    "valid_to": null
  },
  "evidence_ids": ["ev_01J..."],
  "belief_variables": [
    "bv_truth_acme_churn_risk_high",
    "bv_effective_acme_churn_risk_high",
    "bv_temporal_validity_acme_churn_risk_high"
  ],
  "metadata": {
    "domain": "support_escalation"
  }
}
```



### **7\.7 BeliefVariable Spec**



```JSON
{
  "id": "bv_truth_acme_churn_risk_high",
  "type": "BeliefVariable",
  "owner": {
    "owner_id": "claim_acme_churn_risk_high",
    "owner_type": "ClaimNode"
  },
  "belief_kind": "truth",
  "value_space": {
    "type": "boolean",
    "positive_label": true
  },
  "prior": 0.35,
  "posterior": 0.72,
  "uncertainty": 0.14,
  "status": "active",
  "valid_time": {
    "valid_from": "2026-05-14T00:00:00Z",
    "valid_to": null
  },
  "evidence_ids": ["ev_01J..."],
  "factor_ids": ["factor_outages_increase_churn"],
  "last_updated_at": "2026-05-14T10:03:00Z",
  "metadata": {
    "calibration_bucket": "churn_risk_v1"
  }
}
```



### **7\.8 Factor Template Spec**



```JSON
{
  "id": "ft_support_weighted_log_odds",
  "type": "FactorTemplate",
  "factor_type": "support",
  "description": "A source claim supports a target claim using weighted log-odds update.",
  "input_kinds": ["truth"],
  "output_kinds": ["truth"],
  "compute_mode": "log_odds",
  "params_schema": {
    "weight": "number",
    "max_contribution": "number",
    "correlation_policy": "string"
  },
  "version": "0.1"
}
```



### **7\.9 Factor Instance Spec**



```JSON
{
  "id": "factor_outages_increase_churn",
  "type": "Factor",
  "template_id": "ft_support_weighted_log_odds",
  "factor_type": "support",
  "input_variables": [
    "bv_truth_acme_repeated_outages"
  ],
  "output_variables": [
    "bv_truth_acme_churn_risk_high"
  ],
  "weight": 1.4,
  "confidence": 0.78,
  "correlation_group": "customer_sentiment_outage_cluster",
  "activation_condition": {
    "domain": "support_escalation",
    "valid_only_if_input_posterior_gt": 0.5
  },
  "params": {
    "max_contribution": 1.5
  },
  "version": "0.1"
}
```



### **7\.10 DecisionNode Spec**



```JSON
{
  "id": "decision_escalate_acme_to_vp",
  "type": "DecisionNode",
  "action": "escalate_to_vp",
  "target_entity": "entity_salesforce_account_001xx",
  "state": "candidate",
  "required_beliefs": [
    "bv_truth_acme_churn_risk_high",
    "bv_applicability_retention_exception_policy"
  ],
  "belief_variables": [
    "bv_justification_escalate_acme_to_vp",
    "bv_safety_escalate_acme_to_vp"
  ],
  "metadata": {
    "domain": "support_escalation"
  }
}
```



### **7\.11 OutcomeNode Spec**



```JSON
{
  "id": "outcome_acme_renewed_after_escalation",
  "type": "OutcomeNode",
  "related_decision_id": "decision_escalate_acme_to_vp",
  "observed_result": "renewed",
  "success": true,
  "observed_at": "2026-06-10T00:00:00Z",
  "affected_variables": [
    "bv_outcome_prediction_acme_renewal"
  ],
  "metadata": {
    "renewal_amount": 500000
  }
}
```



### **7\.12 BeliefUpdateTrace Spec**



```JSON
{
  "id": "trace_01J...",
  "type": "BeliefUpdateTrace",
  "inference_run_id": "run_01J...",
  "target_variable": "bv_truth_acme_churn_risk_high",
  "previous_posterior": 0.58,
  "new_posterior": 0.72,
  "delta": 0.14,
  "active_factors": [
    "factor_outages_increase_churn",
    "factor_usage_drop_increase_churn"
  ],
  "positive_contributors": [
    {
      "variable": "bv_truth_acme_repeated_outages",
      "contribution": 0.09
    },
    {
      "variable": "bv_truth_acme_usage_drop_gt_30",
      "contribution": 0.06
    }
  ],
  "negative_contributors": [
    {
      "variable": "bv_truth_acme_confirmed_renewal",
      "contribution": -0.01
    }
  ],
  "evidence_ids": ["ev_01J..."],
  "created_at": "2026-05-14T10:03:00Z"
}
```



**\-\-\-**



## **8\. API Spec**



### **8\.1 Python SDK API**



#### **observe**



```Python
memory.observe(
    source_type="message",
    content="Acme is threatening to churn after repeated outages.",
    actor={"id": "user_123", "type": "human", "role": "account_executive"},
    observed_at="2026-05-14T10:00:00Z",
    metadata={"domain": "support_escalation"},
)
```



返回：



```Python
ObserveResult(
    episode_id="ev_01J...",
    extracted_mentions=[...],
    extracted_claims=[...],
    updated_beliefs=[...],
    inference_run_id="run_01J...",
)
```



#### **believe**



```Python
memory.believe("truth(acme_has_high_churn_risk)")
```



返回：



```JSON
{
  "variable_id": "bv_truth_acme_churn_risk_high",
  "posterior": 0.72,
  "uncertainty": 0.14,
  "valid_time": {
    "valid_from": "2026-05-14T00:00:00Z",
    "valid_to": null
  },
  "top_evidence": ["ev_01J..."],
  "top_factors": ["factor_outages_increase_churn"],
  "conflicts": []
}
```



#### **context**



```Python
memory.context(
    task="decide whether to escalate Acme renewal risk",
    focal_entities=["entity_salesforce_account_001xx"],
    max_variables=100,
    include_conflicts=True,
    include_missing_evidence=True,
)
```



返回：



```JSON
{
```

"beliefs": \[*\.\.\.*\],

"conflicts": \[*\.\.\.*\],

"missing\_evidence": \[*\.\.\.*\],

"relevant\_precedents": \[*\.\.\.*\],

"recommended\_actions": \[*\.\.\.*\],

"summary": "Acme likely has elevated churn risk, but VP approval status is unknown\."

\}

```Plain Text

```

#### **decide**



```Python
memory.decide(
    action="escalate_to_vp",
    context={"account": "entity_salesforce_account_001xx"},
    thresholds={
        "auto_execute": 0.9,
        "ask_human": 0.6,
        "block_below": 0.3
    }
)
```



返回：



```JSON
{
  "action": "escalate_to_vp",
  "decision": "ask_human",
  "justification": 0.74,
  "safety": 0.64,
  "missing_evidence": ["VP approval status"],
  "explanation": [
    "Churn risk belief is high",
    "Recent outage evidence is strong",
    "Policy applicability is moderate",
    "Approval status is unknown"
  ]
}
```



#### **next\_best\_evidence**



```Python
memory.next_best_evidence(
    target="justification(escalate_to_vp)",
    context={"account": "entity_salesforce_account_001xx"},
    top_k=5,
)
```



返回：



```JSON
{
  "target": "justification(escalate_to_vp)",
  "candidates": [
    {
      "evidence_type": "tool_call",
      "tool": "billing.get_arr",
      "expected_information_gain": 0.21,
      "reason": "ARR determines policy tier applicability"
    },
    {
      "evidence_type": "human_feedback",
      "ask": "Has VP approved exception?",
      "expected_information_gain": 0.18
    }
  ]
}
```



#### **calibrate**



```Python
memory.calibrate(
    outcome={
        "related_decision_id": "decision_escalate_acme_to_vp",
        "observed_result": "renewed",
        "success": True,
        "observed_at": "2026-06-10T00:00:00Z"
    }
)
```



返回：



```JSON
{
```

"updated\_sources": \[*\.\.\.*\],

"updated\_factors": \[*\.\.\.*\],

"calibration\_trace\_id": "cal\_trace\_01J\.\.\."

\}

```Plain Text

```

### **8\.2 REST API**



|Method|Path|功能|
|---|---|---|
|POST|`/v1/episodes`|observe evidence|
|GET|`/v1/beliefs/{id}`|get belief variable|
|POST|`/v1/context`|assemble belief context|
|POST|`/v1/decisions/evaluate`|evaluate action|
|POST|`/v1/outcomes`|record outcome and calibrate|
|GET|`/v1/traces/{id}`|get belief update trace|
|POST|`/v1/factors`|create factor template / instance|
|GET|`/v1/graph/neighborhood`|inspect semantic or factor neighborhood|



### **8\.3 MCP Tool Interface**



MCP server 暴露：



```Plain Text
belief_memory.observe
belief_memory.believe
belief_memory.context
belief_memory.decide
belief_memory.next_best_evidence
belief_memory.record_outcome
belief_memory.explain_update
```



**\-\-\-**



## **9\. 计算架构**



### **9\.1 两张图：Semantic Graph 与 Factor Graph**



Belief Memory 中必须明确分离两张图。



#### **Semantic Graph**



存储：



- episodes；

- mentions；

- concepts；

- entities；

- claims；

- decisions；

- outcomes；

- provenance；

- temporal validity；

- semantic edges。

典型边：



```Plain Text
Episode --MENTIONS--> Mention
Mention --EXTRACTED_AS--> Concept
Concept --RESOLVED_TO--> Entity
Claim --SUBJECT--> Entity
Claim --SUPPORTED_BY--> Evidence
Claim --CONTRADICTS--> Claim
Decision --USES_CLAIM--> Claim
Outcome --OBSERVED_AFTER--> Decision
```



#### **Factor Graph**



存储 belief variables 与 factors 的计算关系：



```Plain Text
BeliefVariable <-> Factor <-> BeliefVariable
```



典型 factor：



```Plain Text
source reliability + evidence strength -> claim truth
concept validity + resolution -> claim effective belief
claim A truth -> claim B truth
claim truth + policy rule -> decision justification
decision prediction + outcome -> factor calibration
```



### **9\.2 Active Subgraph 编译**



每次 belief update 不跑全图。



输入：



```Plain Text
seed variables
current task
focal entities
focal decision
time window
factor whitelist
max hops
max variables
max factors
minimum delta threshold
```



输出：



```Plain Text
ActiveFactorGraph
  variables
  factors
  evidence bindings
  boundary variables
  frozen variables
  inference config
```



编译流程：



```Plain Text
1. 从 seed variables 开始
2. 查询 factor index，找到直接关联 factors
3. 加入 factor input/output variables
4. 过滤时间无效变量
5. 过滤 domain 不匹配 factors
6. 过滤权限不可见 evidence
7. 按 max_hops 扩展
8. 按 expected impact / recency / relevance 剪枝
9. 合并 correlation group，避免 double counting
10. 输出 active graph
```



### **9\.3 Working Memory 与 Long\-term Memory**



```Plain Text
Long-term Belief Graph
  - all evidence
  - all entities / claims
  - all belief histories
  - all factors
  - all outcomes

Working Belief Graph
  - current task variables
  - current uncertainty
  - current decision candidates
  - active factors
  - short-lived hypotheses
```



Agent 每次运行：



```Plain Text
1. observe new evidence
2. compile working graph
3. infer belief state
4. expose context bundle
5. decide / act
6. record outcome
7. commit deltas to long-term graph
```



**\-\-\-**



## **10\. 算法设计**



### **10\.1 Initial Belief Assignment**



每个新 belief variable 需要 prior。



Prior 来源优先级：



1. domain\-specific prior；

2. source\-specific historical calibration；

3. extractor confidence；

4. entity type prior；

5. global default prior；

6. user\-specified prior。

示例：



```Plain Text
truth(user_explicit_statement) prior = 0.95
truth(model_extracted_claim_from_ambiguous_text) prior = 0.55
entity_exists(from_system_of_record) prior = 1.0
entity_exists(from_llm_extraction_only) prior = 0.75
source_reliability(billing_db_for_arr) prior = 0.99
source_reliability(slack_for_customer_sentiment) prior = 0.65
```



### **10\.2 Log\-Odds Belief Update MVP**



MVP 使用 log\-odds 进行局部可解释更新。



定义：



```Plain Text
logit(p) = log(p / (1 - p))
sigmoid(x) = 1 / (1 + exp(-x))
```



对于目标 belief variable `Y`：



```Plain Text
logit(P_new(Y)) = logit(P_old(Y)) + Σ contribution_i
```



每个 factor 贡献：



```Plain Text
contribution_i = direction_i
               × factor_weight_i
               × source_reliability_i
               × evidence_strength_i
               × input_activation_i
               × correlation_adjustment_i
               × temporal_decay_i
```



其中：



- `direction_i = +1` 表示 support；

- `direction_i = -1` 表示 contradiction；

- `input_activation_i = 2 × posterior(input) - 1`，将 0\.5 作为无信息点；

- `correlation_adjustment_i` 避免同源 evidence 重复计算；

- `temporal_decay_i` 处理过期或衰减。

### **10\.3 Effective Belief**



Claim 的 truth belief 不等于可用于 action 的 belief。



定义工程近似：



```Plain Text
effective_belief(claim)
= truth_belief(claim)
× subject_grounding_belief
× object_grounding_belief
× predicate_schema_validity
× temporal_validity
× type_compatibility
```



示例：



```Plain Text
truth(Acme has churn risk) = 0.82
resolution(Acme -> Acme Corp) = 0.88
temporal_validity = 0.95
type_compatibility = 1.0

effective = 0.82 × 0.88 × 0.95 × 1.0 = 0.686
```



### **10\.4 Entity Resolution Belief**



Entity resolution factor 使用多特征融合。



特征：



```Plain Text
name similarity
identifier exact match
source system match
time proximity
co-occurring entities
conversation / channel context
embedding similarity
historical resolution frequency
human correction history
```



MVP 计算：



```Plain Text
score = sigmoid(
  w_name * name_similarity
+ w_id * exact_id_match
+ w_context * context_similarity
+ w_source * source_match
+ w_history * historical_match
)
```



输出：



```Plain Text
resolution(concept, entity) = score
```



### **10\.5 Claim Extraction Belief**



Claim extraction belief 来自：



```Plain Text
extractor confidence
source reliability
linguistic certainty
explicitness
schema validity
cross-source support
```



示例规则：



```Plain Text
用户显式说 “I prefer concise answers”
  -> truth(user_prefers_concise_answers) = 0.95

模型从长文中推断 “user may prefer concise answers”
  -> truth(user_prefers_concise_answers) = 0.55
```



### **10\.6 Contradiction Handling**



两种 contradiction：



1\. **Hard contradiction**：同一 subject/predicate/object 在同一时间窗口内互斥。

2\. **Soft contradiction**：两个 claim 倾向相反，但可共存。



Hard contradiction 示例：



```Plain Text
claim: account status = active
claim: account status = closed
```



Soft contradiction 示例：



```Plain Text
claim: customer has high churn risk
claim: customer confirmed renewal verbally
```



MVP 处理：



```Plain Text
contradiction_factor(A, B)
  - 如果 A 上升，B 按权重下降
  - 如果 B 上升，A 按权重下降
  - 如果两个都高，生成 conflict object
```



Conflict object：



```JSON
{
  "id": "conflict_123",
  "claims": ["claim_a", "claim_b"],
  "severity": 0.76,
  "needs_resolution": true,
  "suggested_evidence": ["check canonical system", "ask human owner"]
}
```



### **10\.7 Temporal Decay**



不同 claim 类型有不同半衰期。



示例：



|Claim Type|Half\-life|
|---|---|
|user long\-term preference|180 days|
|current task goal|session\-level|
|active incident status|hours / days|
|customer sentiment|days / weeks|
|legal policy|until superseded|
|canonical account identity|no decay|



Decay 公式：



```Plain Text
p_t = base + (p_0 - base) × exp(-λ × Δt)
```



其中 `base` 通常为 prior 或 0\.5。



### **10\.8 Policy Factor**



Policy factor 把 claim belief 转换为 applicability 或 decision justification。



示例：



```Plain Text
IF customer_tier = enterprise
AND churn_risk_high
AND recent_sev1_incident
THEN retention_exception_policy_applies
```



MVP 可用 rule \+ fuzzy threshold：



```Plain Text
policy_applicability = min(
  effective(customer_tier_enterprise),
  effective(churn_risk_high),
  effective(recent_sev1_incident)
)
```



或 weighted logit：



```Plain Text
logit(policy_applies) = bias + Σ w_i × logit(input_i)
```



Policy engine 可选集成 OPA：



- OPA 输出 allow / deny / required\_approval / reason；

- Belief Memory 将 OPA 结果作为 policy evidence；

- policy applicability belief 仍由 evidence 与 factors 维护。

### **10\.9 Decision Gate Algorithm**



Decision Gate 输入：



```Plain Text
action candidate
required beliefs
risk level
policy requirements
user permission
human approval state
expected utility
```



输出：



```Plain Text
auto_execute
ask_human
need_more_evidence
block
```



MVP 规则：



```Plain Text
if safety < block_threshold:
    block
elif missing_required_evidence:
    need_more_evidence
elif justification >= auto_threshold and safety >= safety_threshold and policy_allows:
    auto_execute
elif justification >= ask_human_threshold:
    ask_human
else:
    need_more_evidence
```



### **10\.10 Next Best Evidence**



目标：找到最能降低目标变量 uncertainty 的证据。



MVP 方法：



1. 找到 target variable 的 Markov blanket / active neighborhood。

2. 找到高权重 factor 中缺失或低置信 input variable。

3. 估算 expected information gain。

4. 映射到 tool call 或 human question。

启发式 EIG：



```Plain Text
EIG(candidate) = factor_weight
               × current_uncertainty(target)
               × missing_input_importance
               × source_reliability(candidate_source)
               × action_impact
               / cost(candidate)
```



### **10\.11 Calibration Algorithm**



Outcome 到来时：



1. 找到相关 decision。

2. 找到 decision 使用的 target belief variables。

3. 找到当时 posterior snapshot。

4. 比较 predicted vs observed。

5. 更新 factor weight 和 source reliability。

MVP 更新 source reliability：



```Plain Text
reliability_new = reliability_old + η × (outcome_correctness - predicted_probability) × source_contribution
```



MVP 更新 factor weight：



```Plain Text
weight_new = weight_old + η × prediction_error × factor_contribution
```



需要做 clipping：



```Plain Text
weight ∈ [min_weight, max_weight]
reliability ∈ [0.05, 0.99]
```



### **10\.12 Bounded Iterative Inference**



对于 active graph 中的循环，MVP 使用 bounded loopy update：



```Plain Text
for iter in max_iters:
    for factor in active_factors:
        compute output delta
        accumulate updates
    apply damping
    if max_delta < convergence_threshold:
        break
```



防震荡机制：



```Plain Text
damping = 0.5
max_delta_per_iter = 0.2
min_delta_to_commit = 0.01
max_iters = 20
```



**\-\-\-**



## **11\. Retrieval 与 Context Assembly**



### **11\.1 Belief\-aware Retrieval**



检索不是只找相关文本，而是组装当前任务的 belief state。



候选来源：



```Plain Text
semantic similarity over episodes / claims
BM25 over raw episodes
entity graph neighborhood
factor graph neighborhood
recently updated beliefs
high uncertainty beliefs
conflict objects
related decisions / outcomes
```



### **11\.2 Reranking Score**



```Plain Text
score = α × semantic_relevance
      + β × graph_proximity
      + γ × belief_strength
      + δ × uncertainty_importance
      + ε × recency
      + ζ × source_reliability
      + η × decision_impact
      - θ × staleness
```



### **11\.3 Context Bundle 输出格式**



```JSON
{
  "task": "...",
```

"focal\_entities": \[*\.\.\.*\],

"belief\_summary": "\.\.\.",

"high\_confidence\_beliefs": \[*\.\.\.*\],

"uncertain\_beliefs": \[*\.\.\.*\],

"conflicts": \[*\.\.\.*\],

"missing\_evidence": \[*\.\.\.*\],

"decision\_recommendations": \[*\.\.\.*\],

"source\_notes": \[*\.\.\.*\],

"raw\_evidence\_refs": \[*\.\.\.*\]

\}

```Plain Text

```

### **11\.4 Agent Prompt Contract**



给 agent 的 context 不应只是一堆 facts，而应包含：



```Plain Text
Known with high confidence:
- ...

Likely but uncertain:
- ...

Conflicting:
- ...

Missing evidence:
- ...

Action guidance:
- safe to do X
- ask human before Y
- block Z
```



**\-\-\-**



## **12\. Agent Integration**



### **12\.1 LangGraph Integration**



提供节点：



```Plain Text
BeliefObserveNode
BeliefContextNode
BeliefDecisionGateNode
BeliefOutcomeNode
```



典型 flow：



```Plain Text
User input
  -> BeliefObserveNode
  -> BeliefContextNode
  -> Planner
  -> BeliefDecisionGateNode
  -> Tool execution or human approval
  -> BeliefOutcomeNode
```



### **12\.2 MCP Integration**



将 memory 暴露为 MCP server，使任意 agent 可调用。



Tools：



```Plain Text
observe_evidence
get_belief
assemble_context
evaluate_action
find_next_best_evidence
record_outcome
explain_belief_update
```



### **12\.3 OpenTelemetry Integration**



可选：将 model calls、tool calls、agent spans 转换为 EvidenceEpisode。



映射：



```Plain Text
LLM span -> EvidenceEpisode(type=model_call)
Tool span -> EvidenceEpisode(type=tool_call)
Agent step span -> DecisionNode / EvidenceEpisode
Error span -> OutcomeNode / negative evidence
```



### **12\.4 OpenLineage\-style Export**



可选：导出 belief lineage：



```Plain Text
belief variable
  derived from evidence episodes
  via factor instances
  produced by inference run
  consumed by decision
```



**\-\-\-**



## **13\. Security, Privacy, Governance**



### **13\.1 权限模型**



Belief Memory 存储敏感 evidence 和 derived belief，因此必须支持：



```Plain Text
tenant isolation
source-level ACL
entity-level ACL
claim-level ACL
decision-level ACL
redaction
retention policy
audit log
```



### **13\.2 Derived Belief 权限继承**



如果一个 belief 来自多个 evidence，其可见性应为：



```Plain Text
viewer can see belief only if viewer can access sufficient supporting evidence
```



MVP 简化：



```Plain Text
belief visibility = intersection / most restrictive source visibility
```



### **13\.3 Human Feedback 与 Correction**



支持：



```Plain Text
mark claim as wrong
mark resolution as wrong
set belief manually
pin canonical source
invalidate evidence
create correction evidence
```



所有人工修正都应作为 EvidenceEpisode 记录，而不是直接覆盖。



**\-\-\-**



## **14\. MVP 范围**



### **14\.1 MVP 必须实现**



1. EvidenceEpisode ingestion。

2. Mention / Concept / Entity / Claim 基础模型。

3. BeliefVariable store。

4. Factor template / instance store。

5. Log\-odds update engine。

6. Temporal decay。

7. Entity resolution belief。

8. Claim truth belief。

9. Effective belief。

10. Active subgraph compiler。

11. BeliefUpdateTrace。

12. `observe` / `believe` / `context` / `decide` API。

13. Belief\-native research agent demo。

14. Golden test suite。

### **14\.2 MVP 不做**



1. 大规模 loopy BP 生产化。

2. 自动 ontology induction。

3. 完整 UI。

4. 企业权限深度集成。

5. 多租户 SaaS。

6. 因果推断。

### **14\.3 V1\.0 目标**



1. LangGraph adapter。

2. MCP server。

3. Neo4j / Falkor / Kuzu backend 之一。

4. Postgres backend。

5. calibration service。

6. conflict resolution UI / CLI。

7. incident response demo。

8. benchmark report。

**\-\-\-**



## **15\. 开发 Roadmap**



### **Phase 0：Spec 与 Prototype**



周期：2–3 周



产出：



- schema JSON；

- Python dataclasses / Pydantic models；

- in\-memory graph；

- log\-odds updater；

- toy research agent demo。

### **Phase 1：Core Runtime**



周期：4–6 周



产出：



- Postgres evidence / belief store；

- semantic graph backend；

- factor registry；

- active subgraph compiler；

- belief update traces；

- API server。

### **Phase 2：Agent Integrations**



周期：4–6 周



产出：



- LangGraph nodes；

- MCP server；

- context assembler；

- decision gate；

- next\-best\-evidence。

### **Phase 3：Calibration 与 Benchmarks**



周期：6–8 周



产出：



- outcome ingestion；

- source reliability update；

- factor weight update；

- calibration metrics；

- benchmark datasets；

- evaluation dashboard。

### **Phase 4：Production Hardening**



周期：8–12 周



产出：



- ACL；

- multi\-tenant support；

- incremental indexing；

- observability；

- replay tool；

- UI explorer。

**\-\-\-**



## **16\. 测试方案**



### **16\.1 测试分层**



```Plain Text
Unit Tests
Integration Tests
Inference Correctness Tests
Temporal Tests
Calibration Tests
Retrieval Tests
Agent Behavior Tests
Regression / Golden Tests
Performance Tests
Adversarial Tests
```



### **16\.2 Unit Tests**



覆盖：



1. schema validation；

2. belief variable creation；

3. logit / sigmoid correctness；

4. log\-odds update；

5. effective belief computation；

6. temporal decay；

7. factor activation condition；

8. conflict detection；

9. trace generation；

10. serialization / deserialization。

示例：



```Plain Text
Given prior = 0.5
And one support factor contribution = +1.0
Then posterior = sigmoid(1.0) ≈ 0.731
```



### **16\.3 Ingestion Tests**



测试输入：



```Plain Text
plain text episode
chat message episode
tool call JSON episode
human correction episode
outcome episode
```



验证：



```Plain Text
episode persisted
mentions extracted
claims created
belief variables initialized
provenance connected
inference run triggered
```



### **16\.4 Entity Resolution Tests**



构造 cases：



1. exact id match；

2. fuzzy name match；

3. ambiguous entity；

4. same name different org；

5. human correction；

6. stale resolution invalidation。

指标：



```Plain Text
resolution accuracy
false merge rate
false split rate
calibration of resolution belief
```



### **16\.5 Belief Update Tests**



Golden cases：



#### **Case A：support evidence increases belief**



```Plain Text
prior churn risk = 0.35
new evidence: repeated outage, strength high
expected posterior > prior
```



#### **Case B：contradictory evidence lowers belief**



```Plain Text
current churn risk = 0.8
new evidence: customer signed renewal
expected posterior lower
conflict object created if both remain high
```



#### **Case C：grounding affects effective, not truth**



```Plain Text
truth claim = 0.8
resolution = 0.5
effective = 0.4
when resolution rises to 0.9
truth stays ~0.8
effective rises to ~0.72
```



#### **Case D：temporal decay**



```Plain Text
active incident claim = 0.9
no update for 14 days
expected posterior decays toward prior or expires
```



### **16\.6 Active Subgraph Tests**



验证：



```Plain Text
max_hops enforced
max_variables enforced
time_window enforced
factor whitelist enforced
permission filtering enforced
correlation group dedup works
boundary variables frozen
```



### **16\.7 Retrieval / Context Tests**



Benchmark queries：



1. current task requires high\-confidence beliefs；

2. current task requires uncertain beliefs；

3. current task involves conflict；

4. stale fact exists；

5. wrong entity resolution exists；

6. missing evidence needed。

指标：



```Plain Text
Recall@K for relevant beliefs
NDCG@K with belief-aware relevance
stale fact suppression rate
conflict surfacing rate
missing evidence precision
context token efficiency
```



### **16\.8 Decision Gate Tests**



Scenarios：



1. high justification \+ high safety → auto\_execute；

2. high justification \+ missing approval → ask\_human；

3. low safety → block；

4. high uncertainty → need\_more\_evidence；

5. policy denies → block；

6. user explicitly authorizes → safety increases but policy still checked。

指标：



```Plain Text
decision accuracy
unsafe auto-execute rate
unnecessary human escalation rate
missing evidence detection rate
```



### **16\.9 Calibration Tests**



输入：



```Plain Text
predicted probability = 0.8
observed outcome = false
```



验证：



```Plain Text
Brier score computed
source reliability adjusted downward if source contributed
factor weight adjusted downward if factor contributed
calibration trace created
```



指标：



```Plain Text
Brier score
Expected Calibration Error
reliability curve
factor weight stability
source reliability convergence
```



### **16\.10 Agent Behavior Tests**



对比 baseline：



1. vector memory agent；

2. GraphRAG\-like memory agent；

3. Belief Memory agent。

任务：



```Plain Text
answer with stale conflicting facts
choose action under uncertainty
ask for missing evidence
avoid wrong entity grounding
gate unsafe action
update after correction
```



成功指标：



```Plain Text
fewer stale fact errors
fewer wrong entity errors
higher conflict awareness
higher appropriate refusal / ask-human rate
better outcome prediction calibration
```



### **16\.11 Adversarial Tests**



测试：



1. duplicate evidence from same source；

2. same claim paraphrased multiple times；

3. malicious low\-reliability source；

4. source conflict；

5. prompt injection inside evidence；

6. outdated document contradicting fresh system\-of\-record；

7. ambiguous entity names。

验证：



```Plain Text
double counting controlled
source reliability matters
fresh canonical source wins when configured
conflict surfaced instead of silently averaged
unsafe injected instruction not converted into high-trust policy
```



### **16\.12 Performance Tests**



目标指标 MVP：



|操作|目标|
|---|---|
|observe small episode|\< 2s excluding LLM extraction|
|belief lookup|\< 100ms|
|context assembly|\< 1s for 100 variables|
|active graph compile|\< 500ms for 500 variables|
|local inference|\< 1s for 500 variables / 1000 factors|
|trace retrieval|\< 200ms|



压力测试：



```Plain Text
1M episodes
10M belief variables
50M factor edges
100 concurrent agents
```



MVP 可以先不达成全部压力目标，但要定义索引策略。



**\-\-\-**



## **17\. 评测数据集设计**



### **17\.1 Synthetic Belief Graph Dataset**



生成：



```Plain Text
entities
claims
sources with reliability
factors with known weights
outcomes
conflicts
stale facts
ambiguous entity mappings
```



用于验证算法是否接近 ground truth。



### **17\.2 Research Agent Dataset**



包含：



```Plain Text
用户长期项目目标
多篇文章摘要
互相冲突的技术判断
source reliability 标签
后续用户 correction
```



任务：



```Plain Text
总结当前 belief state
指出不确定问题
推荐下一步 evidence
更新过时 belief
```



### **17\.3 Incident Dataset**



包含：



```Plain Text
incident logs
deploy events
GitHub PRs
Slack messages
PagerDuty alerts
root cause labels
action decisions
outcomes
```



任务：



```Plain Text
判断 root cause belief
判断 rollback 是否 justified
找缺失证据
比较相似 incident
```



**\-\-\-**



## **18\. Metrics**



### **18\.1 Belief Quality Metrics**



```Plain Text
Brier Score
Expected Calibration Error
Negative Log Likelihood
AUROC for binary claims
AUPRC for rare claims
posterior stability
uncertainty reduction after evidence
```



### **18\.2 Retrieval Metrics**



```Plain Text
Recall@K
Precision@K
NDCG@K
conflict surfacing rate
stale suppression rate
missing evidence precision
```



### **18\.3 Decision Metrics**



```Plain Text
action accuracy
unsafe auto-execute rate
false block rate
human escalation precision
expected utility realized
outcome prediction calibration
```



### **18\.4 System Metrics**



```Plain Text
inference latency
active graph size
delta propagation count
storage growth
trace size
cache hit rate
```



**\-\-\-**



## **19\. CLI / DevEx**



### **19\.1 CLI Commands**



```Bash
beliefgraph init
beliefgraph observe --file episode.json
beliefgraph belief get bv_truth_acme_churn_risk_high
beliefgraph context --task "decide Acme escalation"
beliefgraph graph active --seed bv_truth_acme_churn_risk_high
beliefgraph trace show trace_01J
beliefgraph factor list
beliefgraph factor create factor.yaml
beliefgraph eval run golden_suite.yaml
```



### **19\.2 Local Dev Setup**



```Bash
docker compose up postgres neo4j qdrant
pip install beliefgraph
beliefgraph init
beliefgraph observe --text "User wants a belief graph memory system."
beliefgraph context --task "design PRD"
```



### **19\.3 Example Repo Structure**



```Plain Text
beliefgraph/
  core/
    episode.py
    mention.py
    concept.py
    entity.py
    claim.py
    belief_variable.py
    factor.py
    decision.py
    outcome.py
  ingestion/
    extract.py
    resolve.py
    ground.py
  inference/
    log_odds.py
    temporal_decay.py
    contradiction.py
    active_subgraph.py
    calibration.py
  retrieval/
    hybrid_search.py
    belief_reranker.py
    context_assembler.py
  agent/
    memory_api.py
    decision_gate.py
    langgraph_nodes.py
    mcp_server.py
  storage/
    postgres.py
    graph_backend.py
    vector_backend.py
  tests/
    golden/
    synthetic/
    integration/
  examples/
    research_agent/
    incident_agent/
```



**\-\-\-**



## **20\. Open Questions**



1. Claim 是否应作为 node 还是 edge？MVP 建议 ClaimNode，因为 claim 本身需要 belief、validity、evidence、factor。

2. Factor 是否存储在 graph DB 还是 Postgres？MVP 建议 Postgres 为主，graph DB 建索引用于 traversal。

3. 是否引入严格概率图算法？MVP 先 log\-odds，V1 再加 loopy BP。

4. LLM extractor 的 confidence 如何校准？需要 golden extraction set 和 human correction loop。

5. 如何防止 belief 被模型自信污染？LLM output 必须作为低到中等 reliability evidence，不能直接当 ground truth。

6. 不同 domain 的 factor library 如何管理？建议按 package 分发：`beliefgraph-domain-research`、`beliefgraph-domain-incident`、`beliefgraph-domain-sales`。

7. 是否支持 causal belief？MVP 先区分 predictive factor 和 causal factor，但不做自动因果推断。

8. 用户偏好 belief 是否需要特殊隐私策略？需要，因为偏好和个人画像属于敏感 derived data。

**\-\-\-**



## **21\. 示例：Research Agent 的端到端流程**



用户输入：



```Plain Text
我希望构建大型 belief computation graph，作为 agent 的核心底层 memory 架构。
```



### **Step 1: EvidenceEpisode**



```Plain Text
ev_user_goal_belief_memory
source = user
reliability = 1.0
```



### **Step 2: Concepts**



```Plain Text
concept: belief computation graph
concept: agent memory architecture
concept: probabilistic world model
```



### **Step 3: Claims**



```Plain Text
claim:user_goal_build_belief_memory
claim:belief_graph_should_be_agent_core_memory
claim:memory_should_compute_not_only_store
```



### **Step 4: Beliefs**



```Plain Text
truth(user_goal_build_belief_memory) = 0.99
truth(memory_should_compute_not_only_store) = 0.92
truth(graphiti_is_useful_reference) = 0.86
```



### **Step 5: Factors**



```Plain Text
user_explicit_statement -> user_goal_claim
previous_context_supports -> design_direction_claim
technical_consistency -> architecture_confidence
```



### **Step 6: Context Bundle**



Agent receives:



```Plain Text
High confidence:
- User is designing a belief-native agent memory system.
- User wants a system architecture and PRD, not generic explanation.

Likely:
- Graphiti should be used as structural reference, not copied directly.

Uncertain:
- Best inference backend for V1.
- Whether first demo should be research agent or incident agent.

Recommended response:
- Write full PRD with spec, algorithms, architecture, testing.
```



**\-\-\-**



## **22\. 最终产品判断**



Belief Memory 的核心不是“图 \+ 概率”，而是：



```Plain Text
Evidence Log
  -> Typed Belief Variables
  -> Factorized Computation
  -> Local Inference
  -> Agent Context
  -> Decision Gate
  -> Outcome Calibration
```



它把 agent memory 从：



```Plain Text
What should I retrieve?
```



升级为：



```Plain Text
What should I believe?
How strongly?
Why?
What changed?
What is uncertain?
Can I act?
What should I observe next?
```



这就是该项目的长期差异化。





