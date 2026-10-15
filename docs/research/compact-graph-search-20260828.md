# Answer-directed Compact Graph selection

## Objective

The Agent-facing Graph is useful only when the retained beliefs and relations make the final answer easier to recover. Maximizing graph connectivity or semantic similarity alone is not the objective. The production decision therefore uses two complementary evaluations:

1. fixed-snapshot replay, which compares selectors on the exact same Graph and Agent state without model sampling noise;
2. end-to-end BrowseComp evaluation, which tests whether the selected Graph and its `<context_blocks_guide>` lead to competitive answers and lower total token use.

Reference answers and final decisions are used only for retrospective metrics. They are never selector inputs.

## Failure analysis from real trajectories

The original Compact renderer mainly ranked facts by confidence and extraction method, ranked searches by recency, and kept a relation only when both endpoints survived. Real trajectories exposed four recurring failures:

- confidence is reliability, not answer relevance, so a generic high-confidence fact can displace a specific answer-bearing belief;
- independent top-k selection can retain a query while dropping the result, or retain a result while dropping its premise;
- pure graph traversal favors large connected components even when they are investigation detours;
- relation labels without their stored rationale can be too opaque for the Agent to understand why two otherwise dissimilar beliefs were connected.

The last failure is important because the Graph already contains a `note` written by relation construction. Omitting it forces the Agent to infer the edge semantics again from abbreviated belief text.

## Controlled replay

The replay implementation is `scripts/analyze_compact_graph_search.py`. It aligns recorded model requests with the exact Graph snapshot and Graph context visible at that request. The final replay set contains 498 snapshots from 94 BrowseComp tasks.

The following strategies were compared under a common visible-character budget:

1. legacy independent ranking;
2. semantic ranking;
3. personalized PageRank;
4. continuous relation chains;
5. cost-aware connected chains;
6. evidence-first chains;
7. focused selection with answer-evidence preservation;
8. answer-directed pruning;
9. focused selection forced even when all eligible nodes already fit.

The evaluator measures source-grounded answer evidence separately from a generic answer mention. It also records support-node recall, top-five placement, relation endpoint retention, component structure, and rendered character cost.

## Fixed-snapshot results

The most informative selectors are shown below. Answer and support metrics use only snapshots in which the corresponding retrospective evidence exists.

| Selector | Mean chars | Answer-evidence recall | Answer top-5 | Support recall | Support top-5 | Relation endpoints retained |
|---|---:|---:|---:|---:|---:|---:|
| Legacy ranking | 4,710 | 75.7% | 47.3% | **100.0%** | **45.6%** | 95.6% |
| Semantic | 4,739 | 83.8% | **51.4%** | 94.0% | 41.6% | 95.2% |
| Personalized PageRank | 4,770 | 86.5% | **51.4%** | 96.0% | 41.6% | 96.7% |
| Continuous chains | 4,879 | 87.8% | 48.6% | 96.0% | 41.6% | **97.5%** |
| Focused, 6,600 chars | 4,585 | 89.2% | 50.0% | 96.0% | 41.6% | 92.9% |
| Current rendered context | 4,589 | **91.9%** | 50.0% | 96.0% | 41.6% | 92.8% |
| Focused on every snapshot | **4,423** | 89.2% | 50.0% | 90.7% | 41.6% | 91.0% |
| Answer-directed pruning | 3,819 | 86.5% | 48.6% | 75.3% | 37.6% | 82.2% |

The fixed replay rejects two appealing but harmful ideas:

- forcing selection on small Graphs saves only about 3.5% visible characters but reduces support recall from 96.0% to 90.7%;
- aggressively pruning toward answer-like nodes reduces visible text further but loses too much supporting evidence and too many relation endpoints.

Continuous chains and PageRank improve structural connectivity, but neither reliably places answer evidence earlier. For example, the current rendered fact order places the first answer-bearing node at mean position 7.33; direct selector order moves it to 12.31, while connected-component path order moves it to 11.50. Consequently, relation traversal is used to retain and explain support paths, not to dictate the complete node order.

## Selected production design

The production Compact view uses focused, query-aware selection with a 6,600-character node budget and an 8,000-character rendered Graph budget.

### Selection

The query combines the permanent initial user question with the retained recent Agent state. The Graph server embeds the question, the focused recent state, and each eligible belief. It then combines:

- question and current-state semantic similarity;
- confidence and recency;
- extraction quality and visible character cost;
- personalized graph importance;
- adjacency to already retained evidence;
- relation coherence.

Empty-search beliefs are excluded from the Agent-facing view. Small Graphs keep all eligible nodes and induced relations. Larger Graphs select factual evidence first, reserve only useful search-history connectors, and retain coherent relations whose endpoints both survive.

### Prompt layout

Selected facts are ordered before search history so candidate values are visible without traversing the entire investigation ledger. Relations are rendered as short endpoint-contiguous paths after the nodes. This preserves the original Graph IDs, belief text, relation type, direction, confidence, and relation semantics; the renderer does not invent summaries or new concepts.

At most four high-priority non-deterministic relations retain their original Graph `note`, with a total note-text budget of 360 characters. Deterministic zero-weight edges do not show a note. In the 100-task run, the Agent received 711 Graph contexts: 411 were non-empty, 337 contained at least one relation rationale, and the rationale text added an average of 139 characters per Graph call.

### Context guidance

`<context_blocks_guide>` is aligned with the selected layout. It tells the Agent that confidence estimates reliability rather than relevance; factual beliefs are candidate evidence; search beliefs are prior work; relations express reasoning/provenance rather than truth; and a relation note explains the link but is not independent evidence. The Agent is instructed to compare candidates against pivotal constraints, resolve only answer-changing gaps, avoid repeating equivalent searches, and stop once one candidate is supported without a decisive contradiction.

Guidance variants that prescribed a benchmark-specific search plan, forced an “exhausted branch” policy, or compared candidates through rigid ordinal constraints failed the fixed 22-task gate and were reverted. The production guide stays Graph-semantic rather than benchmark-specific.

## End-to-end validation

The final validation uses the same fixed 100 BrowseComp tasks for both modes. Agent settings are GPT-5.6-Luna, low thinking effort, 10 workers, no task timeout, and the same search tool. BCG keeps two recent completed turns, injects the Compact Graph into the system prompt, and builds the Graph with GPT-5.6-Luna with thinking disabled.

| Mode | Correct | Accuracy | Agent tokens | Graph tokens | Total tokens | Searches | Mean task time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 40/100 | 40.0% | 3,640,596 | 0 | 3,640,596 | 1,645 | 46.1 s |
| BCG focused + relation rationale | 36/100 | 36.0% | 2,183,629 | 1,369,833 | **3,553,462** | **1,582** | 85.4 s |

BCG reduces Agent-model tokens by 40.0% and total Agent-plus-Graph tokens by 2.4%. It also performs 3.8% fewer searches. The paired outcome is 26 both correct, 10 BCG-only correct, 14 Default-only correct, and 50 both incorrect. An exact paired McNemar test on the 24 discordant tasks gives `p = 0.541`, so this sample does not establish a significant accuracy difference. It also does not establish a general accuracy improvement: the observed point estimate is four tasks lower.

The fixed 22-task preflight gate was favorable—Default 11/22, the previous Compact view 12/22, and the relation-rationale view 13/22—but the complete 100-task result shows why the larger validation is necessary. Relation rationale helps some trajectories, yet its benefit is task-dependent.

## Decision

Keep focused selection, the all-fit fast path, evidence-first rendering, short relation paths, and the bounded original relation rationale. These choices provide the strongest answer-evidence coverage observed in fixed replay while keeping total end-to-end tokens below Default.

Do not replace fact ordering with raw selector order or whole-component traversal, and do not introduce aggressive answer-directed pruning. Those alternatives make the Graph look more connected or smaller but reduce the evidence needed to reach the correct answer.

The remaining accuracy gap should be addressed by improving graph construction quality and by learning when a selected relation path is genuinely answer-discriminating, rather than by deleting more nodes. A future selector evaluation should also score whether the retained evidence satisfies each question constraint, but that metric must remain retrospective and must not leak reference answers into production selection.
