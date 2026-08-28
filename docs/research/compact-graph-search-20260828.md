# Query-aware connected Compact Graph selection

## Question

When a Belief Context Graph is larger than the Agent-facing context budget, which existing nodes and relations should be retained so that the resulting context is both compact and useful for reaching the final answer?

The previous Compact Graph renderer ranked factual beliefs mainly by confidence and extraction method, ranked search-history beliefs by recency, and displayed relations only when both endpoints happened to survive. It did not use the current Agent state as a query and did not complete relation paths. This can retain individually strong nodes while severing the query → result → hypothesis chain that makes them useful.

## Controlled replay

The replay tool is `scripts/analyze_compact_graph_search.py`. It aligns each recorded model request with the exact Graph context trace and Graph snapshot that was visible at that request. Every selector receives the same snapshot and the same Agent state. Reference answers and final decisions are used only for evaluation, never as selector input.

Two recorded BrowseComp runs were replayed:

- pseudo-dialogue injection: 96 tasks and 519 aligned non-empty Graph requests;
- block injection: 94 tasks and 480 aligned non-empty Graph requests.

The compared selectors were:

1. legacy independent ranking;
2. semantic ranking against the initial question plus recent Agent state;
3. personalized PageRank;
4. unrestricted continuous relation chains;
5. cost-aware connected chains.

The cost-aware selector combines semantic similarity, personalized PageRank, recency, confidence, extraction quality, and visible character cost. It starts from query-relevant/recent seeds, follows existing directed relations for at most four hops, limits verbose raw Tool Results, and fills remaining budget with high-value nodes adjacent to the retained subgraph. It selects only existing Graph content.

## Fixed-snapshot result

The table pools the two replay sets with each metric weighted by the number of eligible snapshots. Answer and support recall use only snapshots in which the corresponding retrospective evidence exists.

| Selector | Mean chars | Answer-node recall | Final support-chain recall | Largest connected component | Isolated nodes | Relation endpoint retention |
|---|---:|---:|---:|---:|---:|---:|
| Legacy ranking | 4,630 | 86.6% | 93.3% | 67.7% | 10.2% | 96.2% |
| Semantic | 4,682 | 91.6% | 94.6% | 67.8% | 9.8% | 96.2% |
| Personalized PageRank | 4,709 | 89.4% | 95.4% | 72.7% | 8.5% | 97.3% |
| Continuous chains | 4,807 | 93.5% | 99.7% | 74.2% | 8.0% | 97.9% |
| Cost-aware connected chains | 4,720 | **94.6%** | **96.0%** | **72.9%** | **8.7%** | **97.4%** |

Relative to legacy ranking, cost-aware connected chains increased mean visible characters by 1.9%, answer-node recall by 8.0 percentage points, support-chain recall by 2.7 points, and largest-component ratio by 5.2 points. Pure continuous chains maximized support completeness, but consumed more space and sometimes let long raw results crowd out direct distilled facts. The cost-aware variant was more robust across both recorded runs.

Small Graphs receive a special treatment: when every eligible node already fits the node budget, all eligible nodes and induced relations are retained. This removed the two small-Graph regressions found during replay.

## Real trajectory case: Oware

In `browsecomp-0716`, the final relevant Graph state contained:

- B40: the Assistant searched for `Oware "graduate" "2012" game`;
- B46: the resulting Tool Result contained multiple snippets explicitly mentioning Oware;
- an existing relation `B46 depends_on B40`, recording that the evidence was produced by that query.

On the same call-8 snapshot, legacy ranking retained B40 but omitted the long raw result B46. Its answer-node recall was 50%, and the largest connected component covered 36.4% of selected nodes. The production connected selector retained both endpoints and their relation, reaching 100% answer-node recall and a 71.0% largest-component ratio. The Agent subsequently returned `FINAL ANSWER: Oware` correctly. This is the desired behavior: a verbose node is not retained merely because it is recent, but because a relevant query and its result form a continuous answer-bearing chain.

## Production implementation

The Graph Server exposes `POST /context-selection`. For each Agent request, the Compact Graph query is:

1. the permanent initial user question;
2. the retained recent turns;
3. truncated from the oldest transient content if it exceeds 12,000 characters.

Raw Tool Result nodes are retrieved at snippet granularity, while the displayed Graph remains the original node. The selector returns existing node and relation IDs; the Agent renderer then injects those exact Graph items using the existing pseudo-dialogue wrapper. `connected` is the default selector, while `ranked` remains available as a reproducible legacy baseline. If an older or unavailable Graph Server cannot select context, the Agent falls back to ranked rendering.

Two embedding defects were found by the end-to-end test and fixed:

- concurrent `SentenceTransformer.encode` calls from HTTP workers were serialized;
- configured maximum length can lower, but can no longer raise, the model's native sequence limit (`all-MiniLM-L6-v2` was incorrectly configured as 8192 despite a native limit of 256).

A four-worker long-text smoke test completed eight concurrent requests with zero errors after the fix.

## End-to-end validation

The same eight representative BrowseComp tasks were run with GPT-5.6-Luna at low thinking effort, recent turns = 2, Compact Graph injection, four workers, no task timeout, and the same Graph construction configuration.

| Selector run | Correct | Accuracy | Agent + Graph tokens | Searches | Mean task time |
|---|---:|---:|---:|---:|---:|
| Ranked run 1 | 6/8 | 75.0% | 302,658 | 114 | 91.5 s |
| Ranked run 2 | 5/8 | 62.5% | 254,239 | 105 | 85.0 s |
| Connected, clean run | 6/8 | **75.0%** | **198,945** | **102** | **85.4 s** |

The two ranked runs show substantial sampling variance, so the causal selector evidence comes from fixed-snapshot replay rather than raw accuracy alone. Nevertheless, the clean connected run matched the best ranked accuracy, exceeded the two-run ranked mean of 68.75%, and used 28.6% fewer Agent + Graph tokens than the ranked mean. Failed mixed-selector runs produced before the embedding fixes are intentionally excluded.

Only two requests in the clean eight-task run exceeded the node budget and required embedding-based selection; the other non-empty Graphs fit completely. The full 999-snapshot replay therefore remains the broader selector evaluation, while the end-to-end run verifies that the production path works without fallbacks and can preserve an answer-bearing relation chain in a real Agent trajectory.

## Recommendation

Use cost-aware connected selection for Compact Graph view and keep legacy ranking as an ablation/fallback. Do not use pure semantic top-k or unconstrained traversal alone. The useful unit is a small relation-connected evidence path, but path utility must be divided by visible character cost so that long raw search results survive only when they carry unique evidence or complete a high-value chain.
