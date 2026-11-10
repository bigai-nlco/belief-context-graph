# Unified BCG: BRACE-inspired confidence policy (provisional)

## Execution plan

1. Change the Unified extractor to seven verbal labels. The old four names are not aliases; unsupported values normalize to `unknown`.
2. Compute the initial prior from a stance anchor multiplied by the message role or the attributed Tool Result source coefficient. Keep `unknown` neutral at 0.5. This is an explicit BCG adaptation, not a probability table copied from BRACE.
3. Apply the same mapping to *independent later* evidence and discount its log odds by 0.25. Creation-time provenance does not increase confidence. Keep the existing relation propagation policy unchanged so the change can be isolated in evaluation.
4. Attribute search claims by model-selected `source_rank` validated against the parser's result list. Resolve `web_fetch` from its unique `Source:` URL. Keep mixed and missing provenance in a conservative fallback; record category, host, URL, rank, and coefficient on the node and evidence.
5. Compare default and changed policy with the same model, tasks, tool budget, and judge. Score calibration on verified claim labels using Brier score/reliability bins; inspect false merges, incorrect source ranks, blocked-source citations, and graph retrieval changes before tuning numeric anchors.

## Values and interpretation

| Certainty label | BCG anchor | Intended use |
| --- | ---: | --- |
| confirmed | 0.94 | Directly checked against a relevant primary observation in the current step |
| almost_certain | 0.86 | Checked previously with no later counterevidence |
| probable | 0.74 | Strong inference with explicit premises |
| possible | 0.60 | Plausible, unverified but supported possibility |
| unlikely | 0.35 | Checked with little supporting evidence |
| doubtful | 0.20 | Specific contradictory evidence |
| unknown | 0.50 | No usable evidence; neutral rather than false |

The source role multiplier is user=0.95, assistant=0.72, tool fallback=0.60. A selected tool result overrides the tool multiplier: institution's own primary source 0.88, scholarly publisher/repository 0.78, encyclopedia 0.68, general web 0.58, community/forum 0.42, narrowly listed tabloid/entertainment sites 0.38, unattributed/mixed 0.50. These are **hand-picked research priors, not measured probabilities**. A publisher may host a retracted paper; an official site may describe something outside its expertise; a forum may reproduce a primary document. An URL category never upgrades an unsupported assertion to `confirmed` by itself. The allowlist is deliberately narrow and should be calibrated by topic. Unknown always remains 0.5, regardless of source.

`initial_confidence = stance_anchor * role_or_source_multiplier` (except unknown=0.5). Later independent corroborating evidence uses `max(0, 0.25 * logit(initial_confidence(evidence)))`; an untrusted duplicate cannot act as counterevidence. Explicit contradictory claims use `contradicts` relations. Posterior and relation propagation otherwise use existing equations. The source-specific multiplier is attached to each evidence record so a later merge retains the absorbed fact's own origin, rather than inheriting the canonical node's origin.

BRACE supplies an ordinal verbal vocabulary and illustrates nominal probabilities (confirmed ~1, probable ~0.75, possible ~0.5, doubtful ~0.25, unknown ~0). It does **not** establish the above BCG mapping. Here `unknown` is neutral because the user's intended meaning is absence of evidence, not a false claim. The mapping is consequently not strictly ordinal at the final two levels. Do not describe these scores as empirically calibrated likelihoods until a claim-level validation study has been done.

Unified does not load the DeBERTa zero-shot classifier listed under `pipeline.stance` in the example configuration. That block serves the Hybrid backend. Unified asks the configured Graph model to supply the seven-label stance. Historical graph exports are not rescored on read.

## Active configuration

`bcg/config/defaults.yaml` defines the shipped values. To customize, copy `bcg/config/config.example.yaml` to `<project>/bcg.yaml` and edit `pipeline.confidence.stance_quality`, `source_reliability`, `source_coefficients`, `source_domains`, or `evidence_log_odds_weight`. The file can also be selected with `BCG_CONFIG=/path/to/bcg.yaml` or `--config /path/to/bcg.yaml`; the explicit CLI file has the highest priority. `.env` holds API secrets, not confidence policy. No mapping values are declared in the Unified confidence or URL-policy modules. The older Hybrid confidence fields in the example remain specific to Hybrid.

## Propagation boundary after lowering priors

The default edge threshold remains strictly greater than 0.8. A directly confirmed official Tool Result starts at `0.94 × 0.88 = 0.827` and can activate support; confirmed scholarly sources start at `0.733`, encyclopedias at `0.639`, and forums at `0.395`, and therefore do not activate edges on their own. This is intentional conservative gating, but it changes graph connectivity in downstream confidence propagation. Run the no-key two-hop/weak-source check, then compare edge activation in real trajectories before changing the threshold. A config file's `relation_propagation.input_confidence_threshold` may override the default.

## Tool availability in DRB2

The official benchmark specifies report tasks and scoring, not one fixed agent-facing tool set. The BCG report runner's `ReportPolicy.bind` **replaces** the workflow profile's tool list. For DRB2, the actual tools are guarded `web_search`, `web_fetch` plus orchestration tools; shell, arbitrary Python, downloads, and local files are intentionally unavailable. DRBench, a different benchmark, receives restricted staged-file reading tools. The ordinary stateful ReAct profile has more tools, but it does not describe DRB2's effective tool set.

Before expanding DRB2 tools, measure failures by category. Highest-priority candidate is guarded PDF/table extraction for primary source documents, and a bounded statistics tool if tasks need recomputation. Add these only inside `ReportPolicy.bind`, preserving the benchmark's URL/DOI/title blocks, per-task budgets and trace logging; adding a name to the generic profile would not expose it in DRB2 and could bypass source restrictions. Compare with the old tool set in a separate result directory.

## Commands

```bash
# From this repository root; checks need installed project dependencies but no API key.
python -m compileall -q bcg/core/confidence.py bcg/core/graph.py bcg/construct/unified
python scripts/check_unified_confidence_policy.py
python scripts/check_unified_confidence_policy.py --case-study /path/to/bcg-case-study-3.json /path/to/bcg-case-study-4.json
python scripts/sandbox_confidence_smoke.py

# With project dependencies and a configured Graph model, for the full tests:
uv sync --locked --extra dev --extra graph --extra eval
uv run pytest -q tests/bcg/test_confidence_propagation.py tests/bcg/test_confidence_tempered_evidence.py tests/bcg/test_context_selection.py

# Bench II preparation and a small paired run; use distinct output directories.
uv run python -m benchmarks.public.runner.run_reports prepare drb2
uv run python -m benchmarks.public.runner.run_reports list drb2 --language zh
uv run python -m benchmarks.public.runner.run_reports run drb2 --pipeline stateful-react-agent --language zh --limit 2 --seed 42 --workers 1 --out results/drb2-bcg-brace-smoke --dry-run
# Remove --dry-run only after configuring the Agent and Graph server.
uv run python -m benchmarks.public.runner.run_reports judge --out results/drb2-bcg-brace-smoke --dry-run
```
