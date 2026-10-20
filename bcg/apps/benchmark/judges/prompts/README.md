# Official judge prompts

These templates are copied verbatim from the official evaluators and must not
be edited by hand:

| Template | Upstream file |
|---|---|
| `drb2_rubric.txt` | DeepResearch Bench II, `run_evaluation.py` (`PROMPT_TEMPLATE`) |
| `drbench_insight_extraction.txt` | DRBench, `drbench/agents/utils.py` (`break_report_to_insights`) |
| `drbench_insight_scoring.txt` | DRBench, `drbench/prompts/eval_metrics/insight_scoring.txt` |
| `drbench_report_quality.txt` | DRBench, `drbench/metrics/report_quality.py` (first prompt) |
| `drbench_report_quality_retry.txt` | DRBench, `drbench/metrics/report_quality.py` (retry prompt) |

DRBench prompts are Python f-strings upstream; they are stored as equivalent
`str.format` templates (literal braces doubled; `{report_text}`,
`{dr_question}`, `{persona_json}` = `json.dumps(persona, indent=2)`), extracted
with `ast` so that formatting reproduces the f-string byte for byte.

`SOURCES.json` records the upstream commits and the SHA-256 of every template;
the test suite checks it. To update after an upstream change:

```bash
python scripts/sync_judge_prompts.py --drb2 PATH/DeepResearch-Bench-II --drbench PATH/drbench
BCG_DRB2_REPO=PATH/DeepResearch-Bench-II BCG_DRBENCH_REPO=PATH/drbench \
  uv run pytest tests/test_benchmark_judges.py
```

Both upstream projects are licensed under the Apache License 2.0:
<https://github.com/imlrz/DeepResearch-Bench-II>,
<https://github.com/ServiceNow/drbench>.
