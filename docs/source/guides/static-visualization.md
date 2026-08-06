---
title: "Static Visualization"
description: "Generate a self-contained HTML inspector for one graph result."
icon: "chart-network"
---

Use the built-in renderer:

```bash
bcg construct visualize outputs/case-42/result.json
```

Supported inputs:

- `result.json`
- `final_graph.json`
- `belief_graph.jsonl`

Choose an output path:

```bash
bcg construct visualize \
  outputs/case-42/final_graph.json \
  --output reports/case-42.html
```

The generated file is self-contained and includes:

- conversation trajectory
- evidence highlighting
- graph layout by source turn
- role and node-type colors
- relation paths
- node, edge, factor, confidence, timing, and audit inspection
- summary counts

## Static vs live viewer

| Use static visualization when | Use live viewer when |
|---|---|
| Sharing one finalized result | Replaying graph growth |
| No server should be required | Comparing samples |
| Embedding an artifact in a report | Running a new case |
| Inspecting final state | Investigating timing and per-turn changes |
