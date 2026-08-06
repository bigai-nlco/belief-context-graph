---
title: "Live Stream Viewer"
description: "Replay trajectory growth, inspect graph evidence, and audit final decisions."
icon: "eye"
---

The live viewer is located at:

```text
dashboard/bcg_viewer/belief_graph_stream_viewer.html
```

Its visual language also defines this documentation theme: warm paper background, white panels, brick-red primary actions, teal selection, yellow evidence, and role-specific node colors.

## Start the viewer backend

```bash
cd dashboard/bcg_viewer
python3 serve_viewer.py
```

Open:

```text
http://127.0.0.1:8123/belief_graph_stream_viewer.html
```

The server discovers stream outputs and enables **Run new case** when the configured benchmark and construction commands are available.

## Replay-only mode

Open the HTML file and choose a folder containing:

- complete stream sample directories
- standalone compatible `result.json` files
- or both

The viewer discovers compatible files recursively and avoids duplicate entries.

## Main panels

### Conversation trajectory

The left panel streams messages and highlights evidence spans linked to beliefs.

### Graph canvas

The graph rebuilds after each turn. New nodes flash. Relation styles distinguish:

- `depends_on`
- `supplements`
- `contradicts`
- legacy/supportive relations when present

### Inspector

Click a node or edge to inspect text, source, confidence, entities, evidence, and relation details.

### Run insights

The viewer can summarize graph growth, timing phases, merge behavior, confidence history, and the final decision path when those fields are present.

## Selection workflow

- click normally to inspect
- Ctrl-click on Windows/Linux or Command-click on macOS to select nodes
- choose **View subgraph**
- optionally include direct neighbors
- press Escape to clear inspection

## Timing badges

When artifacts include timing data, the trajectory can show:

- agent generation
- tool execution
- node generation
- merge
- LLM merge check
- edge generation

## Supported data

The viewer normalizes multiple schema generations, including `all_nodes` or `nodes`, legacy relation names, `trajectory_index` or `turn_id`, and graph-level evidence records.
