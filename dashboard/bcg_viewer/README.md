# README Demo stream viewer

This directory (`belief_graph_stream_viewer.html`, `serve_viewer.py`,
`build_stream_manifest.py`) contains the interface shown in the README live
demo. It replays persisted trajectories and complete Graph snapshots without
rerunning an Agent or Graph Construction.

Run it from the repository root:

```bash
python3 dashboard/bcg_viewer/serve_viewer.py \
  --stream-dir outputs_stream \
  --host 127.0.0.1 \
  --port 8123
```

Then open <http://127.0.0.1:8123/demo>, select a trajectory, and press
`Start` to replay it through its final Graph.
