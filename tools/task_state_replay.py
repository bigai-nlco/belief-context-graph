"""Replay explicit task annotations over an old snapshot, without model calls.

This is a projection/provenance test, NOT a test of automatic model labelling
quality or an Agent benchmark. The output manifest records that distinction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bcg.construct.unified.task_state import classify_legacy_snapshot, render_task_state


def replay(state: Path, labels: Path, output: Path, budget: int = 1800) -> dict:
    raw = state.read_bytes()
    source = json.loads(raw)
    snapshot = source.get("snapshot", source)
    raw_labels = labels.read_bytes()
    annotations = {int(k): v for k, v in json.loads(raw_labels).items()}
    ids = {n["id"] for n in snapshot["nodes"]}
    if not set(annotations) <= ids:
        raise ValueError("Labels must reference nodes present in this snapshot")
    derived = classify_legacy_snapshot(snapshot, annotations)
    text, node_ids, relation_ids, task_ids = render_task_state(derived, budget)
    summary = {
        "kind": "offline_explicit_label_replay",
        "automatic_labelling_evaluated": False,
        "model_calls": 0,
        "source": str(state.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "labels_sha256": hashlib.sha256(raw_labels).hexdigest(),
        "source_nodes": len(snapshot["nodes"]),
        "source_relations": len(snapshot.get("relations", [])),
        "task_nodes": len(derived["task_state"]["nodes"]),
        "status_counts": derived["task_state"]["status_counts"],
        "rendered_task_ids": task_ids,
        "rendered_belief_ids": node_ids,
        "rendered_relation_ids": relation_ids,
        "rendered_chars": len(text),
        "budget": budget,
    }
    # A replay never modifies original runs or silently replaces a prior replay.
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("graph.json", derived), ("summary.json", summary), ("labels.json", annotations)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "task-context.md").write_text(text + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Agent state.json or native graph snapshot")
    parser.add_argument("--labels", type=Path, required=True, help="Explicitly reviewed annotations keyed by belief ID")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--budget", type=int, default=1800)
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be non-negative")
    print(json.dumps(replay(args.state, args.labels, args.out, args.budget), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
