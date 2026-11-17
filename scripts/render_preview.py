#!/usr/bin/env python3
"""Offline preview of confidence verbalization on real graph snapshots.

Reads the last snapshot of every trial (agent/bcg/graph_snapshots.jsonl, or
state.json when there is none) and compares, over findings only:

  split_confidence   (high / medium / low confidence)
  split_wep_label    (one WEP word from the same score)
  split_wep          (WEP word · source · why the score moved)

No agent run, no model call. Use it to check, before an experiment, that the
labels do not collapse onto one word, how often the "why" part appears, and
what each style costs per line.

    python scripts/render_preview.py results/test/version1117_actions
    python scripts/render_preview.py RUN_DIR --budget 6000 --samples 3
    BCG_WEP_THRESHOLDS=0.80,0.70,0.60,0.40,0.30 python scripts/render_preview.py RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflows.stateful_react_agent.bcg.rendering import (
    WEP_LABELS,
    change_tags,
    classify_node,
    confidence_label,
    finding_head,
    finding_line,
    source_tag,
    wep_label,
    wep_suffix,
    wep_thresholds,
)

TIERS = ("high confidence", "medium confidence", "low confidence")


def last_snapshot(trial: Path) -> tuple[dict[str, Any] | None, str]:
    path = trial / "agent" / "bcg" / "graph_snapshots.jsonl"
    if path.exists():
        last = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:  # files are large; keep only the newest line
                if line.strip():
                    last = line
        if last:
            try:
                return json.loads(last).get("snapshot"), "graph_snapshots.jsonl"
            except ValueError:
                pass
    state = trial / "agent" / "bcg" / "state.json"
    if state.exists():
        try:
            return json.loads(state.read_text(encoding="utf-8")).get("snapshot"), "state.json"
        except ValueError:
            pass
    return None, "-"


def snapshot_nodes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    raw = snapshot.get("nodes")
    if not isinstance(raw, list):
        raw = (snapshot.get("beliefs") or []) + (snapshot.get("decisions") or [])
    seen: dict[int, dict[str, Any]] = {}
    for node in raw:
        if isinstance(node, dict) and type(node.get("id")) is int:
            seen[node["id"]] = node
    return list(seen.values())


def find_trials(run_dir: Path) -> list[Path]:
    markers = ("agent/bcg/graph_snapshots.jsonl", "agent/bcg/state.json")
    found = {p for p in run_dir.rglob("trials/*") if p.is_dir() and any((p / m).exists() for m in markers)}
    return sorted(found)


def entropy(counts: Counter, n_labels: int) -> float:
    total = sum(counts.values())
    if not total or n_labels < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c)
    return h / math.log(n_labels)


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--budget", type=int, default=int(os.environ.get("BCG_GRAPH_CHARS", "6000") or 6000),
                        help="graph char budget used for the nodes-per-budget estimate (default BCG_GRAPH_CHARS or 6000)")
    parser.add_argument("--samples", type=int, default=2, help="example lines per WEP label")
    args = parser.parse_args()

    tiers: Counter = Counter()
    weps: Counter = Counter()
    tier_to_wep: dict[str, Counter] = defaultdict(Counter)
    combo: dict[tuple[str, str], Counter] = defaultdict(Counter)
    tag_kinds: Counter = Counter()
    with_tags = 0
    lengths: dict[str, list[int]] = defaultdict(list)
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n_trials = n_findings = n_actions = 0

    for run_dir in args.run_dirs:
        for trial in find_trials(run_dir):
            snapshot, origin = last_snapshot(trial)
            if not snapshot:
                print(f"skip {trial}: no snapshot", file=sys.stderr)
                continue
            n_trials += 1
            for node in snapshot_nodes(snapshot):
                if not isinstance(node.get("belief") or node.get("decision"), str):
                    continue
                if classify_node(node)["is_action"]:
                    n_actions += 1
                    continue
                n_findings += 1
                tier, wep = confidence_label(node) or "none", wep_label(node) or "none"
                tiers[tier] += 1
                weps[wep] += 1
                tier_to_wep[tier][wep] += 1
                combo[(str(node.get("stance") or "?"), source_tag(node) or "?")][wep] += 1
                tags = change_tags(node)
                if tags:
                    with_tags += 1
                for tag in tags:
                    tag_kinds[tag.split(" [")[0].split(" by ")[0].split(" ")[0]] += 1
                lines = {
                    "split_confidence": finding_line(node, "split_confidence"),
                    "split_wep_label": finding_head(node) + wep_suffix(node, annotate=False),
                    "split_wep": finding_head(node) + wep_suffix(node),
                }
                for style, line in lines.items():
                    lengths[style].append(len(line) + 1)
                if len(samples[wep]) < args.samples:
                    samples[wep].append({"trial": trial.name, "origin": origin, **lines})

    if not n_findings:
        sys.exit("no findings found under the given run directories")

    out: list[str] = [
        f"trials {n_trials}   findings {n_findings}   action nodes skipped {n_actions}",
        f"WEP thresholds {wep_thresholds()}  (confirmed, almost certain, probable, possible, unlikely; below = doubtful)",
        "",
        "1. LABEL DISTRIBUTION (findings)",
    ]
    for label in TIERS:
        out.append(f"   {label:<20} {tiers[label]:>6}  {pct(tiers[label], n_findings)}")
    out.append(f"   normalised entropy  {entropy(tiers, len(TIERS)):.2f}")
    out.append("")
    for label in WEP_LABELS:
        out.append(f"   {label:<20} {weps[label]:>6}  {pct(weps[label], n_findings)}")
    out.append(f"   normalised entropy  {entropy(weps, len(WEP_LABELS)):.2f}"
               "   (1.0 = spread evenly, near 0 = collapsed onto one word)")

    out += ["", "2. HOW EACH 3-TIER LABEL SPLITS INTO WEP"]
    for tier in TIERS:
        row = tier_to_wep.get(tier) or Counter()
        if row:
            out.append(f"   {tier:<18} -> " + ", ".join(f"{w} {c}" for w, c in row.most_common()))

    out += ["", "3. STANCE x SOURCE -> WEP (most frequent combinations)"]
    for (stance, src), row in sorted(combo.items(), key=lambda kv: -sum(kv[1].values()))[:15]:
        out.append(f"   {stance:<15} {src:<26} {sum(row.values()):>5}  -> "
                   + ", ".join(f"{w} {c}" for w, c in row.most_common()))

    out += ["", "4. 'WHY' ANNOTATIONS (score moved after the prior)",
            f"   findings with any     {with_tags:>6}  {pct(with_tags, n_findings)}"]
    for kind, count in tag_kinds.most_common():
        out.append(f"   {kind:<21} {count:>6}  {pct(count, n_findings)}")
    if with_tags == 0:
        out.append("   none: evidence and relation updates never moved a finding in these graphs")

    out += ["", f"5. LINE COST (budget {args.budget} chars)"]
    for style in ("split_confidence", "split_wep_label", "split_wep"):
        mean = sum(lengths[style]) / len(lengths[style])
        out.append(f"   {style:<18} mean {mean:6.1f} chars/line   ~{int(args.budget // mean):>3} findings per budget")

    out += ["", "6. SAMPLES"]
    for label in WEP_LABELS:
        for sample in samples.get(label, []):
            out.append(f"   [{label}] trial {sample['trial']}")
            for style in ("split_confidence", "split_wep"):
                line = sample[style]
                out.append(f"     {style:<17} {line if len(line) <= 220 else line[:200] + ' … ' + line[-60:]}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
