#!/usr/bin/env python3
"""Scan stream outputs and emit a manifest.json the streaming viewer can load.

For every sample directory that has both a `trajectory_stream.jsonl` and a
`belief_graph.jsonl`, we record the claim/question (used as the dropdown label),
the number of streamed turns, and the size of the final belief graph.

Run from anywhere; pass --root pointing at the construct output directory:

    python3 build_stream_manifest.py

The manifest is written to <root>/manifest.json so a viewer served from the
stream folder can fetch it with a relative path.
"""

import argparse
import datetime as _dt
import json
import os
import re

CLAIM_RE = re.compile(r"^\s*Claim:\s*(.+?)\s*$", re.MULTILINE)
CLAIM_ID_RE = re.compile(r"^\s*Claim ID:\s*(.+?)\s*$", re.MULTILINE)


def _first_lines(path, n=200):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                out.append(line)
    return out


def _read_turns(path):
    turns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = obj.get("turn") or {}
            turns.append(turn)
    return turns


def _last_graph(path):
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last or {}


def _claim_of(turns):
    for t in turns:
        if t.get("role") == "user":
            content = t.get("content") or ""
            m = CLAIM_RE.search(content)
            cid = CLAIM_ID_RE.search(content)
            claim = m.group(1).strip() if m else content.strip().splitlines()[0][:160]
            return claim, (cid.group(1).strip() if cid else "")
    return "", ""


def build(root):
    samples = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        traj = os.path.join(d, "trajectory_stream.jsonl")
        graph = os.path.join(d, "belief_graph.jsonl")
        if not (os.path.exists(traj) and os.path.exists(graph)):
            continue
        turns = _read_turns(traj)
        claim, claim_id = _claim_of(turns)
        last = _last_graph(graph)
        n_snap = sum(1 for _ in _first_lines(graph, n=100000))
        samples.append(
            {
                "id": name,
                "claim_id": claim_id,
                "claim": claim,
                "n_turns": len(turns),
                "n_snapshots": n_snap,
                "n_nodes": int(last.get("n_nodes", 0) or 0),
                "n_relations": len(last.get("relations", []) or []),
            }
        )
    # non-empty graphs first, then by claim id
    samples.sort(key=lambda s: (0 if s["n_nodes"] else 1, s["claim_id"], s["id"]))
    return {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "count": len(samples),
        "samples": samples,
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    current_root = os.path.join(repo_root, "outputs")
    default_root = current_root if os.path.isdir(current_root) else current_root
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=default_root,
        help="directory that holds the per-sample stream folders "
        "(default: <repo>/outputs)",
    )
    ap.add_argument(
        "--out", default=None, help="output path (default: <root>/manifest.json)"
    )
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    manifest = build(root)
    out = args.out or os.path.join(root, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(
        f"wrote {out}: {manifest['count']} samples "
        f"({sum(1 for s in manifest['samples'] if s['n_nodes'])} with a non-empty graph)"
    )


if __name__ == "__main__":
    main()
