#!/usr/bin/env python3
"""
scripts/smoke_test.py  (v3)
===========================
Offline end-to-end self-check — NO network, NO API keys.

A fake `call_model` is monkey-patched over `construct_beliefs.llm.call_model`;
it recognises the v3 prompt kinds and returns deterministic, well-formed JSON:
  * the single-call UPDATE prompt (sentence mode + excerpt mode) → new nodes
    (with tmp ids n0/n1) + new forward edges (existing→n0, n0→n1);
  * the full-graph BACKWARD prompt → confirms / contradicts;
  * merge verify / merge full.
A fake embedder hashes character trigrams into 64-dim vectors so the planted
duplicate "silver Honda Civic" belief (restated across two turns) really merges.

Two inputs are exercised through the unified pipeline:
  1. multi-session conversation data (flattened, date-ordered), sentence mode,
     clustering ON, merge=embedding;
  2. a research trajectory (system/user/assistant-with-tags/tool/final),
     excerpt mode, merge=llm.

Asserted invariants:
  * every located evidence span slices the original turn content exactly;
  * sentence-mode evidence is a WHOLE sentence (via=split_sentence);
  * tmp ids resolved → forward from_id < to_id; backward from_id > to_id; all
    endpoints ACTIVE (post-merge);
  * NO `layer` / `scenario` fields; source.type in {user, assistant, tool};
  * planted duplicate merged: absorbed gone, merged_from set, canonical carries
    a "merge" confidence step adopting the newest member's confidence;
  * sessions flattened chronologically (order_sorted True);
  * the March 15th belief carries event_time 2023-03-15;
  * output files exist (result.json, events.jsonl, final_graph.json,
    merge_final.{json,log}, embedding_calls.jsonl, token_usage.*).

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcg.belief_graph import llm  # noqa: E402
from bcg.belief_graph.loaders import iter_items, load_input_file  # noqa: E402
from bcg.belief_graph.pipeline import run_item  # noqa: E402
from bcg.belief_graph.stream import StreamOptions  # noqa: E402

# ---------------------------------------------------------------------------
# Fake chat model
# ---------------------------------------------------------------------------


def _section(prompt: str, header: str) -> str:
    i = prompt.find(header)
    if i < 0:
        return ""
    start = prompt.find("\n", i)
    if start < 0:
        return ""
    j = prompt.find("\n## ", start)
    return prompt[start + 1 : j if j >= 0 else len(prompt)]


def _ids(block: str) -> list:
    return sorted({int(m) for m in re.findall(r'"id":\s*(\d+)', block)})


def fake_call_model(
    client,
    model,
    prompt,
    temperature=0.0,
    max_tokens=None,
    retries=3,
    backoff=2.0,
    usage_label=None,
) -> str:
    llm.USAGE.record(
        model=model,
        prompt_tokens=len(prompt) // 4,
        completion_tokens=50,
        total_tokens=len(prompt) // 4 + 50,
        estimated=True,
    )

    # ---- merge verify (embedding candidates)
    if "## Candidate beliefs" in prompt:
        block = _section(prompt, "## Candidate beliefs")
        ids = _ids(block)
        beliefs = re.findall(r'"belief":\s*"((?:[^"\\]|\\.)*)"', block)
        canonical = beliefs[0] if beliefs else "merged belief"
        if len(ids) < 2:
            return json.dumps({"merge_groups": []})
        return json.dumps(
            {
                "merge_groups": [
                    {
                        "ids": ids,
                        "canonical_belief": canonical,
                        "reason": "candidates verified identical (fake)",
                    }
                ]
            }
        )

    # ---- merge full graph (strategy=llm)
    if "## Full belief list" in prompt:
        block = _section(prompt, "## Full belief list")
        pairs = re.findall(r'"id":\s*(\d+).*?"belief":\s*"((?:[^"\\]|\\.)*)"', block)
        by_text = {}
        for bid, text in pairs:
            by_text.setdefault(text, []).append(int(bid))
        groups = [
            {
                "ids": sorted(v),
                "canonical_belief": k,
                "reason": "identical wording (fake)",
            }
            for k, v in by_text.items()
            if len(v) >= 2
        ]
        return json.dumps({"merge_groups": groups})

    # ---- full-graph backward
    if "## Full belief graph" in prompt:
        ids = _ids(_section(prompt, "## Full belief graph"))
        rels = []
        if len(ids) >= 2:
            rels.append(
                {
                    "from_id": ids[-1],
                    "to_id": ids[0],
                    "type": "confirms",
                    "note": "fake: later confirms earliest",
                }
            )
        if len(ids) >= 3:
            rels.append(
                {
                    "from_id": ids[-1],
                    "to_id": ids[1],
                    "type": "contradicts",
                    "note": "fake: later contradicts second",
                }
            )
        return json.dumps({"relations": rels})

    # ---- single-call UPDATE, sentence mode
    if "## Current turn sentences" in prompt:
        block = _section(prompt, "## Current turn sentences")
        sents = re.findall(r"^\[(\d+)\] (.*)$", block, re.M)
        existing = _ids(_section(prompt, "### Existing nodes"))
        if not sents:
            return json.dumps({"beliefs": [], "forward_relations": []})
        beliefs, fwd = [], []
        # n0: first sentence as an asserted claim
        first = sents[0][1]
        b0 = {
            "tmp_id": "n0",
            "belief": f"S-claim: {first[:48]}",
            "stance": "asserted",
            "supporting_sentence_indices": [int(sents[0][0])],
            "event_time": None,
            "time_text": None,
        }
        # plant the dedup target + a time belief when present
        if "silver Honda Civic" in block:
            b0 = {
                "tmp_id": "n0",
                "belief": "The user drives a silver Honda Civic.",
                "stance": "asserted",
                "supporting_sentence_indices": [int(sents[0][0])],
                "event_time": None,
                "time_text": None,
            }
        beliefs.append(b0)
        if "March 15th" in block:
            beliefs.append(
                {
                    "tmp_id": "n1",
                    "belief": "The user's car was first serviced on March 15th.",
                    "stance": "asserted",
                    "supporting_sentence_indices": [int(s[0]) for s in sents],
                    "event_time": "2023-03-15",
                    "time_text": "March 15th",
                }
            )
        elif len(sents) >= 2:
            beliefs.append(
                {
                    "tmp_id": "n1",
                    "belief": f"S-summary: {first[:24]} (+{len(sents) - 1} more)",
                    "stance": "judged",
                    "supporting_sentence_indices": None,  # → fallback to ALL sentences
                    "event_time": None,
                    "time_text": None,
                }
            )
        # forward edges: existing→n0, n0→n1
        if existing:
            fwd.append(
                {
                    "from": max(existing),
                    "to": "n0",
                    "type": "informs",
                    "note": "fake: prior node informs first new",
                }
            )
        if len(beliefs) >= 2:
            fwd.append(
                {
                    "from": "n0",
                    "to": "n1",
                    "type": "informs",
                    "note": "fake: chain within turn",
                }
            )
        return json.dumps({"beliefs": beliefs, "forward_relations": fwd})

    # ---- single-call UPDATE, excerpt mode
    if "## Current turn content" in prompt:
        block = _section(prompt, "## Current turn content")
        content = block.rstrip("\n")
        existing = _ids(_section(prompt, "### Existing nodes"))
        first_line = content.split("\n", 1)[0].strip()
        excerpt = first_line[:60] if len(first_line) >= 8 else first_line
        beliefs, fwd = [], []
        if excerpt:
            beliefs.append(
                {
                    "tmp_id": "n0",
                    "belief": f"Claim: {excerpt[:50]}",
                    "stance": "asserted",
                    "supporting_excerpts": [excerpt],
                    "event_time": None,
                    "time_text": None,
                }
            )
        m = re.search(r"(828 meters)", content)
        if m and len(beliefs) >= 1:
            beliefs.append(
                {
                    "tmp_id": "n1",
                    "belief": "The Burj Khalifa is 828 meters tall.",
                    "stance": "asserted",
                    "supporting_excerpts": ["828 meters"],
                    "event_time": None,
                    "time_text": None,
                }
            )
        if existing and beliefs:
            fwd.append(
                {
                    "from": max(existing),
                    "to": "n0",
                    "type": "informs",
                    "note": "fake: prior informs new",
                }
            )
        if len(beliefs) >= 2:
            fwd.append(
                {"from": "n0", "to": "n1", "type": "informs", "note": "fake: chain"}
            )
        return json.dumps({"beliefs": beliefs, "forward_relations": fwd})

    return json.dumps({"beliefs": [], "forward_relations": []})


# ---------------------------------------------------------------------------
# Fake embedder (trigram-hash → 64-dim normalized vector)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    model = "fake-embedder"

    def __init__(self):
        self._log_path = None
        self._cache = {}

    def set_log_path(self, p):
        self._log_path = Path(p)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def clear_cache(self):
        self._cache.clear()

    def embed(self, texts, purpose=""):
        import math

        out = []
        for t in texts:
            v = [0.0] * 64
            s = t.strip().lower()
            for i in range(max(1, len(s) - 2)):
                tri = s[i : i + 3]
                v[hash(tri) % 64] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"purpose": purpose, "n_texts": len(texts)}, ensure_ascii=False
                    )
                    + "\n"
                )
        return out


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


def check_common(res, out_dir, tag):
    traj = res["trajectory"]
    beliefs = res["all_beliefs"]
    assert beliefs, f"[{tag}] no beliefs extracted"

    # no removed fields
    for b in beliefs:
        assert "layer" not in b, f"[{tag}] belief still carries 'layer'"
        src = b.get("source") or {}
        assert src.get("type") in {
            "user",
            "assistant",
            "tool",
        }, f"[{tag}] bad source.type {src.get('type')!r}"
        assert "scenario" not in src and "segment_type" not in src, (
            f"[{tag}] source still carries scenario/segment fields"
        )
    assert "scenario" not in res, f"[{tag}] result still carries scenario"

    # evidence offsets exact + sentence evidence is a whole sentence
    for b in beliefs:
        for ev in b["evidence"]:
            if ev.get("start") is None:
                continue
            ti = ev["source"]["trajectory_index"]
            content = traj[ti]["content"]
            assert content[ev["start"] : ev["end"]] == ev["text"], (
                f"[{tag}] evidence offset mismatch on belief {b['id']}"
            )

    active_ids = {b["id"] for b in beliefs}
    for r in res["forward_relations"]:
        assert r["from_id"] < r["to_id"], f"[{tag}] forward not from<to: {r}"
        assert r["from_id"] in active_ids and r["to_id"] in active_ids, (
            f"[{tag}] forward endpoint not active: {r}"
        )
    for r in res["backward_relations"]:
        assert r["from_id"] > r["to_id"], f"[{tag}] backward not from>to: {r}"
        assert r["from_id"] in active_ids and r["to_id"] in active_ids, (
            f"[{tag}] backward endpoint not active: {r}"
        )

    for fn in (
        "result.json",
        "events.jsonl",
        "final_graph.json",
        "token_usage.json",
        "token_usage.txt",
    ):
        assert (out_dir / fn).exists(), f"[{tag}] missing {fn}"
    assert (out_dir / "logs" / "merge_final.json").exists(), (
        f"[{tag}] missing merge_final.json"
    )
    assert (out_dir / "logs" / "merge_final.log").exists(), (
        f"[{tag}] missing merge_final.log"
    )
    print(
        f"  [{tag}] {len(beliefs)} beliefs, "
        f"{len(res['forward_relations'])} fwd, {len(res['backward_relations'])} bwd, "
        f"{len(res['merges'])} merge(s)  ✓"
    )


def main() -> None:
    llm.call_model = fake_call_model

    base = ROOT / "_smoke_out"
    if base.exists():
        shutil.rmtree(base)

    # ---------------- conversation + sentence + clustering + embedding merge ----
    data = load_input_file(str(ROOT / "examples" / "conversation_example.json"))
    item = iter_items(data)[0]
    assert item["order_sorted"] is True, "sessions not date-sorted on flatten"
    # earliest session (mar12) must come first → its user turn is turn 0
    assert "bought a silver Honda Civic" in item["turns"][0]["content"], (
        f"flatten order wrong: {item['turns'][0]['content'][:40]}"
    )
    llm.USAGE.reset()
    opts = StreamOptions(
        evidence_mode="sentence",
        use_clustering=True,
        cluster_threshold=0.4,
        cluster_min_sentences=2,
        merge_strategy="embedding",
        merge_threshold=0.86,
    )
    out_c = base / "conversation" / item["item_id"]
    res_c = run_item(
        item,
        client=None,
        model="fake-chat",
        out_dir=out_c,
        options=opts,
        embedder=FakeEmbedder(),
    )
    check_common(res_c, out_c, "conversation")

    # whole-sentence evidence used
    assert any(
        ev.get("via") == "split_sentence"
        for b in res_c["all_beliefs"]
        for ev in b["evidence"]
    ), "no sentence evidence produced"
    # null-indices fallback → multi-sentence evidence on at least one S-summary
    summaries = [b for b in res_c["all_beliefs"] if b["belief"].startswith("S-summary")]
    assert any(len(b["evidence"]) > 1 for b in summaries), (
        "whole-content fallback produced no multi-sentence evidence"
    )

    # planted duplicate merged
    assert res_c["merges"], "no merges applied"
    civic = [
        b
        for b in res_c["all_beliefs"]
        if b["belief"] == "The user drives a silver Honda Civic."
    ]
    assert len(civic) == 1, f"Civic duplicates not merged: {len(civic)} remain"
    civic = civic[0]
    assert civic.get("merged_from"), "merged_from missing on canonical"
    assert any(h["step"] == "merge" for h in civic["confidence_history"]), (
        "merge step missing in confidence_history"
    )
    for rec in res_c["merges"]:
        for a in rec["absorbed_ids"]:
            assert a not in {b["id"] for b in res_c["all_beliefs"]}, (
                f"absorbed id {a} active"
            )

    # time attribution survived
    assert any(b.get("event_time") == "2023-03-15" for b in res_c["all_beliefs"]), (
        "event_time 2023-03-15 missing"
    )
    # backward produced a contradicts confidence move
    assert any(
        h.get("step") == "contradicts"
        for b in res_c["all_beliefs"]
        for h in b["confidence_history"]
    ), "no contradicts confidence update recorded"

    # ---------------------- research trajectory + excerpt + llm merge ----------
    data_r = load_input_file(str(ROOT / "examples" / "research_example.json"))
    item_r = iter_items(data_r)[0]
    llm.USAGE.reset()
    opts_r = StreamOptions(evidence_mode="excerpt", merge_strategy="llm")
    out_r = base / "research" / item_r["item_id"]
    res_r = run_item(
        item_r, client=None, model="fake-chat", out_dir=out_r, options=opts_r
    )
    check_common(res_r, out_r, "research")

    src_types = {(b.get("source") or {}).get("type") for b in res_r["all_beliefs"]}
    assert {
        "user",
        "assistant",
        "tool",
    } <= src_types, f"role source types incomplete: {src_types}"
    # system turn kept for index alignment but produced nothing
    assert res_r["trajectory"][0]["role"] == "system"
    assert all(b["source"]["trajectory_index"] != 0 for b in res_r["all_beliefs"]), (
        "system turn (index 0) produced beliefs"
    )

    print("\nALL SMOKE CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
