#!/usr/bin/env python3
"""
scripts/online_smoke_test.py  (v3)
==================================
Offline self-check for the v3 streaming interface (construct_beliefs.online).
NO network, NO API keys: a deterministic fake chat model (recognising the v3
single-call update + full-graph backward + merge prompts) is monkey-patched over
construct_beliefs.llm.call_model, and a trigram-hash fake embedder stands in for
the embedding endpoint.

It drives the REAL StreamingBeliefBuilder through StreamingTrajectorySession /
SessionManager, one turn at a time, and asserts:

  * per-turn snapshots are forward-only (backward_relations / merges empty);
    the FINAL snapshot (after is_trajectory_end) carries backward + merges;
  * belief_graph.jsonl has one "turn" snapshot per ingested turn + one "final";
    belief_graph_latest.json matches the last snapshot;
  * trajectory_stream.jsonl logs every received dict in order;
  * trajectory.json is run.py-compatible and complete at the end;
  * result.json / events.jsonl / final_graph.json / token_usage.* exist;
    NO scenario field; source.type in {user, assistant, tool};
  * located evidence slices the ORIGINAL turn content exactly; forward edges
    low->high id, backward high->low, all endpoints active post-merge;
  * the duplicated Civic belief (restated across turns) merges to one node;
  * fragment streaming (is_message_end=False) assembles into ONE turn;
  * interleaving two trajectories keeps token accounting fully isolated;
  * the JSONL driver finalizes on is_trajectory_end AND on EOF, skipping junk.

Run:  python scripts/online_smoke_test.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcg import online_driver as drv  # noqa: E402
from bcg.belief_graph import llm  # noqa: E402
from bcg.belief_graph.online import (  # noqa: E402
    SessionManager,
    StreamingTrajectorySession,
    TrajectoryClosedError,
)
from bcg.belief_graph.stream import StreamOptions  # noqa: E402

CALLS = {"n": 0}


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
    CALLS["n"] += 1
    llm.USAGE.record(
        model=model,
        prompt_tokens=len(prompt) // 4,
        completion_tokens=50,
        total_tokens=len(prompt) // 4 + 50,
        estimated=True,
    )

    if "## Full belief list" in prompt:  # merge (llm)
        block = _section(prompt, "## Full belief list")
        pairs = re.findall(r'"id":\s*(\d+).*?"belief":\s*"((?:[^"\\]|\\.)*)"', block)
        by_text = {}
        for bid, text in pairs:
            by_text.setdefault(text, []).append(int(bid))
        groups = [
            {"ids": sorted(v), "canonical_belief": k, "reason": "identical (fake)"}
            for k, v in by_text.items()
            if len(v) >= 2
        ]
        return json.dumps({"merge_groups": groups})

    if "## Candidate beliefs" in prompt:  # merge (embedding verify)
        block = _section(prompt, "## Candidate beliefs")
        ids = _ids(block)
        beliefs = re.findall(r'"belief":\s*"((?:[^"\\]|\\.)*)"', block)
        if len(ids) < 2:
            return json.dumps({"merge_groups": []})
        return json.dumps(
            {
                "merge_groups": [
                    {
                        "ids": ids,
                        "canonical_belief": beliefs[0] if beliefs else "merged",
                        "reason": "verified identical (fake)",
                    }
                ]
            }
        )

    if "## Full belief graph" in prompt:  # backward (full graph)
        ids = _ids(_section(prompt, "## Full belief graph"))
        rels = []
        if len(ids) >= 2:
            rels.append(
                {
                    "from_id": ids[-1],
                    "to_id": ids[0],
                    "type": "confirms",
                    "note": "fake",
                }
            )
        return json.dumps({"relations": rels})

    if "## Current turn content" in prompt:  # update (excerpt mode)
        content = _section(prompt, "## Current turn content").rstrip("\n")
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
        if "silver Honda Civic" in content:
            beliefs.append(
                {
                    "tmp_id": "n1",
                    "belief": "The user drives a silver Honda Civic.",
                    "stance": "asserted",
                    "supporting_excerpts": ["silver Honda Civic"],
                    "event_time": None,
                    "time_text": None,
                }
            )
        if existing and beliefs:
            fwd.append(
                {"from": max(existing), "to": "n0", "type": "informs", "note": "fake"}
            )
        if len(beliefs) >= 2:
            fwd.append({"from": "n0", "to": "n1", "type": "informs", "note": "fake"})
        return json.dumps({"beliefs": beliefs, "forward_relations": fwd})

    if "## Current turn sentences" in prompt:  # update (sentence mode)
        block = _section(prompt, "## Current turn sentences")
        sents = re.findall(r"^\[(\d+)\] (.*)$", block, re.M)
        if not sents:
            return json.dumps({"beliefs": [], "forward_relations": []})
        return json.dumps(
            {
                "beliefs": [
                    {
                        "tmp_id": "n0",
                        "belief": f"S-claim: {sents[0][1][:48]}",
                        "stance": "asserted",
                        "supporting_sentence_indices": [int(sents[0][0])],
                        "event_time": None,
                        "time_text": None,
                    }
                ],
                "forward_relations": [],
            }
        )

    return json.dumps({"beliefs": [], "forward_relations": []})


class FakeEmbedder:
    model = "fake-trigram-64d"

    def __init__(self):
        self._log_path = None
        self._cache = {}

    def set_log_path(self, path):
        self._log_path = Path(path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def clear_cache(self):
        self._cache.clear()

    def embed(self, texts, purpose=""):
        out = []
        for t in texts:
            v = self._cache.get(t)
            if v is None:
                vec = [0.0] * 64
                s = t.lower()
                for k in range(max(1, len(s) - 2)):
                    vec[hash(s[k : k + 3]) % 64] += 1.0
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                v = [x / norm for x in vec]
                self._cache[t] = v
            out.append(v)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"purpose": purpose, "n_texts": len(texts)}) + "\n")
        return out


def make_turns(problem_id: str):
    return [
        {
            "problem_id": problem_id,
            "role": "system",
            "content": "You are a helpful research agent.",
        },
        {
            "problem_id": problem_id,
            "role": "user",
            "content": "I need car help. I drive a silver Honda Civic and want service info.",
        },
        {
            "problem_id": problem_id,
            "role": "assistant",
            "content": (
                "<think>The user has a silver Honda Civic; I should look up its "
                "service interval.</think>"
                "<tool_call>search(query='Honda Civic service interval')</tool_call>"
            ),
        },
        {
            "problem_id": problem_id,
            "role": "tool",
            "content": (
                "<tool_response>The Honda Civic recommended maintenance is every "
                "12,000 km. GPS navigation is available on the EX trim.</tool_response>"
            ),
        },
        {
            "problem_id": problem_id,
            "role": "assistant",
            "content": (
                "Your silver Honda Civic should be serviced every 12,000 km, and the "
                "EX trim includes GPS navigation."
            ),
            "is_trajectory_end": True,
        },
    ]


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_offsets_and_edges(result: dict, label: str):
    beliefs = result["all_beliefs"]
    active = {b["id"] for b in beliefs}
    assert beliefs, f"[{label}] no beliefs"
    assert "scenario" not in result, f"[{label}] result still carries scenario"
    n_exact = 0
    for b in beliefs:
        assert "layer" not in b, f"[{label}] belief still has layer"
        assert (b.get("source") or {}).get("type") in {
            "user",
            "assistant",
            "tool",
        }, f"[{label}] bad source.type"
        assert b.get("evidence"), f"[{label}] belief {b['id']} has no evidence"
        for ev in b["evidence"]:
            if ev["match"] == "exact" and ev["start"] is not None:
                ti = ev["source"]["trajectory_index"]
                content = result["trajectory"][ti]["content"]
                assert content[ev["start"] : ev["end"]] == ev["text"], (
                    f"[{label}] offset mismatch on belief {b['id']}"
                )
                n_exact += 1
    assert n_exact > 0, f"[{label}] no exact evidence located"
    for r in result["forward_relations"]:
        assert r["from_id"] < r["to_id"], f"[{label}] bad forward dir {r}"
        assert r["from_id"] in active and r["to_id"] in active, (
            f"[{label}] dead forward {r}"
        )
    for r in result["backward_relations"]:
        assert r["from_id"] > r["to_id"], f"[{label}] bad backward dir {r}"
        assert r["from_id"] in active and r["to_id"] in active, (
            f"[{label}] dead backward {r}"
        )


def _opts():
    return StreamOptions(
        evidence_mode="excerpt",
        use_clustering=False,
        merge_strategy="embedding",
        merge_threshold=0.86,
    )


# ---------------------------------------------------------------------------
# Test 1 — single trajectory, full lifecycle
# ---------------------------------------------------------------------------


def test_single(base: Path):
    pid = "prob_single"
    mgr = SessionManager(
        client=None,
        model="fake-chat",
        embedder=FakeEmbedder(),
        output_root=base,
        options=_opts(),
    )
    turns = make_turns(pid)
    snaps = [mgr.push(t) for t in turns]

    for s in snaps[:-1]:
        assert s["stage"] == "turn", f"expected stage=turn, got {s['stage']}"
        assert s["backward_relations"] == [], "per-turn snapshot has backward rels"
        assert s["merges"] == [], "per-turn snapshot has merges"
    final = snaps[-1]
    assert final["stage"] == "final" and final["finalized"] is True
    assert final["backward_relations"], "final graph missing backward relations"
    assert final["merges"], "final graph missing merges (duplicate Civic should merge)"

    out = base / pid
    for fn in (
        "result.json",
        "events.jsonl",
        "final_graph.json",
        "token_usage.json",
        "token_usage.txt",
        "trajectory_stream.jsonl",
        "trajectory.json",
        "belief_graph.jsonl",
        "belief_graph_latest.json",
    ):
        assert (out / fn).exists(), f"[single] missing {fn}"

    lines = read_jsonl(out / "belief_graph.jsonl")
    n_turn = sum(1 for x in lines if x["stage"] == "turn")
    n_final = sum(1 for x in lines if x["stage"] == "final")
    assert n_turn == len(turns), (
        f"[single] {n_turn} turn snapshots, expected {len(turns)}"
    )
    assert n_final == 1, f"[single] expected 1 final snapshot, got {n_final}"
    latest = json.loads((out / "belief_graph_latest.json").read_text(encoding="utf-8"))
    assert latest == lines[-1], "[single] latest.json != last jsonl line"

    raw = read_jsonl(out / "trajectory_stream.jsonl")
    assert len(raw) == len(turns), "[single] stream log line count mismatch"
    assert [r["recv_index"] for r in raw] == list(range(len(turns)))
    assert all(r["ingested"] for r in raw), "[single] some turns not marked ingested"

    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    assert traj["complete"] is True and len(traj["trajectory"]) == len(turns)
    assert [m["role"] for m in traj["trajectory"]] == [t["role"] for t in turns]
    assert all(set(m.keys()) == {"role", "content"} for m in traj["trajectory"])

    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["item_id"] == pid
    assert {b["id"] for b in result["all_beliefs"]} == {
        b["id"] for b in final["beliefs"]
    }
    check_offsets_and_edges(result, "single")

    civic = [
        b
        for b in result["all_beliefs"]
        if b["belief"] == "The user drives a silver Honda Civic."
    ]
    assert len(civic) == 1, f"[single] Civic not merged: {len(civic)} copies"
    assert civic[0].get("merged_from"), "[single] canonical Civic lost merged_from"

    assert result["trajectory"][0]["role"] == "system"
    assert all(b["source"]["trajectory_index"] != 0 for b in result["all_beliefs"])

    try:
        mgr.push({"problem_id": pid, "role": "user", "content": "late"})
        raise AssertionError("[single] closed trajectory accepted a turn")
    except TrajectoryClosedError:
        pass
    print(
        f"  [single] {len(result['all_beliefs'])} beliefs, "
        f"{len(result['backward_relations'])} bwd, {len(result['merges'])} merge(s), "
        f"{n_turn} per-turn + {n_final} final snapshot  ✓"
    )


# ---------------------------------------------------------------------------
# Test 2 — fragment streaming assembles into ONE turn
# ---------------------------------------------------------------------------


def test_fragments(base: Path):
    pid = "prob_frag"
    sess = StreamingTrajectorySession(
        pid,
        client=None,
        model="fake-chat",
        embedder=FakeEmbedder(),
        output_root=base,
        options=_opts(),
    )
    sess.push(
        {
            "problem_id": pid,
            "role": "user",
            "content": "I drive a ",
            "is_message_end": False,
        }
    )
    s_mid = sess.push(
        {
            "problem_id": pid,
            "role": "user",
            "content": "silver Honda ",
            "is_message_end": False,
        }
    )
    assert s_mid["stage"] == "buffered", "[frag] mid-fragment should be buffered"
    sess.push(
        {"problem_id": pid, "role": "user", "content": "Civic.", "is_message_end": True}
    )
    sess.push(
        {
            "problem_id": pid,
            "role": "assistant",
            "content": "Noted: silver Honda Civic.",
            "is_trajectory_end": True,
        }
    )

    out = base / pid
    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    assert len(traj["trajectory"]) == 2, (
        f"[frag] expected 2 turns, got {len(traj['trajectory'])}"
    )
    assert traj["trajectory"][0]["content"] == "I drive a silver Honda Civic.", (
        f"[frag] fragments not concatenated: {traj['trajectory'][0]['content']!r}"
    )

    raw = read_jsonl(out / "trajectory_stream.jsonl")
    assert len(raw) == 4, "[frag] all 4 raw fragments must be logged"
    assert [r["ingested"] for r in raw] == [False, False, True, True]

    lines = read_jsonl(out / "belief_graph.jsonl")
    assert sum(1 for x in lines if x["stage"] == "turn") == 2, (
        "[frag] wrong turn-snapshot count"
    )
    assert sum(1 for x in lines if x["stage"] == "final") == 1
    print("  [fragments] 4 fragments -> 2 turns, content reassembled  ✓")


# ---------------------------------------------------------------------------
# Test 3 — interleaved trajectories keep token accounting isolated
# ---------------------------------------------------------------------------


def test_interleaved_isolation(base: Path):
    mgr = SessionManager(
        client=None,
        model="fake-chat",
        embedder=FakeEmbedder(),
        output_root=base,
        options=_opts(),
    )
    a, b = "prob_A", "prob_B"
    ta, tb = make_turns(a), make_turns(b)

    CALLS["n"] = 0
    for x, y in zip(ta, tb, strict=False):
        mgr.push(x)
        mgr.push(y)

    res_a = json.loads((base / a / "result.json").read_text(encoding="utf-8"))
    res_b = json.loads((base / b / "result.json").read_text(encoding="utf-8"))
    na = res_a["token_usage"]["totals"]["n_calls"]
    nb = res_b["token_usage"]["totals"]["n_calls"]
    assert na > 0 and nb > 0, "[iso] a trajectory recorded zero calls"
    assert na + nb == CALLS["n"], (
        f"[iso] call accounting leaked: A={na} + B={nb} != total {CALLS['n']}"
    )
    assert len(llm.USAGE.records) == 0, (
        "[iso] global USAGE not restored to empty after pushes"
    )

    for res, lbl in ((res_a, "A"), (res_b, "B")):
        check_offsets_and_edges(res, f"iso-{lbl}")
        assert res["merges"], f"[iso-{lbl}] expected a merge"
    print(f"  [interleaved] A={na} + B={nb} == {CALLS['n']} calls, ledgers isolated  ✓")


# ---------------------------------------------------------------------------
# Test 4 — the JSONL driver end to end
# ---------------------------------------------------------------------------


def _new_mgr(base: Path) -> SessionManager:
    return SessionManager(
        client=None,
        model="fake-chat",
        embedder=FakeEmbedder(),
        output_root=base,
        options=_opts(),
    )


def test_driver(base: Path):
    pid_a = "drv_end"
    turns_a = make_turns(pid_a)
    f_a = base / "streamA.jsonl"
    f_a.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in turns_a) + "\n",
        encoding="utf-8",
    )
    with open(f_a, encoding="utf-8") as fh:
        summary_a = drv.drive(_new_mgr(base), drv.iter_jsonl(fh), quiet=True)
    assert summary_a["n_turns_pushed"] == len(turns_a), summary_a
    assert summary_a["finalized"] == [pid_a], summary_a
    out_a = base / pid_a
    for fn in (
        "trajectory.json",
        "result.json",
        "belief_graph.jsonl",
        "trajectory_stream.jsonl",
    ):
        assert (out_a / fn).exists(), f"[driver-A] missing {fn}"
    traj_a = json.loads((out_a / "trajectory.json").read_text(encoding="utf-8"))
    assert traj_a["complete"] is True and len(traj_a["trajectory"]) == len(turns_a)

    pid_b = "drv_eof"
    turns_b = [dict(t) for t in make_turns(pid_b)]
    turns_b[-1].pop("is_trajectory_end", None)
    summary_b = drv.drive(_new_mgr(base), iter(turns_b), quiet=True)
    assert summary_b["finalized"] == [pid_b], summary_b
    out_b = base / pid_b
    assert (out_b / "result.json").exists(), (
        "[driver-B] EOF finalize wrote no result.json"
    )
    assert (
        json.loads((out_b / "trajectory.json").read_text(encoding="utf-8"))["complete"]
        is True
    )

    lines = [
        "",
        "   ",
        "not json at all",
        "[1, 2, 3]",
        json.dumps(
            {
                "problem_id": "drv_skip",
                "role": "user",
                "content": "one good line",
                "is_trajectory_end": True,
            }
        ),
    ]
    summary_c = drv.drive(_new_mgr(base), drv.iter_jsonl(iter(lines)), quiet=True)
    assert summary_c["n_turns_pushed"] == 1, (
        f"[driver-C] bad lines not skipped: {summary_c}"
    )
    assert summary_c["finalized"] == ["drv_skip"], summary_c
    print("  [driver] is_trajectory_end + EOF-finalize + malformed-skip paths  ✓")


def main():
    llm.call_model = fake_call_model
    base = ROOT / "_online_smoke_out"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    test_single(base)
    test_fragments(base)
    test_interleaved_isolation(base)
    test_driver(base)
    print("\nALL ONLINE SMOKE CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
