"""Replay actual Assistant extraction requests from Studio, never rerun the Agent.

python -m tools.replay_task_workflow prepare|audit|run --out results/<directory>
Prepare in a fresh directory. Audit is offline. Run sends the frozen prompts
to the original paid graph provider and should only be invoked with authorization.
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from bcg.construct.unified import extract, llm
from bcg.construct.unified.prompts import build_node_extraction_prompt, format_sentences_for_prompt
from bcg.construct.unified.task_state import (
    build_task_state,
    classify_node,
    render_task_state,
    task_history,
)

CASES = ((35, 6), (35, 14), (36, 2), (36, 4))


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _checked_request(case: dict[str, Any], pair: dict[str, Any]) -> None:
    """Never pay to replay a different template or settings than the prepared pair."""
    block = format_sentences_for_prompt(case["sentences"])
    current = build_node_extraction_prompt("assistant", mode="sentences", sentences_block=block,
                                           graph_nodes=case["history"], task_state_enabled=True)
    candidate = pair["candidate_request"]
    if (current != candidate["prompt"] or case["name"] != pair["case_id"]
            or not all(case["settings"][k] == candidate[k] for k in ("model", "temperature", "max_tokens", "reasoning_effort"))):
        raise ValueError("Frozen request differs from the current code or settings; prepare a new directory.")


def _before_merge(raw: dict[str, Any], sentences: list[str]) -> list[dict[str, Any]]:
    """Compare graph-model outputs at the same stage, not with survivors of later merges."""
    nodes = []
    for group, key in (("beliefs", "belief"), ("decisions", "decision")):
        for row in raw.get(group, []):
            indices = [i for i in row.get("supporting_sentence_indices", [])
                       if type(i) is int and 0 <= i < len(sentences)]
            if not isinstance(row.get(key), str) or not indices:
                continue
            nodes.append({key: row[key], "task_role": "decision" if key == "decision" else row.get("task_role", "other"),
                          "task_ids": row.get("task_ids", []), "supporting_excerpts": [sentences[i] for i in indices]})
    return nodes


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    with (out / "pairs.jsonl").open("x") as pairs:
        for run, turn in CASES:
            run_dir = Path("results/studio") / f"RUN-{run:04}"
            graph, = (run_dir / "graphs").iterdir()
            name = f"{run:04}-t{turn}"
            trajectory = read(graph / "trajectory.json")["trajectory"]
            root = next(t["content"] for t in trajectory if t["role"] == "user")
            events = [e for e in lines(graph / "events.jsonl") if e.get("role") == "assistant" and e.get("raw_output")]
            calls = [x for x in lines(graph / "logs/prompts.jsonl") if
                     x["prompt"].startswith("# Task\nExtract coherent BELIEFS") and "## Current turn sentences\n" in x["prompt"]]
            assert len(events) == len(calls), "do not guess the call-to-event mapping"
            event, call = next((e, c) for e, c in zip(events, calls, strict=True) if e["turn_index"] == turn)
            original = call["prompt"]
            history = original.split("### Earlier nodes\n", 1)[1].splitlines()[0]
            block = original.rsplit("## Current turn sentences\n", 1)[1].removesuffix("\n")
            chunks = re.split(r"^\[(\d+)\] ", block, flags=re.MULTILINE)
            assert not chunks[0] and [int(i) for i in chunks[1::2]] == list(range(len(chunks[1::2])))
            sentences = [s.removesuffix("\n") for s in chunks[2::2]]
            template = dict(role="assistant", mode="sentences", sentences_block=block, task_state_enabled=True)
            assert build_node_extraction_prompt(graph_nodes=history, **template) == original, "baseline input drift"
            root_row, = json.loads(task_history([], [], 12000, original_task=root))
            with_root = json.dumps([root_row, *json.loads(history)], ensure_ascii=False)
            assert len(with_root) <= 12000, "replay must not evict any of the frozen earlier beliefs"
            candidate = build_node_extraction_prompt(graph_nodes=with_root, **template)
            final = read(graph / "belief_graph_latest.json")
            key = f"t{turn}.extract:assistant"
            usage = final["token_usage"]["by_label"][key]
            base = {k: call[k] for k in ("model", "max_tokens", "temperature", "reasoning_effort")}
            base["source_fingerprint"] = hashlib.sha256((history + block + root).encode()).hexdigest()
            pairs.write(json.dumps({"case_id": name, "stage": "assistant_task_state",
                "baseline_request": {**base, "prompt": original}, "candidate_request": {**base, "prompt": candidate}},
                ensure_ascii=False) + "\n")
            raw = event["raw_output"]
            original_extraction = _before_merge(json.loads(raw) if isinstance(raw, str) else raw, sentences)
            write(out / f"{name}.input.json", {"name": name, "root": root, "sentences": sentences,
                  "history": with_root, "settings": {k: base[k] for k in base if k != "source_fingerprint"},
                  "original_extraction": original_extraction,
                  "original_nodes": [n for n in final["nodes"] if n["id"] in event["new_node_ids"]
                                     and n.get("extraction_method") != "rule_tool_call"],
                  "baseline_usage": usage})


def run(out: Path) -> None:
    from dotenv import dotenv_values
    key = dotenv_values(Path.home() / ".bcg/.env").get("LITELLM_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("missing LiteLLM graph key")
    client = llm.make_client({"base_url": "https://litellm.mybigai.ac.cn/", "api_key": key})
    client = client.with_options(max_retries=0, timeout=120)
    for pair in lines(out / "pairs.jsonl"):
        name = pair["case_id"]
        if (out / f"{name}.result.json").exists():
            continue
        case = read(out / f"{name}.input.json")
        _checked_request(case, pair)
        settings = dict(case["settings"])
        model = settings.pop("model")
        assert model == "gpt-6-luna"
        llm.USAGE.reset()
        llm.USAGE.set_label(name)
        started = time.monotonic()
        value = extract.extract_nodes(client, model, role="assistant", sentences=case["sentences"],
            graph_nodes_str=case["history"], task_state_enabled=True, **settings)
        elapsed = time.monotonic() - started
        previous = [{"id": n["id"], "belief": n["content"], "task_role": n.get("task_role")}
                    for n in json.loads(case["history"]) if "id" in n]
        nodes = []
        for n in value["nodes"]:
            indices = n.get("supporting_sentence_indices") or []
            node = {k: n[k] for k in ("belief", "decision", "node_type") if k in n}
            node.update(role="assistant", supporting_excerpts=[case["sentences"][i] for i in indices])
            classify_node(node, n if indices else {}, previous, original_task=case["root"])
            nodes.append(node)
        result = {"stage": "assistant_task_state", "baseline_usage": case["baseline_usage"],
                  "candidate_usage": llm.USAGE.summary()["by_label"][name], "nodes": nodes,
                  "raw": value["raw_output"], "elapsed_seconds": round(elapsed, 3)}
        def for_review(ns: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{k: n[k] for k in ("belief", "decision", "task_role", "task_ids", "task_scope",
                "gap_quote", "next_action_quote", "supporting_excerpts") if k in n} for n in ns]
        choices = [("baseline", for_review(case["original_extraction"])), ("candidate", for_review(nodes))]
        random.Random("workflow-v3:" + name).shuffle(choices)
        write(out / f"{name}.blind.json", {"source": case["sentences"], "A": choices[0][1], "B": choices[1][1]})
        result["blind_key"] = [x[0] for x in choices]
        write(out / f"{name}.result.json", result)
        print(name, result["candidate_usage"], flush=True)


def audit(out: Path) -> None:
    """Replay old model annotations, plus a separately labelled positive fixture; never use the network."""
    report = {"kind": "offline_recorded_outputs_and_explicit_source_fixtures", "provider_calls": 0,
              "archived_task_decisions": [], "render_only": [], "explicit_fixtures": []}
    for run, turn in CASES:
        case = read(out / f"{run:04}-t{turn}.input.json")
        previous = [{"id": n["id"], "belief": n["content"], "task_role": n.get("task_role")}
                    for n in json.loads(case["history"]) if "id" in n]
        for n in case["original_nodes"]:
            if n.get("task_role") != "task":
                continue
            node = {k: n[k] for k in ("id", "role", "belief", "supporting_excerpts")}
            classify_node(node, n, previous, original_task=case["root"])
            report["archived_task_decisions"].append({"run": run, "turn": turn, "belief_id": node["id"],
                "before": "task", "after": node["task_role"], "original_belief_preserved": node["belief"] == n["belief"]})
    for run in (35, 36):
        base = Path("results/studio") / f"RUN-{run:04}" / "drb2/trials/2/agent/bcg"
        snaps = {(s["call"], s["stream_turn_index"]): s["snapshot"] for s in lines(base / "graph_snapshots.jsonl")}
        for row in lines(base / "requests.jsonl"):
            if not row.get("call"):
                continue  # never use the post-answer flush as an Agent input
            snap = snaps[(row["call"], row["graph_n_turns_ingested"] - 1)]
            text, ids, edges, tasks = render_task_state(snap, 1800)
            assert len(text) <= 1800 and set(ids) <= {n["id"] for n in snap["nodes"]}
            assert set(edges) <= {e["id"] for e in snap["relations"]}
            old = row["task_state_message"]["content"]
            entry = {"run": run, "call": row["call"], "before_chars": row["task_state"]["chars"],
                     "after_chars": len(text), "empty_message_omitted": not text, "task_ids": tasks,
                     "source_node_ids": ids, "relation_ids": edges, "rendered": text}
            if run == 36 and row["call"] in (8, 11):
                i = 104 if row["call"] == 8 else 141
                claim = next(n["belief"] for n in snap["nodes"] if n["id"] == i)
                entry["shared_progress_full_text_occurrences"] = [old.count(claim), text.count(claim)]
                assert entry["shared_progress_full_text_occurrences"] == [2, 1]
            report["render_only"].append(entry)
    # New fields were not requested by the old prompt. Do not pretend it produced them.
    seed = []
    for turn, ident, field, quote in [
        (6, 41, "gap_quote", "Bangladesh's BRAC/CTFP program - found mixed information with some positive effects reported, but I need to search more for failures"),
        (14, 134, "next_action_quote", "focusing on self-exclusion mechanisms and invisible design barriers that perpetuate exclusion"),
    ]:
        case = read(out / f"0035-t{turn}.input.json")
        original = next(n for n in case["original_nodes"] if n["id"] == ident)
        assert any(quote in s for s in original["supporting_excerpts"])
        node = {k: original[k] for k in ("id", "role", "belief", "supporting_excerpts")}
        classify_node(node, {"task_role": original["task_role"], "task_scope": "root", field: quote},
                      seed, original_task=case["root"])
        seed.append(node)
    graph = {"nodes": seed, "relations": []}
    graph["task_state"] = build_task_state(graph)
    text, ids, _, tasks = render_task_state(graph, 1800)
    assert ids == [41, 134] and not tasks and "open gap" in text and "next step" in text
    report["explicit_fixtures"].append({"scope": "overall only; no link between B41 and B134",
                                        "state": graph["task_state"]["root"], "rendered": text})
    path = out / "offline-audit.json"
    if path.exists():
        if read(path) != report:
            raise ValueError("Audit changed; prepare a fresh output directory rather than overwriting it")
    else:
        write(path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    {"prepare": prepare, "audit": audit, "run": run}[args.action](args.out)
