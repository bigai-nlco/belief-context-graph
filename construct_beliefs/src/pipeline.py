"""
pipeline.py
===========
Top-level orchestration. Stages save their own output file, so partial runs
are possible. Each stage is also callable independently from scripts/run.py.

Stages
------
    1. segment      → outputs/01_segments.json
    2. io extract   → outputs/02_io_beliefs.json
    3. reasoning extract → outputs/03_reasoning_beliefs.json
    4. linking      → outputs/04_relations.json
    5. apply confidence updates → outputs/result.json   (FINAL, compatible
                                                         with visualize_beliefs_v3.py)

Token accounting
----------------
Every LLM call is logged (input/output tokens, tagged per stage) via the shared
llm.USAGE tracker. After a full run the log is written to
    outputs/token_usage.json   (full per-call detail + totals)
    outputs/token_usage.txt    (human-readable summary table)
and a summary is also embedded under "token_usage" in result.json. Add an
optional "pricing" block to model_config.json to also get a cost estimate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .confidence import apply_relations, initial_confidence
from .extract import attach_source_and_id, extract_one_segment
from .link import link_backward, link_forward
from .llm import USAGE, load_config, make_client
from .segment import IO_TYPES, REASONING_TYPES, Segment, segment_trajectory, summarize_segments


# Minimum content length below which a segment is auto-skipped.
# Saves API calls on boilerplate like "Your answer has been submitted."
MIN_SEGMENT_LEN_BY_TYPE: Dict[str, int] = {
    "tool_response":  60,
    "assistant_other": 20,
    "user_input":      0,
    "tool_call":       0,
    "think":           0,
}


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 1 — segmentation
# ---------------------------------------------------------------------------

def stage_segment(input_path: str, output_dir: Path) -> List[Segment]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trajectory = data.get("trajectory", [])
    segments = segment_trajectory(trajectory)
    payload = {
        "input_path": input_path,
        "trajectory": trajectory,
        "summary": summarize_segments(segments),
        "segments": [s.to_dict() for s in segments],
    }
    _save_json(payload, output_dir / "01_segments.json")
    print(f"  [segment] {len(segments)} segments  ({payload['summary']})")
    print(f"  saved -> {output_dir / '01_segments.json'}")
    return segments


def _load_segments(output_dir: Path) -> Tuple[List[Segment], List[Dict[str, Any]]]:
    """Reload segments + trajectory from the stage-1 output file."""
    payload = _load_json(output_dir / "01_segments.json")
    segs = [Segment(**s) for s in payload["segments"]]
    return segs, payload["trajectory"]


# ---------------------------------------------------------------------------
# Stage 2 — I/O belief extraction
# ---------------------------------------------------------------------------

def stage_io(
    client, model: str, segments: List[Segment], output_dir: Path,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    io_segments = [s for s in segments if s.type in IO_TYPES]
    print(f"  [io] processing {len(io_segments)} I/O segments")

    all_beliefs: List[Dict[str, Any]] = []
    per_segment: List[Dict[str, Any]] = []
    next_id = 0

    for seg in io_segments:
        if len(seg.content.strip()) < MIN_SEGMENT_LEN_BY_TYPE.get(seg.type, 0):
            per_segment.append({
                "segment": seg.to_dict(),
                "beliefs": [],
                "raw_output": None,
                "skipped": True,
                "skip_reason": "too short",
            })
            print(f"    traj[{seg.traj_idx}].seg[{seg.seg_idx}] {seg.type} -> SKIP (too short)")
            continue

        USAGE.set_label(f"io:traj[{seg.traj_idx}].seg[{seg.seg_idx}]")
        result = extract_one_segment(client, model, seg, max_tokens=max_tokens)
        tagged = attach_source_and_id(result["beliefs"], seg, next_id)
        # Set initial confidence
        for b in tagged:
            b["confidence"] = initial_confidence(b["source"]["type"], b["stance"])
        next_id += len(tagged)
        all_beliefs.extend(tagged)
        per_segment.append({
            "segment":    seg.to_dict(),
            "beliefs":    tagged,
            "raw_output": result["raw_output"],
            "skipped":    result.get("skipped", False),
        })
        print(f"    traj[{seg.traj_idx}].seg[{seg.seg_idx}] {seg.type:<18} -> {len(tagged)} belief(s)")

    payload = {
        "layer": "io",
        "beliefs": all_beliefs,
        "per_segment": per_segment,
    }
    _save_json(payload, output_dir / "02_io_beliefs.json")
    print(f"  [io] {len(all_beliefs)} belief(s) total")
    print(f"  saved -> {output_dir / '02_io_beliefs.json'}")
    return all_beliefs


# ---------------------------------------------------------------------------
# Stage 3 — reasoning belief extraction (selective, given I/O context)
# ---------------------------------------------------------------------------

def stage_reasoning(
    client, model: str, segments: List[Segment],
    io_beliefs: List[Dict[str, Any]], output_dir: Path,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    reasoning_segments = [s for s in segments if s.type in REASONING_TYPES]
    print(f"  [reasoning] processing {len(reasoning_segments)} reasoning segments")

    all_beliefs: List[Dict[str, Any]] = []
    per_segment: List[Dict[str, Any]] = []
    next_id = max((b["id"] for b in io_beliefs), default=-1) + 1

    for seg in reasoning_segments:
        if len(seg.content.strip()) < MIN_SEGMENT_LEN_BY_TYPE.get(seg.type, 0):
            per_segment.append({
                "segment": seg.to_dict(),
                "beliefs": [],
                "raw_output": None,
                "skipped": True,
                "skip_reason": "too short",
            })
            print(f"    traj[{seg.traj_idx}].seg[{seg.seg_idx}] {seg.type:<18} -> SKIP (too short)")
            continue

        USAGE.set_label(f"reasoning:traj[{seg.traj_idx}].seg[{seg.seg_idx}]")
        result = extract_one_segment(client, model, seg, io_context=io_beliefs, max_tokens=max_tokens)
        tagged = attach_source_and_id(result["beliefs"], seg, next_id)
        for b in tagged:
            b["confidence"] = initial_confidence(b["source"]["type"], b["stance"])
        next_id += len(tagged)
        all_beliefs.extend(tagged)
        per_segment.append({
            "segment":    seg.to_dict(),
            "beliefs":    tagged,
            "raw_output": result["raw_output"],
            "skipped":    result.get("skipped", False),
        })
        print(f"    traj[{seg.traj_idx}].seg[{seg.seg_idx}] {seg.type:<18} -> {len(tagged)} belief(s)")

    payload = {
        "layer": "reasoning",
        "beliefs": all_beliefs,
        "per_segment": per_segment,
    }
    _save_json(payload, output_dir / "03_reasoning_beliefs.json")
    print(f"  [reasoning] {len(all_beliefs)} belief(s) total")
    print(f"  saved -> {output_dir / '03_reasoning_beliefs.json'}")
    return all_beliefs


# ---------------------------------------------------------------------------
# Stage 4 — forward (derivation) linking
# ---------------------------------------------------------------------------

def stage_forward(
    client, model: str, all_beliefs: List[Dict[str, Any]], output_dir: Path,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    USAGE.set_label("forward")
    result = link_forward(client, model, all_beliefs, max_tokens=max_tokens)
    payload = {
        "n_beliefs": len(all_beliefs),
        "forward_relations": result["forward_relations"],
        "raw_output": result.get("raw_output"),
        "skipped": result.get("skipped", False),
    }
    _save_json(payload, output_dir / "04_forward_relations.json")
    print(f"  [forward] {len(result['forward_relations'])} informs edge(s)")
    print(f"  saved -> {output_dir / '04_forward_relations.json'}")
    return result["forward_relations"]


# ---------------------------------------------------------------------------
# Stage 5 — backward (evaluation) linking
# ---------------------------------------------------------------------------

def stage_backward(
    client, model: str, all_beliefs: List[Dict[str, Any]], output_dir: Path,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    USAGE.set_label("backward")
    result = link_backward(client, model, all_beliefs, max_tokens=max_tokens)
    payload = {
        "n_beliefs": len(all_beliefs),
        "relations": result["relations"],
        "raw_output": result.get("raw_output"),
        "skipped": result.get("skipped", False),
    }
    _save_json(payload, output_dir / "05_backward_relations.json")
    print(f"  [backward] {len(result['relations'])} relation(s)")
    if result["relations"]:
        from collections import Counter
        type_counts = Counter(r["type"] for r in result["relations"])
        print(f"           by type: {dict(type_counts)}")
    print(f"  saved -> {output_dir / '05_backward_relations.json'}")
    return result["relations"]


# ---------------------------------------------------------------------------
# Stage 6 — assemble final, apply dynamic confidence updates
# ---------------------------------------------------------------------------

def stage_finalize(
    trajectory: List[Dict[str, Any]],
    io_beliefs: List[Dict[str, Any]],
    reasoning_beliefs: List[Dict[str, Any]],
    forward_relations: List[Dict[str, Any]],
    backward_relations: List[Dict[str, Any]],
    model: str,
    output_dir: Path,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    all_beliefs = io_beliefs + reasoning_beliefs
    # Only backward relations drive confidence updates.
    updated = apply_relations(all_beliefs, backward_relations)

    payload: Dict[str, Any] = {
        "prompt_name": "construct_beliefs",
        "model": model,
        "trajectory": trajectory,
        "io_beliefs":        [b for b in updated if b.get("layer") == "io"],
        "reasoning_beliefs": [b for b in updated if b.get("layer") == "reasoning"],
        "forward_relations":  forward_relations,
        "backward_relations": backward_relations,
        "all_beliefs": updated,
        "source_counts": _count_by(updated, lambda b: b["source"]["type"]),
        "layer_counts":  _count_by(updated, lambda b: b.get("layer")),
        "stance_counts": _count_by(updated, lambda b: b["stance"]),
    }
    if extra_meta:
        payload.update(extra_meta)
    _save_json(payload, output_dir / "result.json")
    print(f"  [finalize] {len(updated)} belief(s); "
          f"{len(forward_relations)} forward + {len(backward_relations)} backward relation(s)")
    print(f"  saved -> {output_dir / 'result.json'}")
    return payload


def _count_by(items: List[Dict[str, Any]], key_fn) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        k = key_fn(it) or "unknown"
        out[k] = out.get(k, 0) + 1
    return out


def renumber_chronologically(beliefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reassign belief ids so they reflect the order beliefs would be encountered
    when reading the trajectory linearly. Sort key is
        (trajectory_index, segment_index, original_id).

    Why this matters: stage_io assigns ids 0..N to I/O beliefs and stage_reasoning
    assigns N+1..N+M to reasoning beliefs.  Within a single assistant message a
    `<think>` segment (segment_index=0) appears BEFORE a `<tool_call>` segment
    (segment_index=1), but with the per-layer numbering scheme the think
    belief gets a HIGHER id than the tool_call belief.  That breaks the
    forward-direction check (`from_id < to_id`) for legitimate edges like
    `think -> tool_call`.  Chronological renumbering fixes this so the prompt
    can keep the simple "earlier id == earlier in conversation" contract.

    Returns a NEW list of belief dicts (originals are not mutated).
    """
    def key(b: Dict[str, Any]):
        src = b.get("source") or {}
        return (
            src.get("trajectory_index", -1),
            src.get("segment_index", 0),
            b.get("id", 0),
        )
    sorted_beliefs = sorted(beliefs, key=key)
    out: List[Dict[str, Any]] = []
    for new_id, b in enumerate(sorted_beliefs):
        nb = dict(b)
        nb["id"] = new_id
        out.append(nb)
    return out


# ---------------------------------------------------------------------------
# Run-all driver
# ---------------------------------------------------------------------------

def run_all(
    input_path: str, config_path: str, output_dir: Path,
    model_key: Optional[str] = None,
) -> None:
    cfg = load_config(config_path, model_key=model_key)
    client = make_client(cfg)
    model = cfg.get("model") or cfg.get("model_name") or "gpt-4o-mini"
    max_tokens = cfg.get("max_tokens")
    # Optional pricing block for a cost estimate, e.g. in model_config.json:
    #   "pricing": {"input_per_1k": 0.00015, "output_per_1k": 0.0006}
    pricing = cfg.get("pricing")

    # Start a clean token-usage log for this input.
    USAGE.reset()

    # Show what was actually loaded (mask the api_key)
    masked_key = (cfg.get("api_key", "") or "")
    masked_key = (masked_key[:6] + "…" + masked_key[-3:]) if len(masked_key) > 10 else "***"
    print(f"[info] model={model}  base_url={cfg['base_url']}  api_key={masked_key}"
          + (f"  max_tokens={max_tokens}" if max_tokens else ""))
    print(f"[info] output -> {output_dir}")

    print("\n=== stage 1 / segment ===")
    segments = stage_segment(input_path, output_dir)
    trajectory = _load_json(output_dir / "01_segments.json")["trajectory"]

    print("\n=== stage 2 / io extraction (comprehensive) ===")
    io_beliefs = stage_io(client, model, segments, output_dir, max_tokens=max_tokens)

    print("\n=== stage 3 / reasoning extraction (selective) ===")
    reasoning_beliefs = stage_reasoning(client, model, segments, io_beliefs, output_dir, max_tokens=max_tokens)

    # Renumber so belief ids reflect true chronological order across both layers.
    # This is required for the forward/backward direction checks to work.
    combined = renumber_chronologically(io_beliefs + reasoning_beliefs)
    io_beliefs_chrono        = [b for b in combined if b.get("layer") == "io"]
    reasoning_beliefs_chrono = [b for b in combined if b.get("layer") == "reasoning"]

    print("\n=== stage 4 / forward linking (derivation: informs) ===")
    forward_rels = stage_forward(client, model, combined, output_dir, max_tokens=max_tokens)

    print("\n=== stage 5 / backward linking (evaluation: confirms / contradicts / extends) ===")
    backward_rels = stage_backward(client, model, combined, output_dir, max_tokens=max_tokens)

    print("\n=== stage 6 / finalize + dynamic confidence ===")
    stage_finalize(trajectory, io_beliefs_chrono, reasoning_beliefs_chrono,
                   forward_rels, backward_rels,
                   model=model, output_dir=output_dir,
                   extra_meta={"input_path": input_path,
                               "token_usage": USAGE.summary(pricing)})

    print("\n=== token usage / cost ===")
    usage_json = output_dir / "token_usage.json"
    usage_txt = output_dir / "token_usage.txt"
    USAGE.save_json(usage_json, pricing=pricing)
    USAGE.save_text(usage_txt, pricing=pricing)
    totals = USAGE.totals()
    print(f"  LLM calls for this input: {totals['n_calls']}")
    print(f"  tokens: input={totals['input_tokens']:,}  "
          f"output={totals['output_tokens']:,}  total={totals['total_tokens']:,}")
    cost = USAGE.estimate_cost(pricing)
    if cost:
        print(f"  estimated cost: {cost['total_cost']:.6f} {cost['currency']}")
    print(f"  saved -> {usage_json}")
    print(f"  saved -> {usage_txt}")

    print("\n[done]")
