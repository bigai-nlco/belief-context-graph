"""Two-stage review with bounded calls, exact citations and restart-safe artifacts."""
from __future__ import annotations

import fcntl
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tools.ai_review.client import LiteLLMReviewer, ReviewConfig, ReviewError
from tools.ai_review.evidence import batches, prepare_case, sha
from tools.ai_review.prompts import PROMPT_VERSION, messages
from tools.ai_review.schema import DIMENSIONS, Findings, Summary


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def validate_findings(value: Findings, packet: dict) -> None:
    steps = {s["id"]: s for s in packet["steps"]}
    for finding in value.findings:
        step = steps.get(finding.anchor_id)
        if not step or finding.anchor_id not in packet["eligible_ids"]:
            raise ValueError("anchor_id must be a shown eligible record")
        if step["graph"]["status"] != "matched":
            raise ValueError("finding requires a recorded, unambiguous Graph injection")
        if any(len(q.strip()) < 8 or q not in step["graph"]["text"] for q in finding.graph_quotes):
            raise ValueError("graph_quotes must exactly quote the anchor's injected Graph")
        if len(finding.action_quote.strip()) < 8 or finding.action_quote not in step["action_text"]["text"]:
            raise ValueError("action_quote must exactly quote the anchor's Assistant output")
        if finding.report_quote and finding.report_quote not in packet["final_report"]["text"]:
            raise ValueError("report_quote must exactly quote the final report")
        if any(ref not in steps for ref in finding.related_ids):
            raise ValueError("related_ids must refer to shown records")
        if "A7" in finding.dimensions:
            raise ValueError("A7 cannot be evaluated from one run without a controlled comparison")
        if "A8" in finding.dimensions and len({s["agent"] for s in packet["steps"]}) < 2:
            raise ValueError("A8 requires actual multi-agent evidence")


def validate_summary(value: Summary, candidates: list[dict]) -> None:
    pool = {c["id"]: c for c in candidates}
    selected = set(value.selected_finding_ids)
    if len(selected) != len(value.selected_finding_ids) or not selected <= pool.keys():
        raise ValueError("selected_finding_ids must be unique existing candidates")
    if len({r.dimension for r in value.ratings}) != len(value.ratings):
        raise ValueError("each dimension must be rated once")
    linked = set()
    for rating in value.ratings:
        if len(set(rating.finding_ids)) != len(rating.finding_ids):
            raise ValueError("duplicate finding reference")
        for ref in rating.finding_ids:
            if ref not in selected or rating.dimension not in pool[ref]["dimensions"]:
                raise ValueError("ratings must link selected candidates supporting that dimension")
            linked.add(ref)
    if linked != selected:
        raise ValueError("each selected finding must be linked to a dimension")


def validated_call(client: LiteLLMReviewer, stage: str, packet: dict, *, candidates: list[dict] | None = None) -> dict:
    prompt = messages("summary" if candidates is not None else "findings", packet)
    schema = Summary if candidates is not None else Findings
    for attempt in range(2):
        raw = client.complete(prompt, stage + (".repair" if attempt else ""))
        try:
            value = schema.model_validate_json(raw)
            if candidates is None:
                validate_findings(value, packet)
            else:
                validate_summary(value, candidates)
            return value.model_dump()
        except (ValidationError, ValueError) as exc:
            if attempt:
                raise ReviewError("AI 标注两次未通过结构或原文引用校验，未作为有效标注发布") from None
            # Validation errors can echo arbitrary evidence. Send a short diagnostic,
            # not credentials or the entire malformed response, to the correction call.
            reason = "JSON schema mismatch" if isinstance(exc, ValidationError) else str(exc)
            prompt = [*prompt, {"role": "assistant", "content": raw},
                      {"role": "user", "content": "校验未通过：" + reason + "。请根据原始材料重新输出完整 JSON；不要编造引用。"}]
    raise ReviewError("Review validation failed")


def annotation(case: dict, summary: dict, candidates: list[dict]) -> dict:
    pool, steps = {f["id"]: f for f in candidates}, {s["id"]: s for s in case["steps"]}
    events = []
    for ref in summary["selected_finding_ids"]:
        f = pool[ref]
        s = steps[f["anchor_id"]]
        events.append({"id": ref, "label": f["label"], "agent": s["agent"], "sourceRef": "",
            "sourceLocator": s["source"],
            "reference": f"{s['source']['path']} · 记录 {s['source']['index']} · call {s['call']}",
            "quote": s["action_text"]["text"], "objective": "", "original": "", "ingestion": "",
            "need": f["expected"], "behavior": f["observed"], "impact": f["impact"],
            "visible": "\n\n".join(f["graph_quotes"]), "alternatives": f["alternative_explanation"],
            "fault": "无明显失效" if f["kind"] == "benefit" else "Agent 使用",
            "strength": f["strength"], "dimensions": f["dimensions"],
            "graphEvidence": {"text": s["graph"]["text"], "call": s["call"], "status": s["graph"]["status"]},
            "reportQuote": f["report_quote"], "ai_kind": f["kind"]})
    dimensions = {key: {"opportunity": "无", "utilization": "", "effect": "", "eventIds": [],
                         "evidence": "", "note": ""} for key in DIMENSIONS}
    for r in summary["ratings"]:
        dimensions[r["dimension"]].update(opportunity="有", utilization=r["utilization"],
                                         effect=r["effect"], eventIds=r["finding_ids"], note=r["note"])
    return {"reviewer": "AI · GPT-6.1-Sol", "status": "draft", "goal": summary["goal"],
            "criteria": "", "settings": "", "team": "单 Agent" if len({s["agent"] for s in case["steps"]}) == 1 else "多 Agent",
            "result": "尚未评判", "customDimensions": [], "noApplicableDimensions": not events,
            "dimensions": dimensions, "events": events,
            "conclusion": {**summary["conclusion"], "focus": ["Agent 使用"] if any(e["ai_kind"] != "benefit" for e in events) else []}}


def usage(journal: Path) -> dict:
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
             "total_tokens": 0, "reported_cost": 0.0, "cost_reported_calls": 0}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            u = row.get("usage") or {}
            total["calls"] += 1
            total["input_tokens"] += u.get("prompt_tokens", 0) or 0
            total["output_tokens"] += u.get("completion_tokens", 0) or 0
            total["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            if isinstance(u.get("cost"), (float, int)):
                total["reported_cost"] += u["cost"]
                total["cost_reported_calls"] += 1
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    total["cost_complete"] = total["calls"] > 0 and total["cost_reported_calls"] == total["calls"]
    return total


def markdown(result: dict) -> str:
    a = result["annotation"]
    lines = [f"# Task {result['task_id']} · AI Review", "", a["goal"], "", "AI 标注草稿，不是官方正确率评分。", ""]
    for e in a["events"]:
        lines.extend([f"## {e['label']}", "", "### Belief 注入原文", "", "> " + e["visible"].replace("\n", "\n> "), "",
                      f"### 第 {e['sourceLocator']['call']} 轮 call", "", "> " + e["quote"].replace("\n", "\n> "), ""])
        if e["reportQuote"]:
            lines.extend(["### 最终报告原文节选", "", "> " + e["reportQuote"].replace("\n", "\n> "), ""])
        for key, label in (("need", "BCG 应发挥的作用"), ("behavior", "实际情况"),
                           ("impact", "影响与改进线索"), ("alternatives", "其他解释"), ("strength", "证据强度")):
            lines.extend([f"{label}：{e[key]}", ""])
    lines.extend(["## 维度评价", ""])
    for key, r in a["dimensions"].items():
        if r["opportunity"] == "有":
            lines.extend([f"### {key} {DIMENSIONS[key]}", "", f"{r['utilization']} · {r['effect']}", "", r["note"], ""])
    lines.extend(["## 归因总结", ""])
    for key, label in (("benefit", "主要作用"), ("problem", "主要问题"), ("improvement", "改进方案"),
                       ("comparison", "对照与其他因素"), ("verification", "验证方法")):
        lines.extend([f"{label}：{a['conclusion'][key]}", ""])
    lines.extend(["## 分析范围与用量", "", "```json", json.dumps({"coverage": result["coverage"],
                  "usage": result["usage"]}, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def run_review(source: Path, output: Path, *, config: ReviewConfig | None = None,
               dry_run: bool = False, retry_failed: bool = False, chunk_chars: int = 60000,
               chunk_turns: int = 6, max_batches: int = 64, client: LiteLLMReviewer | None = None) -> dict:
    config = config or ReviewConfig.from_env()
    if not 12000 <= chunk_chars <= 200000 or not 1 <= chunk_turns <= 12 or not 1 <= max_batches <= 100:
        raise ValueError("Invalid review batching limits")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ReviewError("同一输出目录已有 AI Review 正在运行") from None
        case = prepare_case(source)
        packets = batches(case, chunk_chars=chunk_chars, chunk_turns=chunk_turns)
        if len(packets) > max_batches:
            raise ReviewError("轨迹超出当前批次数量上限；未发起模型请求")
        fingerprint = sha(json.dumps([case["evidence_sha256"], config.public(), PROMPT_VERSION, chunk_chars, chunk_turns], sort_keys=True))
        manifest_path = output / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if previous and previous.get("fingerprint") != fingerprint:
            raise ReviewError("输出目录已有其他证据或配置的结果，请使用新目录")
        if previous.get("status") == "completed":
            return json.loads((output / "review.json").read_text())
        if previous.get("status") in {"running", "failed"} and not retry_failed:
            raise ReviewError("上次 Review 未完成；使用 --retry 续跑，已完成的批次不会重复请求")
        for index, packet in enumerate(packets):
            write_json(output / f"input-{index + 1:03}.json", packet)
        write_json(output / "prepared.json", {k: v for k, v in case.items() if k != "steps"})
        state = {"fingerprint": fingerprint, "status": "prepared" if dry_run else "running",
                 "model": config.public(), "prompt_version": PROMPT_VERSION, "batches": len(packets),
                 "task_id": case["task_id"], "evidence_sha256": case["evidence_sha256"],
                 "started_at": datetime.now(UTC).isoformat()}
        write_json(manifest_path, state)
        if dry_run:
            write_json(output / "prompt-preview.json", messages("findings", packets[0]))
            return state
        own_client = client is None
        started = time.monotonic()
        try:
            client = client or LiteLLMReviewer(config, output / "calls.jsonl")
            candidates, limitations = [], []
            for index, packet in enumerate(packets):
                target = output / f"findings-{index + 1:03}.json"
                if target.exists():
                    found = json.loads(target.read_text())
                    validate_findings(Findings.model_validate(found), packet)
                else:
                    found = validated_call(client, f"findings-{index + 1:03}", packet)
                    write_json(target, found)
                for finding in found["findings"]:
                    candidates.append({**finding, "id": f"F{len(candidates) + 1:03}"})
                limitations.extend(found["limitations"])
            write_json(output / "candidates.json", candidates)
            packet = {"question": case["question"], "final_report": case["final_report"],
                      "candidates": candidates, "coverage": case["coverage"], "limitations": limitations}
            write_json(output / "summary-input.json", packet)
            summary = validated_call(client, "summary", packet, candidates=candidates)
            write_json(output / "summary.json", summary)
            result = {"format": "bcg-ai-review", "version": 1, "task_id": case["task_id"],
                      "benchmark": case["benchmark"], "evidence_sha256": case["evidence_sha256"],
                      "model": config.public(), "prompt_version": PROMPT_VERSION,
                      "coverage": {**case["coverage"], "limitations": limitations,
                                   "question_truncated": case["question"]["truncated"],
                                   "report_truncated": case["final_report"]["truncated"]},
                      "usage": usage(output / "calls.jsonl"), "annotation": annotation(case, summary, candidates)}
            write_json(output / "review.json", result)
            (output / "review.md").write_text(markdown(result), encoding="utf-8")
            state.update(status="completed", finished_at=datetime.now(UTC).isoformat(),
                         duration_seconds=round(time.monotonic() - started, 3), usage=result["usage"])
            write_json(manifest_path, state)
            return result
        except Exception as exc:
            state.update(status="failed", error=str(exc) if isinstance(exc, ReviewError) else type(exc).__name__,
                         usage=usage(output / "calls.jsonl"))
            write_json(manifest_path, state)
            raise
        finally:
            if own_client and client is not None:
                client.close()
