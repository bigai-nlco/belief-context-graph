"""Two-stage review with bounded calls, exact citations and restart-safe artifacts."""
from __future__ import annotations

import fcntl
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tools.ai_review.client import LiteLLMReviewer, ReviewConfig, ReviewError
from tools.ai_review.evidence import batches, prepare_case, sha
from tools.ai_review.prompts import PROMPT_VERSION, READABLE_REVIEW_VERSIONS, messages
from tools.ai_review.rules import validate_concise_summary, validate_groups, validate_search
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
        validate_search(finding, step)


def validate_summary(value: Summary, candidates: list[dict], packet: dict) -> None:
    pool = {c["id"]: c for c in candidates}
    selected = set(value.selected_finding_ids)
    if len(selected) != len(value.selected_finding_ids) or not selected <= pool.keys():
        raise ValueError("selected_finding_ids must be unique existing candidates")
    if len({r.dimension for r in value.ratings}) != len(value.ratings):
        raise ValueError("each dimension must be rated once")
    dimensions = validate_groups(value, pool)
    linked = set()
    for rating in value.ratings:
        if len(set(rating.finding_ids)) != len(rating.finding_ids):
            raise ValueError("duplicate finding reference")
        for ref in rating.finding_ids:
            if ref not in selected or rating.dimension not in dimensions[ref]:
                raise ValueError("ratings must link selected candidates supporting that dimension")
            if rating.dimension == "A2" and rating.utilization in {"未利用", "误用"}:
                verdict = (pool[ref].get("search_assessment") or {}).get("verdict")
                if verdict in {"合理核验", "新增信息搜索", "证据不足"}:
                    raise ValueError("A2 rating contradicts the search assessment")
            linked.add(ref)
    if linked != selected:
        raise ValueError("each selected finding must be linked to a dimension")
    validate_concise_summary(value, pool)
    for text in [r.note for r in value.ratings] + [str(v) for v in value.conclusion.model_dump().values()]:
        if not set(re.findall(r"\bF\d{3,}\b", text)) <= selected:
            raise ValueError("prose must refer only to selected representative findings")
    assessment = value.task_assessment
    if not assessment.rationale.strip():
        raise ValueError("task assessment requires a substantive rationale")
    for quotes, source in ((assessment.task_quotes, packet["question"]["text"]),
                           (assessment.report_quotes, packet["final_report"]["text"])):
        if any(not quote.strip() or quote not in source for quote in quotes):
            raise ValueError("task assessment quotes must exactly match the question or final report")
    truncated = packet["question"]["truncated"] or packet["final_report"]["truncated"]
    if truncated and not assessment.limitations:
        raise ValueError("task assessment must disclose truncated evidence in limitations")
    if truncated and assessment.result == "正确 / 达标":
        raise ValueError("truncated task/report cannot establish full task compliance")
    if any(not item.strip() for item in assessment.limitations):
        raise ValueError("task assessment limitations cannot be blank")


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
                validate_summary(value, candidates, packet)
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


def annotation(case: dict, summary: dict, candidates: list[dict], *, model: str) -> dict:
    pool, steps = {f["id"]: f for f in candidates}, {s["id"]: s for s in case["steps"]}
    groups = {g["representative_id"]: g for g in summary["process_groups"]}
    events = []
    for ref in summary["selected_finding_ids"]:
        f = pool[ref]
        s = steps[f["anchor_id"]]
        group = groups[ref]
        occurrences = []
        seen_steps = {f["anchor_id"]}
        for member_id in group["member_ids"]:
            member = pool[member_id]
            for step_id in [member["anchor_id"], *member["related_ids"]]:
                if step_id in seen_steps:
                    continue
                seen_steps.add(step_id)
                other = steps[step_id]
                occurrences.append({"sourceLocator": other["source"], "agent": other["agent"],
                    "call": other["call"], "quote": other["action_text"]["text"],
                    "graphEvidence": {"text": other["graph"]["text"], "call": other["call"], "status": other["graph"]["status"]}})
        events.append({"id": ref, "label": f["label"], "agent": s["agent"], "sourceRef": "",
            "sourceLocator": s["source"],
            "reference": f"{s['source']['path']} · 记录 {s['source']['index']} · call {s['call']}",
            "quote": s["action_text"]["text"], "objective": "", "original": "", "ingestion": "",
            "need": f["expected"], "behavior": f["observed"], "impact": f["impact"],
            "visible": "\n\n".join(f["graph_quotes"]), "alternatives": f["alternative_explanation"],
            "fault": "无明显失效" if f["kind"] == "benefit" else "Agent 使用",
            "strength": f["strength"], "dimensions": sorted({d for member in group["member_ids"] for d in pool[member]["dimensions"]}),
            "graphEvidence": {"text": s["graph"]["text"], "call": s["call"], "status": s["graph"]["status"]},
            "reportQuote": f["report_quote"], "ai_kind": f["kind"],
            "searchAssessment": f["search_assessment"], "relatedOccurrences": occurrences,
            "groupedFindingIds": group["member_ids"], "groupReason": group["reason"]})
    dimensions = {key: {"opportunity": "无", "utilization": "", "effect": "", "eventIds": [],
                         "evidence": "", "note": ""} for key in DIMENSIONS}
    for r in summary["ratings"]:
        dimensions[r["dimension"]].update(opportunity="有", utilization=r["utilization"],
                                         effect=r["effect"], eventIds=r["finding_ids"], note=r["note"])
    labels = {e["id"]: f"E{i:02}" for i, e in enumerate(events, 1)}
    def display_refs(text: str) -> str:
        return re.sub(r"\bF\d{3,}\b", lambda match: labels.get(match[0], match[0]), text)
    for row in dimensions.values():
        row["note"] = display_refs(row["note"])
    conclusion = {key: display_refs(value) for key, value in summary["conclusion"].items()}
    return {"reviewer": model, "status": "draft", "goal": summary["goal"],
            "criteria": "", "settings": "", "team": "单 Agent" if len({s["agent"] for s in case["steps"]}) == 1 else "多 Agent",
            "result": summary["task_assessment"]["result"],
            "resultAssessment": {"source": "ai", **summary["task_assessment"]},
            "customDimensions": [], "noApplicableDimensions": not events, "reviewFormat": "focused-v3",
            "dimensions": dimensions, "events": events,
            "conclusion": {**conclusion, "focus": ["Agent 使用"] if any(e["ai_kind"] != "benefit" for e in events) else []}}


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
    assessment = a.get("resultAssessment")
    if assessment:
        lines.extend(["## 任务结果 · AI 评估", "", f"结果：{a['result']}", "", assessment["rationale"], ""])
        for key, label in (("task_quotes", "任务要求原文"), ("report_quotes", "最终报告原文")):
            lines.extend([f"### {label}", ""])
            for quote in assessment[key]:
                lines.extend(["> " + quote.replace("\n", "\n> "), ""])
        if assessment["limitations"]:
            lines.extend(["### 判断范围与待核实内容", "", *[f"- {item}" for item in assessment["limitations"]], ""])
    labels = {e["id"]: f"E{i:02}" for i, e in enumerate(a["events"], 1)}
    for e in a["events"]:
        lines.extend([f"## {labels[e['id']]} · {e['label']}", "", "### Belief 注入原文", "", "> " + e["visible"].replace("\n", "\n> "), "",
                      f"### 第 {e['sourceLocator']['call']} 轮 call", "", "> " + e["quote"].replace("\n", "\n> "), ""])
        if e["reportQuote"]:
            lines.extend(["### 最终报告原文节选", "", "> " + e["reportQuote"].replace("\n", "\n> "), ""])
        for key, label in (("need", "BCG 应发挥的作用"), ("behavior", "实际情况"),
                           ("impact", "影响"), ("alternatives", "其他解释"), ("strength", "证据强度")):
            lines.extend([f"{label}：{e[key]}", ""])
        if e.get("searchAssessment"):
            search = e["searchAssessment"]
            lines.extend(["<details>", f"<summary>搜索判断 · {search['verdict']}</summary>", ""])
            for key, label in (("information_need", "信息需求"), ("graph_sufficiency", "Graph 是否已满足"),
                               ("verification_reason", "核验目的"), ("expected_gain", "预期新增内容"),
                               ("basis", "判断依据"), ("observed_gain", "实际新增证据"),
                               ("result_quote", "工具结果原文"), ("unavailable_quote", "事前已知的工具状态")):
                if search[key]:
                    lines.extend([f"{label}：{search[key]}", ""])
            lines.extend(["</details>", ""])
        if e.get("relatedOccurrences"):
            lines.extend(["<details>", "<summary>同类过程与关联轮次</summary>", "", e["groupReason"], ""])
            for other in e["relatedOccurrences"]:
                lines.extend([f"### {other['agent']} · call {other['call']}", "", "Belief 注入原文：", "",
                              "> " + other["graphEvidence"]["text"].replace("\n", "\n> "), "", "Agent 输出：", "",
                              "> " + other["quote"].replace("\n", "\n> "), ""])
            lines.extend(["</details>", ""])
    lines.extend(["## 维度评价", ""])
    for key, r in a["dimensions"].items():
        if r["opportunity"] == "有":
            refs = "、".join(labels[ref] for ref in r["eventIds"])
            lines.extend([f"### {key} {DIMENSIONS[key]}", "", f"{r['utilization']} · {r['effect']} · 见 {refs}", "", r["note"], ""])
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
        def fingerprint_for(version: str) -> str:
            return sha(json.dumps([case["evidence_sha256"], config.public(), version, chunk_chars, chunk_turns], sort_keys=True))
        fingerprint = fingerprint_for(PROMPT_VERSION)
        manifest_path = output / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        legacy = False
        if previous and previous.get("fingerprint") != fingerprint:
            compatible = (previous.get("prompt_version") in READABLE_REVIEW_VERSIONS
                          and previous.get("fingerprint") == fingerprint_for(previous["prompt_version"]))
            if not compatible:
                raise ReviewError("输出目录已有其他证据或配置的结果，请使用新目录")
            legacy = True
        if previous.get("status") == "completed":
            return json.loads((output / "review.json").read_text())
        if previous.get("status") in {"running", "failed"} and not retry_failed:
            raise ReviewError("上次 Review 未完成；使用 --retry 续跑，已完成的批次不会重复请求")
        if legacy:
            if not retry_failed:
                raise ReviewError("Review 规则已更新；使用新目录或 --retry 按新规则重新分析")
            # Explicit retry only: preserve the old artifacts and the cumulative
            # usage journal. Old findings must not bypass the v3 search checks.
            archive = output / "previous-versions" / previous["prompt_version"]
            archive.mkdir(parents=True, exist_ok=True)
            # Move the manifest last so an interrupted migration remains
            # identifiable as legacy and can continue without overwriting files.
            for path in sorted(output.glob("*.json"), key=lambda item: item.name == "manifest.json"):
                target = archive / path.name
                if target.exists():
                    raise ReviewError("旧版本归档已存在同名文件，请使用新的 Review 输出目录")
                path.rename(target)
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
            write_json(output / "summary-prompt-preview.json", messages("summary", {
                "question": case["question"], "final_report": case["final_report"],
                "candidates": [], "coverage": case["coverage"], "limitations": [],
            }))
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
                      "usage": usage(output / "calls.jsonl"), "annotation": annotation(case, summary, candidates, model=config.model)}
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
