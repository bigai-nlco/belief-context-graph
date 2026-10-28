"""Build a public, code-backed methodology snapshot. No keys, runs or API requests."""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import tempfile
from pathlib import Path

from tools.ai_review.client import ReviewConfig
from tools.ai_review.evidence import batches, prepare_case, sha
from tools.ai_review.pipeline import annotation, run_review, validate_findings, validate_summary
from tools.ai_review.prompts import FIND, PROMPT_VERSION, SYNTHESIZE, SYSTEM, messages
from tools.ai_review.schema import DIMENSIONS, Findings, Summary

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "tools/case_study_studio/review-pipeline-contract.json"
SOURCE_TARGETS = {
    "admit": ("tools/case_study_studio/review_queue.py", "enqueue"),
    "freeze": ("tools/case_study_studio/control.py", "evidence"),
    "load": ("tools/ai_review/evidence.py", "load_files"),
    "align": ("tools/ai_review/evidence.py", "prepare_case"),
    "batch": ("tools/ai_review/evidence.py", "batches"),
    "prompt": ("tools/ai_review/prompts.py", "messages"),
    "client": ("tools/ai_review/client.py", "complete"),
    "repair": ("tools/ai_review/pipeline.py", "validated_call"),
    "findings_gate": ("tools/ai_review/pipeline.py", "validate_findings"),
    "search_gate": ("tools/ai_review/rules.py", "validate_search"),
    "orchestrate": ("tools/ai_review/pipeline.py", "run_review"),
    "summary_gate": ("tools/ai_review/pipeline.py", "validate_summary"),
    "groups_gate": ("tools/ai_review/rules.py", "validate_groups"),
    "concise_gate": ("tools/ai_review/rules.py", "validate_concise_summary"),
    "serialize": ("tools/ai_review/pipeline.py", "annotation"),
    "usage": ("tools/ai_review/pipeline.py", "usage"),
    "publish": ("tools/case_study_studio/review_job.py", "execute"),
}


def illustration() -> dict:
    """Synthetic model outputs validated through the real parser and serializer."""
    question = "请核对项目 X 是否正式运营，并说明规划投运日期、投运条件和依据。"
    graph_a = "B17: 项目 X 仍在规划阶段，目标投运日期为 2030 年。confidence=0.82"
    graph_b = graph_a + "\nB18: 项目 X 尚未投运，投运取决于安全评估与许可。来源：项目官网。confidence=0.88"
    result_text = "项目 X 尚未投运。2030 年是目标日期，投运取决于安全评估与许可。"
    actions = [
        "Graph 的 B17 只有规划摘要，我需要读取官网原文，核对安全评估和许可等投运条件。",
        "搜索项目 X 在 2030 年投运需要哪些安全评估和许可。",
        "再次搜索项目 X 的规划投运时间和审批条件。",
    ]
    report = "# 项目 X 核查报告\n\n项目 X 尚未正式运营。2030 年为规划目标，实际投运仍需通过安全评估并获得许可。\n\n依据：项目官网 https://example.org/project-x 。"
    records: list[dict] = [{"t": "start", "user_message": question}]
    injections = []
    for call, action in enumerate(actions, 1):
        tool = "web_fetch" if call == 1 else "web_search"
        args = {"url": "https://example.org/project-x"} if call == 1 else {"query": action}
        records.extend([
            {"t": "llm", "turn": call, "content": action,
             "tool_calls": [{"name": tool, "arguments": args, "id": f"tool-{call}"}]},
            {"t": "result", "turn": call, "name": tool, "tool_call_id": f"tool-{call}", "result": result_text},
        ])
        graph = graph_a if call == 1 else graph_b
        injections.append({"call": call, "graph_chars": len(graph),
                           "system_memory": "<belief_graph>" + graph + "</belief_graph>"})
    with tempfile.TemporaryDirectory(prefix="bcg-review-method-") as directory:
        root = Path(directory)
        files = {"case.json": json.dumps({"task_id": "METHOD-DEMO", "benchmark": "illustration", "question": question}),
            "result.json": json.dumps({"status": "completed"}), "report.md": report,
            "agent/trajectories/react_agent.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            "agent/bcg/requests.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in injections)}
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        case = prepare_case(root)
    packet = batches(case)[0]
    candidates = []
    for index, step in enumerate(case["steps"]):
        positive = index == 0
        finding = {"anchor_id": step["id"], "kind": "benefit" if positive else "missed_opportunity",
            "label": "从规划摘要定向补原文条件" if positive else "条件已充分注入后重复查找",
            "dimensions": ["A4", "A6"] if positive else ["A1", "A2"],
            "graph_quotes": [graph_a if positive else graph_b.split("\n")[1]], "action_quote": actions[index],
            "report_quote": "", "related_ids": [],
            "expected": "利用摘要定位原文缺口。" if positive else "复用已有日期和条件，推进报告。",
            "observed": "明确引用 B17，并请求官网原文。" if positive else "再次查询已充分注入的条件，返回相同信息。",
            "impact": "获得安全评估与许可条件。" if positive else "本次调用没有增加可见证据。",
            "alternative_explanation": "任务也要求核对条件。" if positive else "近期工具结果也提供了相同信息。",
            "strength": "明确可见" if positive else "有较强迹象",
            "search_assessment": {"verdict": "合理核验" if positive else "无效重复",
                "information_need": "项目 X 的投运条件。", "graph_sufficiency": "部分" if positive else "充分",
                "verification_reason": "原文细节" if positive else "无",
                "expected_gain": "获得摘要未给出的投运条件。" if positive else "没有明确的剩余核验缺口。",
                "basis": "请求目标为已定位的官网原文。" if positive else "Graph 已保留相同时间范围的条件，查询没有新目标。",
                "observed_gain": "有新增证据" if positive else "无新增证据", "result_quote": result_text,
                "unavailable_quote": ""}}
        candidates.append({**finding, "id": f"F{index + 1:03}"})
    found = {"findings": [{k: v for k, v in f.items() if k != "id"} for f in candidates], "limitations": []}
    validate_findings(Findings.model_validate(found), packet)
    summary = {"goal": "核对项目 X 的运营状态及投运条件。", "selected_finding_ids": ["F002", "F001"],
        "process_groups": [
            {"representative_id": "F002", "member_ids": ["F002", "F003"], "reason": "两轮针对相同缺口重复取证，机制与影响方向一致。"},
            {"representative_id": "F001", "member_ids": ["F001"], "reason": "保留独立的原文核验优势，不与无效重复合并。"}],
        "ratings": [
            {"dimension": "A2", "finding_ids": ["F002"], "utilization": "未利用", "effect": "负向", "note": ""},
            {"dimension": "A4", "finding_ids": ["F001"], "utilization": "充分利用", "effect": "正向", "note": ""}],
        "task_assessment": {"result": "正确 / 达标", "rationale": "报告区分了当前运营状态、目标日期与条件，回应了任务要求。",
            "task_quotes": [question], "report_quotes": ["项目 X 尚未正式运营。2030 年为规划目标，实际投运仍需通过安全评估并获得许可。"],
            "limitations": ["此为结构演示，不进行外部事实核验。"]},
        "conclusion": {"benefit": "F001 展示了把摘要缺口转为定向取证的能力。",
            "problem": "优先处理 F002 的证据状态未转化为搜索停止条件的问题。", "evidence": "有较强迹象",
            "improvement": "发起查询前检查当前缺口，并区分已获原文和仍需核验的内容。",
            "verification": "比较同类任务中无增量查询数量，并确认必要核验仍被保留。",
            "comparison": "没有同题对照，不能由此证明总体 Token 节省。"}}
    synthesis = {"question": case["question"], "final_report": case["final_report"], "candidates": candidates,
                 "coverage": case["coverage"], "limitations": []}
    validate_summary(Summary.model_validate(summary), candidates, synthesis)
    return {"label": "结构示例，非真实运行或模型评测", "native": {"question": question, "records": records, "injections": injections, "report": report},
        "case": case, "packets": [packet], "findings": found, "candidates": candidates,
        "summary_input": synthesis, "summary": summary,
        "annotation": annotation(case, summary, candidates, model=ReviewConfig.model),
        "requests": {"findings": messages("findings", packet), "summary": messages("summary", synthesis)}}


def build() -> dict:
    files, sources = {}, {}
    for key, (name, symbol) in SOURCE_TARGETS.items():
        text = (ROOT / name).read_text()
        files[name] = sha(text)
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol)
        sources[key] = {"path": name, "symbol": symbol, "line": node.lineno, "end_line": node.end_lineno,
                        "text": "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])}
    for name in ("tools/ai_review/prompts.py", "tools/ai_review/schema.py", "tools/ai_review/client.py", "tools/case_study_studio/shared.mjs"):
        files[name] = sha((ROOT / name).read_text())
    defaults = inspect.signature(run_review).parameters
    return {"format": "bcg-review-method-contract", "version": 1, "prompt_version": PROMPT_VERSION,
        "source_fingerprint": sha(json.dumps(files, sort_keys=True)), "source_files": files, "sources": sources,
        "model": ReviewConfig().public(),
        "defaults": {key: defaults[key].default for key in ("chunk_chars", "chunk_turns", "max_batches")},
        "dimensions": DIMENSIONS, "schemas": {"findings": Findings.model_json_schema(), "summary": Summary.model_json_schema()},
        "prompts": {"system": SYSTEM, "findings": FIND, "summary": SYNTHESIZE}, "example": illustration()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not DEST.exists() or DEST.read_text() != text:
            raise SystemExit("Pipeline snapshot is stale. Run python -m tools.case_study_studio.build_review_pipeline")
        print("Pipeline snapshot matches runtime sources and validated example.")
    else:
        DEST.write_text(text, encoding="utf-8")
        print(f"Built {DEST.name} ({len(text.encode())} bytes); no model requests.")


if __name__ == "__main__":
    main()
