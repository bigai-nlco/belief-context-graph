"""Regressions for verification-vs-duplication and non-repetitive synthesis."""
from copy import deepcopy

import pytest
from test_ai_review import findings, summary
from test_ai_review import trial as trial_fixture

from tools.ai_review.evidence import batches, prepare_case
from tools.ai_review.pipeline import annotation, validate_findings, validate_summary
from tools.ai_review.schema import Findings, Summary


@pytest.fixture(name="trial")
def review_trial(tmp_path):
    return trial_fixture.__wrapped__(tmp_path)


def search_case(trial, *, verdict="无效重复"):
    packet = batches(prepare_case(trial))[0]
    value = findings(packet)
    f = value["findings"][0]
    f.update(dimensions=["A2"], kind="missed_opportunity")
    f["search_assessment"] = {"verdict": verdict, "information_need": "核对当前运营状态。",
        "graph_sufficiency": "充分", "verification_reason": "无", "expected_gain": "没有可识别的剩余信息缺口。",
        "basis": "同一时间范围的运营状态已在 Graph 中给出，结果没有新增信息。",
        "observed_gain": "无新增证据", "result_quote": "官网说明还在规划。", "unavailable_quote": ""}
    return packet, value


def test_grounded_redundancy_is_accepted(trial):
    packet, value = search_case(trial)
    validate_findings(Findings.model_validate(value), packet)


@pytest.mark.parametrize("change", [
    {"graph_sufficiency": "部分"}, {"verification_reason": "原文细节"}, {"observed_gain": "有新增证据"},
    {"observed_gain": "不可见"}, {"result_quote": ""}, {"result_quote": "虚构的工具返回原文"}, {"basis": " "},
])
def test_repeated_topic_without_full_evidence_cannot_be_negative(trial, change):
    packet, value = search_case(trial)
    value["findings"][0]["search_assessment"].update(change)
    with pytest.raises(ValueError):
        validate_findings(Findings.model_validate(value), packet)


def test_truncated_results_cannot_establish_no_gain(trial):
    packet, value = search_case(trial)
    packet["steps"][0]["tool_results"]["truncated"] = True
    with pytest.raises(ValueError, match="complete results"):
        validate_findings(Findings.model_validate(value), packet)


def test_failed_but_targeted_original_source_verification_is_not_redundancy(trial):
    packet, value = search_case(trial, verdict="合理核验")
    f = value["findings"][0]
    f.update(kind="benefit", dimensions=["A4"])
    f["search_assessment"].update(graph_sufficiency="部分", verification_reason="原文细节",
        expected_gain="补齐运营条件的原文范围。", basis="Graph 只有摘要，抓取已定位的原文。")
    # No new result does not turn a reasonable action into a redundancy failure.
    validate_findings(Findings.model_validate(value), packet)
    f["search_assessment"]["verification_reason"] = "无"
    with pytest.raises(ValueError, match="specific verification"):
        validate_findings(Findings.model_validate(value), packet)


@pytest.mark.parametrize("verdict", ["合理核验", "新增信息搜索", "证据不足"])
def test_search_rubric_cannot_be_overridden_by_negative_a2_labels(trial, verdict):
    packet, value = search_case(trial, verdict=verdict)
    value["findings"][0]["search_assessment"]["verification_reason"] = "原文细节"
    with pytest.raises(ValueError, match="negative A2"):
        validate_findings(Findings.model_validate(value), packet)


def test_a2_needs_search_rubric(trial):
    packet, value = search_case(trial)
    value["findings"][0]["search_assessment"] = None
    with pytest.raises(ValueError, match="requires an explicit"):
        validate_findings(Findings.model_validate(value), packet)


def test_first_tool_failure_is_not_a_known_unavailable_retry(trial):
    packet, value = search_case(trial, verdict="无效工具重试")
    quote = "Search budget exhausted. Use the evidence already collected."
    packet["steps"][0]["tool_results"]["text"] = quote
    value["findings"][0]["search_assessment"].update(result_quote=quote, unavailable_quote=quote)
    with pytest.raises(ValueError, match="before the current"):
        validate_findings(Findings.model_validate(value), packet)
    packet["steps"][0]["previous_calls"] = [{"action_text": {"text": "search"}, "tool_results": {"text": quote}}]
    validate_findings(Findings.model_validate(value), packet)


def group_case(trial):
    case = prepare_case(trial)
    f = findings(batches(case)[0])["findings"][0]
    pool = [{**deepcopy(f), "id": "F001"}, {**deepcopy(f), "id": "F002"}]
    pool[1]["dimensions"] = ["A6"]
    s = summary()
    s["process_groups"][0]["member_ids"].append("F002")
    s["ratings"].append({"dimension": "A6", "finding_ids": ["F001"], "utilization": "部分利用", "effect": "正向", "note": ""})
    return case, pool, s


def test_one_process_can_support_multiple_dimensions_without_duplicate_events(trial):
    case, pool, s = group_case(trial)
    validate_summary(Summary.model_validate(s), pool, case)
    s["conclusion"]["benefit"] = "F001 保留了证据的适用范围。"
    result = annotation(case, s, pool, model="gpt-6.1-sol")
    assert len(result["events"]) == 1
    assert result["events"][0]["dimensions"] == ["A1", "A6"]
    assert result["dimensions"]["A6"]["eventIds"] == ["F001"]
    assert result["conclusion"]["benefit"] == "E01 保留了证据的适用范围。"


@pytest.mark.parametrize("mutation", ["different_kind", "different_verdict", "duplicate_member", "unknown_member", "missing_representative", "wrong_order"])
def test_bad_groups_are_rejected(trial, mutation):
    case, pool, s = group_case(trial)
    if mutation == "different_kind":
        pool[1]["kind"] = "misuse"
    elif mutation == "different_verdict":
        pool[1]["search_assessment"] = {"verdict": "合理核验"}
    elif mutation == "duplicate_member":
        s["process_groups"][0]["member_ids"].append("F002")
    elif mutation == "unknown_member":
        s["process_groups"][0]["member_ids"].append("F999")
    elif mutation == "missing_representative":
        s["process_groups"][0]["member_ids"] = ["F002"]
    else:
        s["process_groups"][0]["representative_id"] = "F002"
    with pytest.raises(ValueError):
        validate_summary(Summary.model_validate(s), pool, case)


def test_grouped_occurrences_keep_exact_per_turn_evidence(trial):
    case, pool, s = group_case(trial)
    second = deepcopy(case["steps"][0])
    second.update(id="second", call=2, source={**second["source"], "call": 2, "index": 4})
    second["graph"]["text"] = "B13: 第二轮实际注入的内容。"
    second["action_text"]["text"] = "第二轮实际输出。"
    case["steps"].append(second)
    pool[1]["anchor_id"] = "second"
    result = annotation(case, s, pool, model="gpt-6.1-sol")
    occurrence = result["events"][0]["relatedOccurrences"][0]
    assert occurrence["call"] == 2 and occurrence["sourceLocator"]["index"] == 4
    assert occurrence["quote"] == second["action_text"]["text"]
    assert occurrence["graphEvidence"]["text"] == second["graph"]["text"]


def test_summary_cannot_reverse_reasonable_search_into_a2_misuse(trial):
    case, pool, s = group_case(trial)
    for f in pool:
        f.update(dimensions=["A2"], search_assessment={"verdict": "合理核验"})
    s["ratings"] = [{"dimension": "A2", "finding_ids": ["F001"], "utilization": "未利用", "effect": "负向", "note": ""}]
    with pytest.raises(ValueError, match="contradicts"):
        validate_summary(Summary.model_validate(s), pool, case)


def test_summary_references_are_visible_representatives_and_notes_are_brief(trial):
    case, pool, s = group_case(trial)
    s["conclusion"]["problem"] = "F002 存在问题。"
    with pytest.raises(ValueError, match="representative"):
        validate_summary(Summary.model_validate(s), pool, case)
    s["conclusion"]["problem"] = "没有充分问题证据。"
    s["ratings"][0]["note"] = "重复叙述" * 100
    with pytest.raises(ValueError):
        Summary.model_validate(s)


@pytest.mark.parametrize("target", ["rating", "conclusion"])
def test_summary_cannot_copy_full_event_analysis(trial, target):
    case, pool, s = group_case(trial)
    detail = "Agent 已看到项目处于规划阶段的证据，却将计划中的运营日期改写成正式运营日期，导致最终报告丢失了当前尚未投入运营这一关键限定。"
    pool[0]["observed"] = detail
    if target == "rating":
        s["ratings"][0]["note"] = detail
    else:
        s["conclusion"]["problem"] = detail
    with pytest.raises(ValueError, match="not copy"):
        validate_summary(Summary.model_validate(s), pool, case)
    s["ratings"][0]["note"] = ""
    s["conclusion"]["problem"] = "F001 暴露了由证据转写到报告时的范围丢失。"
    validate_summary(Summary.model_validate(s), pool, case)


def test_one_candidate_cannot_appear_in_multiple_groups(trial):
    case, pool, s = group_case(trial)
    s["selected_finding_ids"].append("F002")
    s["process_groups"].append({"representative_id": "F002", "member_ids": ["F002"], "reason": "重复选择。"})
    with pytest.raises(ValueError, match="one group"):
        validate_summary(Summary.model_validate(s), pool, case)
