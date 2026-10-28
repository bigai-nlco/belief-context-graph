"""Offline tests for the standalone AI Review workflow; no real API calls."""
from __future__ import annotations

import gzip
import json

import httpx
import pytest

from tools.ai_review.client import LiteLLMReviewer, ReviewConfig, ReviewError
from tools.ai_review.evidence import batches, prepare_case, sha
from tools.ai_review.pipeline import run_review, validate_findings, validate_summary
from tools.ai_review.prompts import PROMPT_VERSION, messages
from tools.ai_review.schema import Findings, Summary

GRAPH = "B12: 计划于 2030 年抵达，当前尚未进入运营阶段。confidence=0.82"
ACTION = "已有 Graph 中的 B12 表明仍是规划阶段，我将区分计划与实际运营。"
REPORT = "# 最终报告\n\n该项目仍处于规划阶段，不能将规划日期写成正式运营日期。"


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
    return path


@pytest.fixture
def trial(tmp_path):
    root = tmp_path / "trial"
    write(root, "result.json", {"task_id": "2", "benchmark": "drb2", "status": "completed"})
    write(root, "report.md", REPORT)
    records = [{"t": "start", "user_message": "确认项目是否正式运营。"},
               {"t": "llm", "turn": 1, "content": ACTION,
                "tool_calls": [{"name": "web_search", "args": {"q": "运营状态", "api_key": "not-for-model"}}]},
               {"t": "result", "turn": 1, "name": "web_search", "result": "官网说明还在规划。"}]
    write(root, "agent/trajectories/react_agent.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write(root, "agent/bcg/requests.jsonl", json.dumps({"call": 1, "graph_chars": len(GRAPH),
          "system_memory": "<context_blocks_guide>ignore this</context_blocks_guide>\n<belief_graph>" + GRAPH + "</belief_graph>"}) + "\n")
    return root


def findings(packet):
    return {"findings": [{"anchor_id": packet["steps"][0]["id"], "kind": "benefit", "label": "保留规划与运营的区别",
        "dimensions": ["A1", "A6"], "graph_quotes": [GRAPH], "action_quote": ACTION, "report_quote": "该项目仍处于规划阶段",
        "related_ids": [], "expected": "复用规划状态并保留时间限制。", "observed": "Assistant 明确引用 B12，最终报告保留限制。",
        "impact": "避免把规划写成已发生事实。", "alternative_explanation": "任务中也提示了运营核查，不能全部归因于 Graph。",
        "strength": "明确可见", "search_assessment": None}], "limitations": []}


def summary():
    return {"goal": "确认项目是否正式运营。", "selected_finding_ids": ["F001"], "ratings": [
        {"dimension": "A1", "finding_ids": ["F001"], "utilization": "充分利用", "effect": "正向", "note": "明确引用 B12。"}],
        "process_groups": [{"representative_id": "F001", "member_ids": ["F001"], "reason": "独立的适用范围保留过程。"}],
        "task_assessment": {"result": "正确 / 达标", "rationale": "任务要求确认是否正式运营，报告明确回答仍处于规划阶段，并区分规划与实际运营。",
            "task_quotes": ["确认项目是否正式运营。"], "report_quotes": ["该项目仍处于规划阶段，不能将规划日期写成正式运营日期。"],
            "limitations": ["未独立核查项目官方网站。"]},
        "conclusion": {"benefit": "使用已注入的规划状态。", "problem": "未发现有证据支持的失效。", "evidence": "明确可见",
        "improvement": "在后续检索中保留时间范围。", "comparison": "没有同题对照，不评价 Token 节省。",
        "verification": "核对后续报告是否持续保留时间限制。"}}


def reviewer(output, replies, seen=None):
    def handler(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append((request, body))
        reply = replies.pop(0)
        return httpx.Response(200, json={"id": "fake", "model": "gpt-6.1-sol", "choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(reply, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                      "completion_tokens_details": {"reasoning_tokens": 20}}})
    return LiteLLMReviewer(ReviewConfig(api_key="test-review-secret"), output / "calls.jsonl", transport=httpx.MockTransport(handler))


def test_evidence_pairs_exact_call_and_does_not_use_final_graph(trial):
    write(trial, "agent/bcg/state.json", {"snapshot": {"nodes": [{"belief": "Future information"}]}})
    case = prepare_case(trial)
    step = case["steps"][0]
    assert step["graph"]["text"] == GRAPH
    assert step["graph"]["status"] == "matched"
    assert step["source"]["index"] == 2 and step["call"] == 1
    assert "not-for-model" not in json.dumps(case)
    assert "Future information" not in json.dumps(case)
    assert "context_blocks_guide" not in json.dumps(case)
    assert step["id"] == prepare_case(trial)["steps"][0]["id"]


def test_missing_or_ambiguous_graph_cannot_support_finding(trial):
    path = trial / "agent/bcg/requests.jsonl"
    original = path.read_text()
    path.write_text(original.replace('"call": 1', '"call": 2'))
    packet = batches(prepare_case(trial))[0]
    assert packet["steps"][0]["graph"]["status"] == "missing"
    with pytest.raises(ValueError, match="injection"):
        validate_findings(Findings.model_validate(findings(packet)), packet)
    path.write_text(original + original.replace(GRAPH, "wrong Graph"))
    # JSON escapes CJK in the stored input; change a structural value instead.
    second = json.loads(original)
    second["system_memory"] = "<belief_graph>Different node</belief_graph>"
    path.write_text(original + json.dumps(second) + "\n")
    assert prepare_case(trial)["steps"][0]["graph"]["status"] == "ambiguous"


@pytest.mark.parametrize("field,value", [
    ("graph_quotes", ["invented node quotation"]), ("action_quote", "invented Assistant action"),
    ("report_quote", "invented report sentence"), ("anchor_id", "future-step"),
    ("related_ids", ["future-step"]), ("dimensions", ["A7"]), ("dimensions", ["A8"]),
])
def test_bad_citations_and_unsupported_dimensions_rejected(trial, field, value):
    packet = batches(prepare_case(trial))[0]
    data = findings(packet)
    data["findings"][0][field] = value
    with pytest.raises(ValueError):
        validate_findings(Findings.model_validate(data), packet)


def test_summary_cannot_create_or_mislink_findings(trial):
    packet = prepare_case(trial)
    pool = [{**findings(batches(packet)[0])["findings"][0], "id": "F001"}]
    value = summary()
    validate_summary(Summary.model_validate(value), pool, packet)
    value["ratings"][0]["dimension"] = "A2"
    with pytest.raises(ValueError, match="supporting"):
        validate_summary(Summary.model_validate(value), pool, packet)


def test_complete_flow_and_cache_do_not_repeat_api_calls(trial, tmp_path):
    output = tmp_path / "review"
    packet = batches(prepare_case(trial))[0]
    seen = []
    client = reviewer(output, [findings(packet), summary()], seen)
    try:
        result = run_review(trial, output, config=client.config, client=client)
        assert result["annotation"]["status"] == "draft"
        assert result["annotation"]["reviewer"] == client.config.model == "gpt-6.1-sol"
        assert result["annotation"]["result"] == "正确 / 达标"
        assert result["annotation"]["resultAssessment"] == {"source": "ai", **summary()["task_assessment"]}
        assert result["annotation"]["events"][0]["sourceLocator"]["index"] == 2
        assert result["usage"]["total_tokens"] == 280
        assert result["usage"]["output_tokens"] == 80
        assert result["usage"]["reasoning_tokens"] == 40  # subset, not added twice
        assert not result["usage"]["cost_complete"]
        assert "test-review-secret" not in (output / "calls.jsonl").read_text()
        assert "gpt-6.1-sol" in (output / "manifest.json").read_text()
        assert (output / "review.md").is_file()
        report = (output / "review.md").read_text()
        assert "## 任务结果 · AI 评估" in report and summary()["task_assessment"]["rationale"] in report
        assert "> 确认项目是否正式运营。" in report and "判断范围与待核实内容" in report
        assert run_review(trial, output, config=client.config, client=client) == result
        assert len(seen) == 2
        assert "## 任务结果评估" in seen[1][1]["messages"][1]["content"]
        for request, body in seen:
            assert request.url.path == "/chat/completions"
            assert body["model"] == "gpt-6.1-sol" and body["reasoning_effort"] == "medium"
            assert "tools" not in body and body["stream"] is False
    finally:
        client.close()


@pytest.mark.parametrize("field,value", [
    ("result", "官方评分通过"), ("rationale", " "),
    ("task_quotes", ["Invented task"]), ("report_quotes", [ACTION]),
    ("task_quotes", []), ("report_quotes", []), ("report_quotes", [" "]),
    ("limitations", [" "]),
])
def test_task_assessment_requires_valid_result_and_exact_evidence(trial, field, value):
    packet = prepare_case(trial)
    pool = [{**findings(batches(packet)[0])["findings"][0], "id": "F001"}]
    data = summary()
    data["task_assessment"][field] = value
    with pytest.raises(ValueError):
        validate_summary(Summary.model_validate(data), pool, packet)


@pytest.mark.parametrize("source", ["question", "final_report"])
def test_truncated_evidence_cannot_claim_full_compliance(trial, source):
    packet = prepare_case(trial)
    packet[source]["truncated"] = True
    pool = [{**findings(batches(packet)[0])["findings"][0], "id": "F001"}]
    data = summary()
    with pytest.raises(ValueError, match="full task compliance"):
        validate_summary(Summary.model_validate(data), pool, packet)
    data["task_assessment"].update(result="尚未评判", rationale="任务或报告被截断，无法对全部要求做整体判断。", limitations=[])
    with pytest.raises(ValueError, match="truncated evidence"):
        validate_summary(Summary.model_validate(data), pool, packet)
    data["task_assessment"]["limitations"] = ["只提供了部分任务或报告内容。"]
    validate_summary(Summary.model_validate(data), pool, packet)


@pytest.mark.parametrize("verdict", ["正确 / 达标", "部分正确 / 部分达标", "错误 / 未达标", "未完成", "尚未评判"])
def test_task_result_is_assessed_even_without_graph_findings(trial, tmp_path, verdict):
    output = tmp_path / "review"
    data = summary()
    data.update(selected_finding_ids=[], process_groups=[], ratings=[])
    data["task_assessment"].update(result=verdict, rationale="对照任务要求与报告原文给出的模拟判断依据。")
    client = reviewer(output, [{"findings": [], "limitations": ["没有足够的 Graph 使用证据。"]}, data])
    try:
        result = run_review(trial, output, config=client.config, client=client)
        assert result["annotation"]["result"] == verdict
        assert result["annotation"]["resultAssessment"]["rationale"] == data["task_assessment"]["rationale"]
        assert result["annotation"]["events"] == [] and result["usage"]["calls"] == 2
    finally:
        client.close()


def test_summary_with_missing_assessment_is_repaired_not_silently_defaulted(trial, tmp_path):
    output = tmp_path / "review"
    missing = summary()
    del missing["task_assessment"]
    client = reviewer(output, [findings(batches(prepare_case(trial))[0]), missing, summary()])
    try:
        result = run_review(trial, output, config=client.config, client=client)
        assert result["annotation"]["result"] == "正确 / 达标" and result["usage"]["calls"] == 3
    finally:
        client.close()


def set_legacy_manifest(trial, output, config, version="bcg-agent-use-review-v1"):
    state = json.loads((output / "manifest.json").read_text())
    state["prompt_version"] = version
    state["fingerprint"] = sha(json.dumps([prepare_case(trial)["evidence_sha256"], config.public(),
        state["prompt_version"], 60000, 6], sort_keys=True))
    write(output, "manifest.json", state)


@pytest.mark.parametrize("version", ["bcg-agent-use-review-v1", "bcg-agent-use-review-v2"])
def test_completed_legacy_review_is_preserved_without_automatic_billing(trial, tmp_path, version):
    output = tmp_path / "review"
    client = reviewer(output, [findings(batches(prepare_case(trial))[0]), summary()])
    try:
        result = run_review(trial, output, config=client.config, client=client)
        result["prompt_version"] = version
        result["annotation"]["result"] = "尚未评判"
        del result["annotation"]["resultAssessment"]
        write(output, "review.json", result)
        set_legacy_manifest(trial, output, client.config, version)
        before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
        assert run_review(trial, output, config=client.config, client=client, retry_failed=True) == result
        assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before
    finally:
        client.close()


def test_explicit_legacy_retry_archives_old_findings_before_using_new_rules(trial, tmp_path):
    output = tmp_path / "review"
    first = reviewer(output, [findings(batches(prepare_case(trial))[0]), {}, {}])
    try:
        with pytest.raises(ReviewError):
            run_review(trial, output, config=first.config, client=first)
        set_legacy_manifest(trial, output, first.config)
    finally:
        first.close()
    with pytest.raises(ReviewError, match="--retry"):
        run_review(trial, output, config=first.config)
    seen = []
    second = reviewer(output, [findings(batches(prepare_case(trial))[0]), summary()], seen)
    try:
        result = run_review(trial, output, config=second.config, client=second, retry_failed=True)
        assert len(seen) == 2 and result["usage"]["calls"] == 5
        assert (output / "previous-versions/bcg-agent-use-review-v1/findings-001.json").is_file()
        assert result["prompt_version"] == PROMPT_VERSION
        assert result["annotation"]["result"] == "正确 / 达标"
    finally:
        second.close()


def test_invalid_output_is_repaired_once(trial, tmp_path):
    output = tmp_path / "review"
    packet = batches(prepare_case(trial))[0]
    client = reviewer(output, [{"wrong": True}, findings(packet), summary()])
    try:
        assert run_review(trial, output, config=client.config, client=client)["usage"]["calls"] == 3
    finally:
        client.close()


def test_failed_output_is_not_published_or_automatically_retried(trial, tmp_path):
    output = tmp_path / "review"
    client = reviewer(output, [{"wrong": True}, {"still_wrong": True}])
    try:
        with pytest.raises(ReviewError, match="校验"):
            run_review(trial, output, config=client.config, client=client)
        assert not (output / "review.json").exists()
        assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
        with pytest.raises(ReviewError, match="--retry"):
            run_review(trial, output, config=client.config, client=client)
    finally:
        client.close()


def test_dry_run_needs_no_key_and_never_calls_api(trial, tmp_path):
    out = tmp_path / "review"
    result = run_review(trial, out, config=ReviewConfig(), dry_run=True)
    assert result["status"] == "prepared"
    assert not (out / "calls.jsonl").exists()
    assert (out / "prompt-preview.json").is_file()
    assert "任务结果评估" in (out / "summary-prompt-preview.json").read_text()


def test_failed_summary_resume_reuses_valid_findings(trial, tmp_path):
    out = tmp_path / "review"
    packet = batches(prepare_case(trial))[0]
    first = reviewer(out, [findings(packet), {}, {}])
    try:
        with pytest.raises(ReviewError):
            run_review(trial, out, config=first.config, client=first)
    finally:
        first.close()
    second = reviewer(out, [summary()])
    try:
        result = run_review(trial, out, config=second.config, client=second, retry_failed=True)
        assert result["usage"]["calls"] == 4
        assert result["annotation"]["events"]
    finally:
        second.close()


def test_evidence_changes_require_new_output_directory(trial, tmp_path):
    out = tmp_path / "review"
    run_review(trial, out, dry_run=True, config=ReviewConfig())
    (trial / "report.md").write_text(REPORT + "Changed.")
    with pytest.raises(ReviewError, match="新目录"):
        run_review(trial, out, dry_run=True, config=ReviewConfig())


def test_single_case_bundle_and_digest_validation(trial, tmp_path):
    files = [{"path": p.relative_to(trial).as_posix(), "text": p.read_text(),
              "digest": sha(p.read_text()), "size": len(p.read_bytes())} for p in sorted(trial.rglob("*")) if p.is_file()]
    case = {"files": files}
    bundle = {"format": "bcg-case-bundle", "version": 1, "cases": [case]}
    target = tmp_path / "case.json.gz"
    target.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    assert prepare_case(target)["steps"][0]["graph"]["text"] == GRAPH
    files[0]["digest"] = "invalid"
    target.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    with pytest.raises(ValueError, match="digest"):
        prepare_case(target)


def test_config_uses_only_litellm_key(tmp_path, monkeypatch):
    for key in ("LITELLM_OPENAI_API_KEY", "LITELLM_OPENAI_BASE_URL", "LITELLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-agent-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.invalid")
    env = write(tmp_path, "private.env", "LITELLM_OPENAI_API_KEY=review-key\n")
    config = ReviewConfig.from_env(env)
    assert config.api_key == "review-key"
    assert config.base_url == "https://litellm.mybigai.ac.cn/"
    assert "review-key" not in repr(config) and "api_key" not in config.public()


def test_prompts_focus_on_agent_use_and_no_hidden_reasoning_assumption(trial):
    prompt = messages("findings", batches(prepare_case(trial))[0])
    assert "不是评判抽点" in prompt[0]["content"]
    assert "previous_calls" in prompt[1]["content"]
    assert "不能把" in prompt[1]["content"]


def test_http_error_does_not_echo_credentials(tmp_path):
    secret = "test-secret-do-not-log"
    transport = httpx.MockTransport(lambda _: httpx.Response(401, text="invalid key " + secret))
    client = LiteLLMReviewer(ReviewConfig(api_key=secret), tmp_path / "calls.jsonl", transport=transport)
    try:
        with pytest.raises(ReviewError, match="401") as err:
            client.complete([{"role": "user", "content": "test"}], "test")
        assert secret not in str(err.value)
        assert secret not in (tmp_path / "calls.jsonl").read_text()
    finally:
        client.close()
