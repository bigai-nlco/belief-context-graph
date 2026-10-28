"""Opt-in browser fixture. Every model response is generated locally."""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from tools.ai_review.client import LiteLLMReviewer, ReviewConfig
from tools.ai_review.evidence import prepare_case
from tools.ai_review.pipeline import run_review
from tools.case_study_studio.control import Control
from tools.case_study_studio.review_job import execute


@contextmanager
def mock_reviews(control: Control) -> Iterator[None]:
    original = control.ai_reviews.enqueue
    settings = control.settings
    scheduled, lock = set(), threading.Lock()

    def mock_pipeline(source: Path, output: Path, **kwargs: Any) -> dict:
        case = prepare_case(source)
        step = case["steps"][0]
        finding = {"anchor_id": step["id"], "kind": "benefit", "label": "使用已有证据区分两个方案",
            "dimensions": ["A1"], "graph_quotes": [step["graph"]["text"]],
            "action_quote": step["action_text"]["text"], "report_quote": "方案一适合较小样本。",
            "related_ids": [], "expected": "复用已经注入的信息。", "observed": "后续输出与 B1 的内容一致。",
            "impact": "保留小样本适用范围。", "alternative_explanation": "近期消息也可能提供相同信息。",
            "strength": "有较强迹象", "search_assessment": None}
        summary = {"goal": "比较两个研究方案的适用场景", "selected_finding_ids": ["F001"],
            "process_groups": [{"representative_id": "F001", "member_ids": ["F001"], "reason": "保留独立的适用范围证据。"}],
            "task_assessment": {"result": "正确 / 达标", "rationale": "报告分别说明小样本和长期观察场景，回应了比较两个研究方案的要求。",
                "task_quotes": [case["question"]["text"]], "report_quotes": ["方案一适合较小样本。方案二适合长期观察。"],
                "limitations": ["未独立核查报告中的外部事实。"]},
            "ratings": [{"dimension": "A1", "finding_ids": ["F001"], "utilization": "部分利用",
                "effect": "正向", "note": "证据中保留了适用范围。"}],
            "conclusion": {"benefit": "保留适用范围。", "problem": "未发现其他有充分证据的问题。",
                "evidence": "有较强迹象", "improvement": "保留可核查的限制条件。",
                "comparison": "没有同题对照。", "verification": "核对最终报告范围。"}}
        replies = [{"findings": [finding], "limitations": []}, summary]

        def response(_: httpx.Request) -> httpx.Response:
            time.sleep(2)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(replies.pop(0), ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

        client = LiteLLMReviewer(ReviewConfig(api_key="offline-fake"), output / "calls.jsonl",
                                transport=httpx.MockTransport(response))
        try:
            return run_review(source, output, config=client.config, client=client, **kwargs)
        finally:
            client.close()

    def worker(job_id: int) -> None:
        time.sleep(1)
        with control.connect() as db:
            db.execute("UPDATE ai_reviews SET status='running',pid=? WHERE id=?", (os.getpid(), job_id))
        execute(control.ai_reviews, job_id)

    def submit(*args: Any, **kwargs: Any) -> dict:
        job = original(*args, **kwargs)
        with lock:
            if job["status"] == "queued" and job["id"] not in scheduled:
                scheduled.add(job["id"])
                threading.Thread(target=worker, args=(job["id"],), daemon=True).start()
        return job

    with patch.object(control.ai_reviews, "enqueue", side_effect=submit), \
         patch.object(control, "settings", side_effect=lambda: {**settings(), "ai_review_offline_fixture": True}), \
         patch("tools.case_study_studio.review_job.run_review", side_effect=mock_pipeline), \
         patch("tools.case_study_studio.review_queue.ReviewConfig.from_env", return_value=ReviewConfig(api_key="offline-fake")):
        yield
