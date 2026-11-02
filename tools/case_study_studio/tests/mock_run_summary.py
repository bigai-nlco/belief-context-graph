"""Opt-in, no-network Run-summary browser fixture."""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.ai_review.client import ReviewConfig
from tools.case_study_studio.control import Control
from tools.case_study_studio.run_summary import run_summary
from tools.case_study_studio.run_summary_job import execute
from tools.case_study_studio.tests.test_run_summary import mock_client, reply


@contextmanager
def mock_summaries(control: Control) -> Iterator[None]:
    enqueue, settings = control.run_summaries.enqueue, control.settings
    threads: list[threading.Thread] = []
    scheduled: set[int] = set()
    lock = threading.Lock()

    def generate(context: dict, output: Path) -> dict:
        client = mock_client(output, [reply(c) for c in context["cohorts"]])
        try:
            time.sleep(1)
            return run_summary(context, output, client=client)
        finally:
            client.close()

    def worker(job_id: int) -> None:
        time.sleep(.5)
        with control.connect() as db:
            db.execute("UPDATE run_summaries SET status='running',pid=? WHERE id=?", (os.getpid(), job_id))
        execute(control.run_summaries, job_id)

    def submit(*args: Any, **kwargs: Any) -> dict:
        job = enqueue(*args, **kwargs)
        with lock:
            if job["status"] == "queued" and job["id"] not in scheduled:
                scheduled.add(job["id"])
                thread = threading.Thread(target=worker, args=(job["id"],), daemon=True)
                threads.append(thread)
                thread.start()
        return job

    with patch.object(control.run_summaries, "enqueue", side_effect=submit), \
         patch.object(control, "settings", side_effect=lambda: {**settings(), "run_summary_offline_fixture": True}), \
         patch("tools.case_study_studio.run_summary_queue.configuration", return_value=replace(ReviewConfig(), api_key="offline-fake")), \
         patch("tools.case_study_studio.run_summary_job.run_summary", side_effect=generate):
        try:
            yield
        finally:
            for thread in threads:
                thread.join(timeout=15)
