"""Optional official CLM harness adapter for the existing report runner.

The harness runs in its own Python environment. Tools, credentials, recording
and the strict sandbox remain in this repository's report-worker process.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def installation() -> dict:
    root = Path(os.getenv("CLM_ROOT", str(ROOT.parent / "context-language-models"))).expanduser().resolve()
    python = Path(os.getenv("CLM_PYTHON", str(root / ".venv/bin/python"))).expanduser().absolute()
    available = python.is_file() and (root / "clm/clm_harness/clm_agent/harness.py").is_file()
    return {"available": available, "root": str(root), "python": str(python),
            "reason": "" if available else "CLM 未安装，请配置 CLM_ROOT 和 CLM_PYTHON"}


def provenance() -> dict:
    info = installation()
    if not info["available"]:
        raise ValueError(info["reason"])
    root = Path(info["root"])
    command = subprocess.run([info["python"], "-c", "import clm_harness.clm_agent.harness"],
                             env={**os.environ, "PYTHONPATH": str(root / "clm"),
                                  "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
                             capture_output=True, text=True, timeout=30)
    if command.returncode:
        raise ValueError("CLM Python 环境缺少依赖，请在 CLM 环境中安装官方 harness 的依赖")
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    digest = hashlib.sha256()
    for path in sorted((root / "clm/clm_harness").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {**info, "commit": git.stdout.strip() if git.returncode == 0 else None,
            "harness_sha256": digest.hexdigest()}


def settings(profile: dict, max_turns: int | None = None) -> dict:
    llm, agent = profile.get("llm", {}), profile.get("agent", {})
    options = agent.get("clm") or {}
    budget = int(options.get("context_budget_tokens", 28672))
    reserve = int(options.get("context_budget_reserve_tokens", 2048))
    output = int(llm.get("max_tokens", 8192))
    if not 0 <= reserve < budget or budget - reserve < 4096:
        raise ValueError("CLM context budget must leave at least 4096 effective input tokens")
    if budget + output > int(agent.get("max_len", 65536)):
        raise ValueError("CLM context budget plus output cap exceeds the selected model context window")
    cap = int(max_turns or agent.get("main_max_turns", 200))
    if cap < 1:
        raise ValueError("CLM max turns must be positive")
    extra = llm.get("extra_body") or {}
    return dict(model_name="openai/" + str(llm["model"]), max_steps=cap, lm_call_cap=cap,
                temperature=llm.get("temperature"), top_p=llm.get("top_p"), max_tokens=output,
                context_budget_tokens=budget, context_budget_reserve_tokens=reserve,
                nudge_ratios="0.25,0.5,0.75", persistent_nudge_ratio=0.9,
                max_num_retry_on_limit=6, retry_edit_margin_tokens=2048,
                command_timeout=120, observation_max_chars=20000,
                persistent_bash=True, task_template="open_problems", allow_edit_growth="shrink",
                enable_thinking=bool(extra.get("chat_template_kwargs", {}).get("enable_thinking", True)),
                send_chat_template_kwargs=False, emit_ctx_trajectory=False)


async def run_clm(policy, profile: dict, config: dict, trial: Path) -> dict:
    from .bridge import Bridge, save

    info = config.get("clm_runtime") or provenance()
    # The configured installation must still exist when a queued task starts.
    if not Path(info["python"]).is_file():
        raise RuntimeError("CLM Python environment is unavailable")
    out = trial / "clm"
    out.mkdir(parents=True, exist_ok=True)
    effective = settings(profile, config.get("max_turns"))
    if config.get("clm_settings") and effective != config["clm_settings"]:
        raise RuntimeError("CLM settings changed after the run was created")
    save(out / "manifest.json", {"settings": effective, "clm_commit": info.get("commit"),
                                "limits": {"search_calls": config["search_calls"], "fetch_calls": config["fetch_calls"]},
                                "clm_harness_sha256": info.get("harness_sha256"),
                                "reasoning_effort": (profile.get("llm", {}).get("extra_body") or {}).get("reasoning_effort"),
                                "scope": "Official zero-shot CLM harness with existing DRB2 report tools; no BCG graph."})
    async with Bridge(policy, profile["llm"], out, trial) as bridge:
        # No provider/search credentials enter the CLM worker's environment.
        env = {k: v for k, v in os.environ.items() if k in {
            "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "TIKTOKEN_CACHE_DIR"}}
        env.update(OPENAI_API_KEY=bridge.token, OPENAI_BASE_URL=bridge.api_base,
                   PYTHONPATH=str(Path(info["root"]) / "clm"),
                   PYTHONUNBUFFERED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
        with (out / "run.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                info["python"], "-u", str(Path(__file__).with_name("worker.py")),
                "--output", str(out), "--control", str(bridge.control),
                "--api-base", bridge.api_base, stdout=log, stderr=asyncio.subprocess.STDOUT,
                env=env, cwd=info["root"],
            )
            # Inherit the report worker's process group so batch cancellation
            # and per-task timeout also reach the harness (never a detached job).
            loop, parent = asyncio.get_running_loop(), asyncio.current_task()
            old = signal.getsignal(signal.SIGTERM)
            loop.add_signal_handler(signal.SIGTERM, parent.cancel)
            try:
                code = await process.wait()
            finally:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 2)
                    except TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            process.kill()
                        await process.wait()
                loop.remove_signal_handler(signal.SIGTERM)
                signal.signal(signal.SIGTERM, old)
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {}
        report = policy.workspace / "report.md"
        valid = report.resolve().is_relative_to(policy.workspace.resolve()) and report.is_file()
        text = report.read_text() if valid and report.stat().st_size <= 8 * 1024 * 1024 else ""
        if text:
            # A safe, immutable copy for the viewer; never serve the sandbox tree.
            (out / "report.md").write_text(text)
        completed = code == 0 and summary.get("submitted") and bool(text.strip())
        return {"final_answer": text, "stopped_by": "completed" if completed else "aborted",
                "llm_error_reason": None if completed else summary.get("status", "clm_worker_error"),
                "answer_status": "submitted" if completed else "draft"}
