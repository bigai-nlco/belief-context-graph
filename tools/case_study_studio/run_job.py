"""Execute a validated Studio run; survive UI disconnects and stop owned services."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from types import FrameType


def configure_judge(values: Mapping[str, str | None]) -> dict:
    """Keep the existing judge on its own endpoint even with a local Agent."""
    from benchmarks.public.reports.drb2_judge import JudgeConfig

    # Explicit Judge settings win. Studio's fallback is the configured LiteLLM
    # service, not the Agent's OPENAI_* values (which may point to local Qwen).
    os.environ["JUDGE_API_KEY"] = values.get("JUDGE_API_KEY") or values.get("LITELLM_OPENAI_API_KEY") or ""
    os.environ["JUDGE_BASE_URL"] = (values.get("JUDGE_BASE_URL") or values.get("LITELLM_OPENAI_BASE_URL")
                                    or "https://litellm.mybigai.ac.cn/")
    for name in ("JUDGE_MODEL", "DRB2_JUDGE_REASONING_EFFORT"):
        if values.get(name):
            os.environ[name] = str(values[name])
    return JudgeConfig.from_env().public()


def stop_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    rows = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True)
    pairs = [tuple(map(int, line.split())) for line in rows.stdout.splitlines()]
    owned = {process.pid}
    while True:
        added = {pid for pid, parent in pairs if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    for pid in sorted(owned - {process.pid}, reverse=True):
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    for pid in owned - {process.pid}:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    os.chdir(args.root)
    os.umask(0o077)
    import httpx
    import yaml
    from dotenv import dotenv_values

    from tools.case_study_studio.control import Control, now, read_json
    from tools.case_study_studio.run_version import capture_version

    control = Control(args.root, args.state)
    row = control.row(args.run_id)
    config = json.loads(row["config"])
    mode = config.get("mode", "bcg")
    root = control.directory(row)
    graph = runner = None
    interrupted = False
    metadata = {"started_at": now(), "status": "starting", "workers": config["workers"], "mode": mode,
                "studio_run_number": control.label(args.run_id), "requested_ids": config["tasks"],
                "agent": {"model": config["agent_model"]},
                "graph": ({"enabled": True, "model": "gpt-6-luna", "reasoning_effort": "none", "provider": "litellm"}
                          if mode != "default" else {"enabled": False}),
                "exit_codes": {}, "notes": "Report generation status is not an official quality score."}

    def save() -> None:
        tmp = root / "batch.json.tmp"
        tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(root / "batch.json")

    def cancelled() -> bool:
        return interrupted or bool(control.row(args.run_id)["cancel"])

    def signal_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    final_status, error = "failed", None
    try:
        # Capture the executing checkout, not the revision when a queued run was
        # submitted or when its page is viewed. Persist before any model calls.
        metadata["code_version"], metadata["source_sha256"] = capture_version(args.root, mode)
        save()
        if mode not in ("bcg", "bcg_no_retrieval", "default"):
            raise RuntimeError("不支持的 Agent context mode")
        if set(config.get("tasks", {})) != {"drb2"}:
            raise RuntimeError("共享页面仅支持 DeepResearch Bench II，旧批次未启动；已有结果和标注保留")
        values = dotenv_values(Path.home() / ".bcg/.env")
        key = values.get("LITELLM_OPENAI_API_KEY")
        if not key:
            raise RuntimeError("缺少服务端 LITELLM_OPENAI_API_KEY")
        if not values.get("SERPER_API_KEY"):
            raise RuntimeError("缺少服务端 SERPER_API_KEY")
        if config.get("judge_reports"):
            judge_values = {name: value for source in (values, dotenv_values(args.root / ".env"), os.environ)
                            for name, value in source.items() if value}
            metadata["judge"] = {**configure_judge(judge_values), "enabled": True, "workers": config["workers"]}
        local = config["profile"] == "qwen35-9b"
        os.environ.update(
            OPENAI_API_KEY="EMPTY" if local else key,
            OPENAI_MODEL=config["agent_model"],
            OPENAI_BASE_URL="http://127.0.0.1:8001/v1" if local else "https://litellm.mybigai.ac.cn/",
            LITELLM_OPENAI_API_KEY=key,
            SERPER_API_KEY=values.get("SERPER_API_KEY") or "",
            SERPER_BASE_URL=(values.get("SERPER_ENDPOINT") or "https://google.serper.dev/search").rstrip("/").removesuffix("/search"),
            WEB_SEARCH_PROVIDER="serper", SANDBOX_BACKEND="bwrap",
            LLM_FALLBACK_CHAIN="[]", LLM_FALLBACK_MODEL="", LLM_PROVIDER="openai",
            TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
            AGENT_CORE_TRAJECTORY_FORMATS="json,jsonl", PYTHONUNBUFFERED="1",
        )
        os.environ.pop("BCG_GRAPH_API_KEY", None)
        profile = yaml.safe_load((args.root / control.profiles()[config["profile"]]["source"]).read_text())
        profile["agent"]["context_mode"] = mode
        graph_endpoint: tuple[str, int] | None = None
        # Profile files contain references to env keys, never resolved credentials.
        if mode != "default":
            if mode == "bcg_no_retrieval":
                profile["agent"]["bcg"]["retrieval"] = "all"
            graph_profile = yaml.safe_load((args.root / "tools/case_study_studio/profiles/graph.yaml").read_text())
            if config.get("task_state"):
                profile["agent"]["bcg"]["task_state_enabled"] = True
                graph_profile.setdefault("pipeline", {})["task_state"] = {"enabled": True}
            embedding = values.get("BCG_EMBEDDING_MODEL") or str(Path.home() / "PLM/all-MiniLM-L6-v2")
            if Path(embedding).is_dir():
                graph_profile["models"]["embedding"]["model"] = embedding
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            graph_url = f"http://127.0.0.1:{port}"
            graph_endpoint = (graph_url, port)
            profile["agent"]["bcg"]["url"] = graph_url
            os.environ["BCG_GRAPH_URL"] = graph_url
            (root / "graph.yaml").write_text(yaml.safe_dump(graph_profile, sort_keys=False))
            metadata["graph"]["url"] = graph_url
        else:
            profile["agent"].pop("bcg", None)
            os.environ.pop("BCG_GRAPH_URL", None)
        (root / "agent.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
        metadata["config"] = config
        save()

        def command(benchmark: str, dry: bool = False) -> list[str]:
            ids = benchmark + ":" + ",".join(config["tasks"][benchmark])
            cmd = [sys.executable, "-u", "-m", "benchmarks.public.runner.run_reports", "run", benchmark,
                "--task-ids", ids, "--pipeline", "stateful-react-agent", "--profile", str(root / "agent.yaml"),
                "--workers", str(config["workers"]), "--timeout", str(config["timeout"]),
                "--max-turns", str(config["max_turns"]), "--search-calls", str(config["search_calls"]),
                "--fetch-calls", str(config["fetch_calls"]), "--out", str(root / benchmark)]
            return cmd + (["--dry-run"] if dry else [])

        for benchmark in config["tasks"]:
            if cancelled():
                final_status = "cancelled"
                return
            with (root / "preflight.log").open("ab") as stream:
                preflight = subprocess.run(command(benchmark, True), stdout=stream, stderr=subprocess.STDOUT, timeout=120)
            if preflight.returncode:
                raise RuntimeError("任务或数据集校验失败，请检查 task ID 和本机数据文件")
        if local:
            response = httpx.get("http://127.0.0.1:8001/v1/models", timeout=8, trust_env=False)
            response.raise_for_status()
            if not any(m.get("id") == "Qwen3.5-9B" and int(m.get("max_model_len", 0)) >= 49152
                       for m in response.json().get("data", [])):
                raise RuntimeError("本机 Qwen 服务需要提供 Qwen3.5-9B 和至少 49152 context")
        if graph_endpoint is not None:
            graph_url, port = graph_endpoint
            with (root / "graph-server.log").open("ab") as stream:
                graph = subprocess.Popen([sys.executable, "-u", "-m",
                    "bcg.apps.online_server", "unified", "--config", str(root / "graph.yaml"),
                    "--host", "127.0.0.1", "--port", str(port), "--output-dir", str(root / "graphs")],
                    cwd=args.root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(180):
                if cancelled():
                    final_status = "cancelled"
                    return
                if graph.poll() is not None:
                    raise RuntimeError("本次 Graph 服务启动失败")
                try:
                    if httpx.get(graph_url + "/health", timeout=2, trust_env=False).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("本次 Graph 服务启动超时")
        with control.connect() as db:
            db.execute("UPDATE runs SET status='running' WHERE id=?", (args.run_id,))
        metadata["status"] = "running"
        metadata["phase"] = "reports"
        failures = False
        for benchmark in config["tasks"]:
            metadata["current_benchmark"] = benchmark
            save()
            with (root / f"{benchmark}.log").open("ab") as stream:
                runner = subprocess.Popen(command(benchmark), cwd=args.root,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            metadata["runner_pid"] = runner.pid
            save()
            while runner.poll() is None:
                if cancelled():
                    final_status = "cancelled"
                    return
                time.sleep(1)
            metadata["exit_codes"][benchmark] = runner.returncode
            save()
            if runner.returncode:
                raise RuntimeError("任务 runner 异常退出，已有结果已保留")
            counts = read_json(root / benchmark / "summary.json").get("status_counts", {})
            failures |= any(count for status, count in counts.items() if status != "completed")
        if config.get("judge_reports"):
            # Report grading needs no Graph server. Keep only this run's judge
            # process alive, cancellable through the same existing stop path.
            stop_tree(graph)
            if cancelled():
                final_status = "cancelled"
                return
            metadata["phase"] = "judging"
            metadata["judge"]["started_at"] = now()
            save()
            for benchmark in config["tasks"]:
                with (root / f"{benchmark}-judge.log").open("ab") as stream:
                    runner = subprocess.Popen(
                        [sys.executable, "-u", "-m", "benchmarks.public.runner.run_reports", "judge",
                         "--out", str(root / benchmark), "--workers", str(config["workers"])],
                        cwd=args.root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                metadata["runner_pid"] = runner.pid
                save()
                while runner.poll() is None:
                    if cancelled():
                        final_status = "cancelled"
                        return
                    time.sleep(1)
                metadata["exit_codes"][f"{benchmark}_judge"] = runner.returncode
                rubric = read_json(root / benchmark / "summary.json").get("rubric") or {}
                if runner.returncode or not rubric.get("complete"):
                    final_status = "finished_with_judge_errors"
                    error = "报告生成已结束，LLM Judge 未完成；报告已保留，可用 run_reports judge 单独重试评分"
                    return
            metadata["judge"]["finished_at"] = now()
        final_status = "finished_with_task_errors" if failures else "finished"
    except Exception as exc:
        # Known explicit RuntimeErrors are safe; transport exceptions may contain keys.
        if metadata.get("phase") == "judging":
            final_status = "finished_with_judge_errors"
            error = "LLM Judge 执行失败，报告已保留；请查看本次评分日志并单独重试评分"
        else:
            error = str(exc) if type(exc) is RuntimeError else f"启动或执行失败（{type(exc).__name__}），请管理员查看本机日志"
        print(error, flush=True)
    finally:
        for process in (runner, graph):
            try:
                stop_tree(process)
            except (OSError, subprocess.SubprocessError):
                error = "停止本次进程时遇到问题，请管理员检查本机进程"
                final_status = "failed"
        metadata.update(status=final_status, finished_at=now(), graph_stopped=graph is None or graph.poll() is not None)
        save()
        with control.connect() as db:
            db.execute("UPDATE runs SET status=?,finished_at=?,error=? WHERE id=?",
                       (final_status, metadata["finished_at"], error, args.run_id))


if __name__ == "__main__":
    main()
