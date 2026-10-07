"""Explicit live-test runner. Never imported or run automatically.

Starts only its own Graph/proxy subprocesses and always tears them down.
Codex gets an isolated home/workspace and a read-only search MCP tool. Dataset
answers and API keys never enter its prompt. No global Codex config is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from bcg.core.env import read_env_file
from bcg.integrations.codex_proxy.config import ProxyConfig
from bcg.integrations.codex_proxy.server import ProxyServer
from bcg.integrations.codex_proxy.verification import verify_bcg_run

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_OPENAI_API_KEY",
        "openai/gpt-5.6-luna",
    ),
    "litellm": (
        "https://litellm.mybigai.ac.cn/v1",
        "LITELLM_OPENAI_API_KEY",
        "gpt-5.6-luna",
    ),
}
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def seed_codex_auth(source: Path, home: Path) -> Path:
    """Use an isolated, private copy; let Codex own token refresh.

    Never symlink the live auth cache: a test must not overwrite the user's
    original credentials. The caller deletes this copy during teardown.
    """
    data = json.loads(source.expanduser().read_text())
    tokens = data.get("tokens") or {}
    if data.get("auth_mode") != "chatgpt" or not all(
        tokens.get(name) for name in ("access_token", "account_id")
    ):
        raise ValueError("The selected auth.json must contain a Codex ChatGPT login")
    destination = home / "auth.json"
    save(destination, data)
    return destination


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(
            data
            if isinstance(data, str)
            else json.dumps(data, ensure_ascii=False, indent=2)
        )


def preflight(route: str, env: dict[str, str], output: Path) -> int:
    base, key_name, model = ROUTES[route]
    key = env[key_name]
    history = [
        {
            "role": "user",
            "content": "Call get_probe_value once, then report its returned value. Do not guess the value.",
        }
    ]
    tools = [
        {
            "type": "function",
            "name": "get_probe_value",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    with httpx.Client(timeout=100, trust_env=False) as client:
        for turn in range(2):
            request = {
                "model": model,
                "input": history,
                "tools": tools,
                "tool_choice": {"type": "function", "name": "get_probe_value"}
                if turn == 0
                else "none",
                "store": False,
                "max_output_tokens": 1024,
                "reasoning": {
                    "effort": "low",
                    "summary": "auto",
                    "context": "current_turn" if turn == 0 else "all_turns",
                },
                "include": ["reasoning.encrypted_content"],
            }
            response = client.post(
                base + "/responses",
                headers={"Authorization": "Bearer " + key},
                json=request,
            )
            try:
                payload = response.json()
            except ValueError:
                payload = {"error": response.text[:1000]}
            save(
                output / f"{route}-turn{turn + 1}.json",
                {
                    "request": request,
                    "http_status": response.status_code,
                    "response": payload,
                },
            )
            print(
                json.dumps(
                    {
                        "route": route,
                        "turn": turn + 1,
                        "http_status": response.status_code,
                        "model": payload.get("model"),
                        "usage": payload.get("usage"),
                        "error": payload.get("error")
                        if not response.is_success
                        else None,
                    }
                ),
                flush=True,
            )
            if not response.is_success:
                return 2
            items = payload.get("output", [])
            print("item types:", [item.get("type") for item in items], flush=True)
            for item in items:
                if item.get("type") == "reasoning":
                    opaque = item.get("encrypted_content") or ""
                    print(
                        json.dumps(
                            {
                                "reasoning_summary_present": bool(item.get("summary")),
                                "encrypted_chars": len(opaque),
                                "encrypted_sha256": hashlib.sha256(
                                    opaque.encode()
                                ).hexdigest(),
                            }
                        ),
                        flush=True,
                    )
            if turn == 0:
                calls = [item for item in items if item.get("type") == "function_call"]
                if not calls:
                    print("Preflight failed: no function call", flush=True)
                    return 3
                history.extend(items)
                history.extend(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "42",
                    }
                    for call in calls
                )
            else:
                text = "\n".join(
                    part.get("text", "")
                    for item in items
                    if item.get("type") == "message"
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                )
                print("preflight final:", text, flush=True)
                if "42" not in text:
                    return 3
    return 0


def wait_ready(url: str, process: subprocess.Popen, timeout: int = 90) -> None:
    until = time.monotonic() + timeout
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < until:
            if process.poll() is not None:
                raise RuntimeError(
                    "Graph server exited before readiness; inspect graph-server.log"
                )
            try:
                if client.get(url + "/health").is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise RuntimeError("Graph readiness timeout")


def run(args, env: dict[str, str]) -> int:
    if args.graph_port == args.proxy_port:
        raise ValueError("Graph and proxy need distinct ports")
    # Refuse to reuse another experiment's service or send data to it.
    for port in (args.graph_port, args.proxy_port):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    chatgpt = args.provider == "codex"
    if args.auth_only and not chatgpt:
        raise ValueError("--auth-only requires --provider codex")
    if chatgpt:
        base, key_name, model = (
            CODEX_BASE_URL,
            "OPENROUTER_OPENAI_API_KEY",
            "gpt-6-luna",
        )
        graph_base, graph_model = "https://openrouter.ai/api/v1", "openai/gpt-6-luna"
    else:
        base, key_name, model = ROUTES[args.provider]
        graph_base, graph_model = base, model
    rows = json.loads(args.dataset.read_text())
    row = next(row for row in rows if row["task_id"] == args.task_id)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "run.json").exists():
        raise ValueError("Use a new output directory for each attempt")
    isolated = Path(tempfile.mkdtemp(prefix="bcg-codex-browsecomp-", dir="/tmp"))
    workspace, home = isolated / "workspace", isolated / "codex-home"
    workspace.mkdir()
    home.mkdir()
    clean_home = isolated / "home"
    clean_home.mkdir()
    inherited = {**os.environ, **env}
    # Graph runs with only the selected existing key; no model substitution.
    if not args.auth_only:
        inherited["BCG_CODEX_GRAPH_KEY"] = env[key_name]
    inherited["BCG_ENV_FILE"] = str(isolated / "absent.env")
    inherited["BCG_HOME"] = str(isolated / "bcg-home")
    inherited["PYTHONPATH"] = str(ROOT)
    inherited["PYTHONUNBUFFERED"] = "1"
    inherited["HF_HUB_OFFLINE"] = "1"
    inherited["TRANSFORMERS_OFFLINE"] = "1"
    inherited["TOKENIZERS_PARALLELISM"] = "false"
    graph_config = {
        "graph-model": {
            "model": graph_model,
            "base_url": graph_base,
            "api_key_env": "BCG_CODEX_GRAPH_KEY",
            "reasoning_effort": "none",
            "max_tokens": 8192,
        },
        "embedding": {
            "provider": "local",
            "model": "/home/yofuria/PLM/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 8,
            "input_prefix": "Document: ",
        },
        "belief_graph": {
            "runtime": {"context_chars": 12000, "extraction_history_turns": 2},
            "edge_generation": {"max_previous_windows": 3},
            "tool_results": {
                "semantic_extraction": True,
                "max_semantic_calls": 12,
                "max_facts": 3,
            },
        },
    }
    save(output / "graph-config.json", graph_config)
    graph_url = f"http://127.0.0.1:{args.graph_port}"
    graph_args = [
        sys.executable,
        "-m",
        "bcg.apps.online_server",
        "unified",
        "--config",
        str(output / "graph-config.json"),
        "--model-key",
        "graph-model",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.graph_port),
        "--context-chars",
        "12000",
        "--output-dir",
        str(output / "graphs"),
    ]
    graph_log = None
    graph = None
    gateway = None
    gateway_thread = None
    codex = None
    auth_copy = None
    try:
        if chatgpt:
            auth_copy = seed_codex_auth(args.codex_auth_file, home)
        if not args.auth_only:
            graph_log = (output / "graph-server.log").open("w")
            graph = subprocess.Popen(
                graph_args,
                cwd=ROOT,
                env=inherited,
                stdout=graph_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(graph_url, graph)
        if not chatgpt:
            os.environ["BCG_CODEX_UPSTREAM_API_KEY"] = env[key_name]
        os.environ["BCG_CODEX_GATEWAY_TOKEN"] = secrets.token_urlsafe(32)
        settings = ProxyConfig(
            upstream_base_url=base,
            upstream_auth="codex_chatgpt" if chatgpt else "api_key",
            graph_base_url=graph_url,
            mode="default" if args.auth_only else "bcg",
            port=args.proxy_port,
            upstream_timeout=300,
            graph_timeout=300,
            audit_dir=str(output / "proxy"),
            replay_reasoning_context=not chatgpt,
            record_model_io=True,
            max_session_requests=3 if args.auth_only else 40,
        )
        gateway = ProxyServer(settings)
        gateway_thread = threading.Thread(
            target=gateway.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        gateway_thread.start()
        session_id = "browsecomp-" + secrets.token_hex(8)
        # Do not inherit this assistant's session identity, unrelated provider
        # credentials or global app integration state in the nested Codex.
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "PATH",
                "LANG",
                "LC_ALL",
                "TERM",
                "TMPDIR",
                "SSL_CERT_FILE",
                "LD_LIBRARY_PATH",
            }
        }
        child_env.update(
            {
                "HOME": str(clean_home),
                "CODEX_HOME": str(home),
                "BCG_CODEX_GATEWAY_TOKEN": os.environ["BCG_CODEX_GATEWAY_TOKEN"],
                "BCG_CODEX_SESSION_ID": session_id,
                "SERPER_API_KEY": env["SERPER_API_KEY"],
                "SERPER_ENDPOINT": env.get(
                    "SERPER_ENDPOINT", "https://google.serper.dev/search"
                ),
                "BCG_SEARCH_MAX_CALLS": "30",
                "PYTHONPATH": str(ROOT),
            }
        )
        provider_auth = (
            "requires_openai_auth = true"
            if chatgpt
            else 'env_key = "BCG_CODEX_GATEWAY_TOKEN"'
        )
        extra_headers = {"X-BCG-Session-ID": "BCG_CODEX_SESSION_ID"}
        if chatgpt:
            extra_headers["X-BCG-Gateway-Token"] = "BCG_CODEX_GATEWAY_TOKEN"
        header_toml = (
            "{ "
            + ", ".join(
                f"{json.dumps(k)} = {json.dumps(v)}" for k, v in extra_headers.items()
            )
            + " }"
        )
        config_text = f"""model = {json.dumps(model)}
model_provider = "bcg_proxy"
model_reasoning_effort = "low"
model_reasoning_summary = "auto"
model_supports_reasoning_summaries = true
cli_auth_credentials_store = "file"
web_search = "disabled"
sandbox_mode = "read-only"
approval_policy = "on-request"

[features]
shell_tool = false
multi_agent = false
enable_request_compression = false
remote_plugin = false
skill_mcp_dependency_install = false

[model_providers.bcg_proxy]
name = "BCG isolated BrowseComp test"
base_url = "http://127.0.0.1:{args.proxy_port}/v1"
wire_api = "responses"
{provider_auth}
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 900000
env_http_headers = {header_toml}

[mcp_servers.search]
command = {json.dumps(sys.executable)}
args = ["-m", "bcg.integrations.codex_proxy.search_mcp"]
env_vars = ["SERPER_API_KEY", "SERPER_ENDPOINT", "BCG_SEARCH_MAX_CALLS", "PYTHONPATH"]
startup_timeout_sec = 20
tool_timeout_sec = 60
required = true
"""
        save(home / "config.toml", config_text)
        save(output / "codex-config.toml", config_text)
        prompt = (
            row["input"]
            + "\n\nResearch this question using the available web_search tool. Verify the identifying clues against sources before giving a concise final answer with source URLs. Do not use local files, stored benchmark answers, or other agents."
        )
        if args.auth_only:
            prompt = "Return exactly BCG_READY. Do not call any tools."
        save(output / "question.txt", prompt)
        run_info = {
            "task_id": "auth-probe" if args.auth_only else args.task_id,
            "auth_only": args.auth_only,
            "model": model,
            "provider": args.provider,
            "agent_reasoning_effort": "low",
            "graph_reasoning_effort": "none",
            "graph_model": graph_model,
            "graph_provider": "openrouter" if chatgpt else args.provider,
            "recent_turns": 2,
            "graph_selection": "focused",
            "node_char_budget": 6600,
            "graph_url": graph_url,
            "proxy_url": f"http://127.0.0.1:{args.proxy_port}",
            "isolated_home": str(home),
            "isolated_workspace": str(workspace),
            "source_code": str(ROOT),
            "started_at": time.time(),
        }
        save(output / "run.json", run_info)
        command = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            str(workspace),
            "--json",
            "--output-last-message",
            str(output / "final-answer.txt"),
            "-",
        ]
        print(json.dumps({"event": "started", **run_info}), flush=True)
        with (
            (output / "codex-events.jsonl").open("w") as stdout,
            (output / "codex-stderr.log").open("w") as stderr,
        ):
            codex = subprocess.Popen(
                command,
                env=child_env,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            codex.communicate(prompt.encode(), timeout=args.timeout)
        result = {
            **run_info,
            "finished_at": time.time(),
            "codex_exit_code": codex.returncode,
            "reference_answer": None if args.auth_only else row["ground_truth_answer"],
        }
        save(output / "result.json", result)
        print(
            json.dumps(
                {
                    "event": "finished",
                    "codex_exit_code": codex.returncode,
                    "output": str(output),
                }
            ),
            flush=True,
        )
        return codex.returncode
    finally:
        if codex and codex.poll() is None:
            os.killpg(codex.pid, signal.SIGTERM)
            try:
                codex.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(codex.pid, signal.SIGKILL)
                codex.wait()
        if gateway:
            gateway.shutdown()
            if gateway_thread:
                gateway_thread.join(timeout=5)
            gateway.server_close()
        if graph and graph.poll() is None:
            os.killpg(graph.pid, signal.SIGTERM)
            try:
                graph.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(graph.pid, signal.SIGKILL)
                graph.wait()
        if graph_log:
            graph_log.close()
        if auth_copy:
            auth_copy.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".bcg/.env")
    parser.add_argument("--provider", choices=[*ROUTES, "codex"], required=True)
    parser.add_argument(
        "--auth-only",
        action="store_true",
        help="Codex login smoke test only; no Graph or benchmark solving",
    )
    parser.add_argument(
        "--codex-auth-file", type=Path, default=Path.home() / ".codex/auth.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/yofuria/Desktop/GraphMemoryEvaluation/bcg/datasets/browse_comp/data.json"
        ),
    )
    parser.add_argument("--task-id", default="browsecomp-0000")
    parser.add_argument("--graph-port", type=int, default=8861)
    parser.add_argument("--proxy-port", type=int, default=8860)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    env = read_env_file(args.env_file)
    if args.preflight_only:
        if args.provider == "codex":
            parser.error("Use --auth-only for the real Codex login preflight")
        return preflight(args.provider, env, args.output)
    code = run(args, env)
    if args.auth_only:
        return code
    report = verify_bcg_run(args.output.resolve())
    save(args.output / "verification.json", report)
    print(json.dumps({"event": "verification", **report}), flush=True)
    return code or (0 if report["passed"] else 4)


if __name__ == "__main__":
    raise SystemExit(main())
