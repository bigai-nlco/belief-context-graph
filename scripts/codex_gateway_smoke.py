"""Manually driven, one-process gateway TUI smoke test; no benchmark answer input.

Run in a PTY. The tester sends multiple prompts and one native /compact
to the SAME process. Global config/auth are fingerprinted and never modified.
Only the isolated auth copy is supplied to native Codex and removed at exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bcg.core.env import read_env_file
from bcg.integrations.codex.config import private_write
from bcg.integrations.codex.install import install

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gateway"], default="gateway")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--question-file", type=Path)
    p.add_argument("--env-file", type=Path, default=Path.home() / ".bcg/.env")
    p.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    p.add_argument("--graph-model", help="Override only the Graph model")
    p.add_argument("--graph-base-url", help="OpenAI-compatible Graph API base URL")
    p.add_argument("--graph-api-key-env", help="Environment variable name, not a key")
    args = p.parse_args()
    os.umask(0o077)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Choose a new smoke-test output directory")
    workspace = output / "workspace"
    workspace.mkdir(parents=True, mode=0o700)
    home = Path.home()
    auth = home / ".codex/auth.json"
    originals = [auth, home / ".codex/config.toml", home / ".codex/hooks.json"]
    before = {str(path): fingerprint(path) for path in originals}
    env_file = args.env_file.expanduser().resolve()
    config = install(
        workspace,
        mode="gateway",
        env_file=env_file,
        embedding_model=args.embedding_model,
        allow_upload=True,
    )
    env_values = read_env_file(env_file)
    graph_path = Path(config.graph_config)
    graph_config = json.loads(graph_path.read_text())
    graph_settings = graph_config["graph-model"]
    for key, value in (
        ("model", args.graph_model),
        ("base_url", args.graph_base_url),
        ("api_key_env", args.graph_api_key_env),
    ):
        if value is not None:
            graph_settings[key] = value
    if not env_values.get(graph_settings["api_key_env"]):
        raise SystemExit(f"Missing {graph_settings['api_key_env']} in --env-file")
    private_write(graph_path, graph_config)
    with tempfile.TemporaryDirectory(prefix="bcg-gateway-smoke-") as temp:
        temp = Path(temp)
        isolated_home, codex_home = temp / "home", temp / "codex"
        isolated_home.mkdir()
        codex_home.mkdir(mode=0o700)
        auth_data = json.loads(auth.read_text())
        if auth_data.get("auth_mode") != "chatgpt":
            raise SystemExit("Native ChatGPT login required")
        private_write(codex_home / "auth.json", auth_data)
        text = f"""model = "gpt-6-luna"
model_reasoning_effort = "low"
model_reasoning_summary = "auto"
cli_auth_credentials_store = "file"
check_for_update_on_startup = false
disable_paste_burst = true
web_search = "disabled"
sandbox_mode = "read-only"
approval_policy = "on-request"
[features]
shell_tool = false
multi_agent = false
remote_plugin = false
skill_mcp_dependency_install = false
[projects.{json.dumps(str(workspace))}]
trust_level = "trusted"
[mcp_servers.search]
command = {json.dumps(sys.executable)}
args = ["-m", "bcg.integrations.codex_proxy.search_mcp"]
env_vars = ["SERPER_API_KEY", "SERPER_ENDPOINT", "BCG_SEARCH_MAX_CALLS", "PYTHONPATH"]
startup_timeout_sec = 20
tool_timeout_sec = 60
"""
        private_write(codex_home / "config.toml", text)
        env = {
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
        env.update(
            HOME=str(isolated_home),
            CODEX_HOME=str(codex_home),
            PYTHONPATH=str(ROOT),
            SERPER_API_KEY=env_values["SERPER_API_KEY"],
            SERPER_ENDPOINT=env_values.get(
                "SERPER_ENDPOINT", "https://google.serper.dev/search"
            ),
            BCG_SEARCH_MAX_CALLS="12",
            TOKENIZERS_PARALLELISM="false",
        )
        codex_args = [
            "--no-alt-screen",
            "--cd",
            str(workspace),
        ]
        if args.question_file:
            question = args.question_file.read_text()
            private_write(output / "question.txt", question)
            codex_args.append(question)
        command = [
            sys.executable,
            "-m",
            "bcg.integrations.codex.cli",
            "--config",
            str(workspace / ".bcg-codex/config.json"),
            "--record-model-io",
            "--",
            *codex_args,
        ]
        private_write(
            output / "run.json",
            {
                "mode": args.mode,
                "agent_model": "gpt-6-luna",
                "agent_effort": "low",
                "graph_model": graph_settings["model"],
                "graph_effort": graph_settings["reasoning_effort"],
                "graph_base_url": graph_settings["base_url"],
                "graph_api_key_env": graph_settings["api_key_env"],
                "env_file": str(env_file),
                "workspace": str(workspace),
                "codex_home": str(codex_home),
                "started": time.time(),
                "auth_file_is_private_copy": True,
            },
        )
        print(f"Native smoke workspace: {workspace}", flush=True)
        code = subprocess.call(command, env=env, cwd=workspace)
        # Preserve only native history needed to inspect context and compaction.
        # Never copy auth, settings DB or credential files.
        import shutil

        for name in ("sessions", "archived_sessions"):
            source = codex_home / name
            if source.exists():
                shutil.copytree(source, output / name)
        after = {str(path): fingerprint(path) for path in originals}
        private_write(
            output / "exit.json",
            {
                "exit_code": code,
                "global_config_and_auth_unchanged": before == after,
                "finished": time.time(),
            },
        )
        print(
            f"Native Codex exited {code}; global config/auth unchanged={before == after}"
        )
        # The current runtime can use a paginated history store. Copying its
        # complete DB could copy unrelated internal account state, so do not.
        assert before == after


if __name__ == "__main__":
    main()
