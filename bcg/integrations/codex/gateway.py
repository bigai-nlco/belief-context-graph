"""Optional strong context intervention, around ONE native interactive Codex."""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
from pathlib import Path

from bcg.integrations.codex_proxy.config import ProxyConfig
from bcg.integrations.codex_proxy.graph import GraphError
from bcg.integrations.codex_proxy.server import ProxyServer

from .config import Config


class SdkGraph:
    def __init__(self, config: Config):
        self.config = config
        self.manager = None
        self._manager_lock = threading.Lock()

    def _get(self):
        if self.manager is None:
            # Independent Codex threads share this adapter. Publish one fully
            # initialized manager so all writes and selections use its sessions.
            # A failed initialization leaves None and releases the lock for retry.
            with self._manager_lock:
                if self.manager is None:
                    from .runtime import create_graph_manager

                    self.manager = create_graph_manager(self.config)
        return self.manager

    def push(self, problem_id: str, turns: list[dict]) -> dict:
        try:
            manager = self._get()
            result = manager.push_many(
                [
                    {
                        **t,
                        "problem_id": problem_id,
                        "is_message_end": True,
                        "is_trajectory_end": False,
                    }
                    for t in turns
                ]
            )
            return result["latest"][problem_id]
        except Exception as exc:
            raise GraphError("native_graph_write_failed") from exc

    def select(self, problem_id: str, query: str, question: str, budget: int) -> dict:
        try:
            session = self._get().get_session(problem_id, create=False)
            if session is None:
                raise ValueError("missing Graph session")
            return session.select_context(
                query, strategy="focused", question=question, node_char_budget=budget
            )
        except Exception as exc:
            raise GraphError("native_graph_selection_failed") from exc


def run(
    config: Config,
    arguments: list[str],
    *,
    codex: str = "codex",
    record_io: bool = False,
) -> int:
    if not config.enabled:
        raise ValueError("Enable BCG before running the gateway")
    # An enabled=false override alone creates an invalid MCP entry when the
    # gateway is used without a legacy memory MCP installation.
    has_memory_mcp = (
        subprocess.run(
            [codex, "mcp", "get", "bcg_memory", "--json"],
            cwd=config.project,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        ).returncode
        == 0
    )
    state = Path(config.state_dir)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Ephemeral config and random local token: no global config/auth writes.
    token = secrets.token_urlsafe(32)
    os.environ["BCG_CODEX_GATEWAY_TOKEN"] = token
    os.environ["BCG_ENV_FILE"] = config.env_file
    settings = ProxyConfig(
        upstream_base_url="https://chatgpt.com/backend-api/codex",
        upstream_auth="codex_chatgpt",
        port=0,
        graph_base_url="http://127.0.0.1:1",  # SDK replaces the HTTP adapter.
        recent_turns=2,
        node_char_budget=6600,
        audit_dir=str(state / "gateway"),
        record_model_io=record_io,
        allow_native_compaction=True,
    )
    server = ProxyServer(settings)
    server.engine.graph = SdkGraph(config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    env = dict(os.environ)
    env.update(
        BCG_CODEX_GATEWAY_TOKEN=token,
        BCG_CODEX_SESSION_ID=secrets.token_hex(16),
        BCG_NATIVE_HOOKS_DISABLED="1",
    )
    provider = (
        '{name="BCG context gateway", wire_api="responses", requires_openai_auth=true, '
        f'base_url="http://127.0.0.1:{port}/v1", supports_websockets=false, '
        'env_http_headers={"X-BCG-Gateway-Token"="BCG_CODEX_GATEWAY_TOKEN", "X-BCG-Session-ID"="BCG_CODEX_SESSION_ID"}}'
    )
    # Each launch must use this process's temporary provider configuration,
    # rather than reuse a Codex daemon carrying settings from an earlier run.
    codex_arguments = list(arguments)
    if "--no-daemon" not in codex_arguments:
        codex_arguments.insert(0, "--no-daemon")
    command = [
        codex,
        *codex_arguments,
        "-c",
        'model_provider="bcg_gateway"',
        "-c",
        f"model_providers.bcg_gateway={provider}",
        "-c",
        "features.enable_request_compression=false",
    ]
    if has_memory_mcp:
        command.extend(["-c", "mcp_servers.bcg_memory.enabled=false"])
    print(
        "BCG gateway: one native Codex session; request-side two-step window. Experimental; native compaction may limit retained history.",
        file=sys.stderr,
    )
    original = sys.stdout
    log = (state / "gateway-graph.log").open("a", buffering=1)
    sys.stdout = log
    try:
        return subprocess.call(command, cwd=config.project, env=env)
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        sys.stdout = original
        log.close()
