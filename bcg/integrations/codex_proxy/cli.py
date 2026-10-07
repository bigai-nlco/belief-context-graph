"""Opt-in commands. No command installs or edits Codex configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import ProxyConfig
from .context import ContextEngine, RequestError, Session

app = typer.Typer(
    help="Experimental Codex Responses gateway. Review before connecting a real Codex session.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def serve(config: Annotated[Path, typer.Option(exists=True, dir_okay=False)]) -> None:
    """Start the loopback proxy; does not launch Codex or a Graph server."""
    from .server import ProxyServer

    try:
        settings = ProxyConfig.load(config)
        server = ProxyServer(settings)
    except (ValueError, OSError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"BCG Codex proxy: http://{settings.host}:{server.server_port}/v1")
    typer.echo(f"Mode: {settings.mode}; recent complete steps: {settings.recent_turns}")
    typer.echo("Codex is NOT launched/configured. Ctrl+C stops the proxy.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command("codex-config")
def codex_config(
    model: Annotated[
        str,
        typer.Option(
            help="Explicit model supported by your upstream Responses endpoint."
        ),
    ],
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8860,
    chatgpt: Annotated[
        bool, typer.Option(help="Use Codex login and a separate local gateway header.")
    ] = False,
) -> None:
    """Print a review-only TOML snippet to stdout; never modify ~/.codex."""
    auth = (
        "requires_openai_auth = true"
        if chatgpt
        else 'env_key = "BCG_CODEX_GATEWAY_TOKEN"'
    )
    headers = '"X-BCG-Session-ID" = "BCG_CODEX_SESSION_ID"'
    if chatgpt:
        headers += ', "X-BCG-Gateway-Token" = "BCG_CODEX_GATEWAY_TOKEN"'
    typer.echo(f"""# REVIEW ONLY. Do not replace your existing Codex configuration.
# Use a fresh BCG_CODEX_SESSION_ID for every independent thread/process.
# BCG_CODEX_GATEWAY_TOKEN authenticates to localhost, not to the model provider.
[model_providers.bcg_proxy]
name = "BCG local context gateway (experimental)"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
{auth}
supports_websockets = false
stream_idle_timeout_ms = 900000
env_http_headers = {{ {headers} }}

[profiles.bcg_proxy]
model_provider = "bcg_proxy"
model = {json.dumps(model)}

[profiles.bcg_proxy.features]
enable_request_compression = false
""")


@app.command()
def preview(
    request: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    snapshot: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    selection: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    recent_turns: Annotated[int, typer.Option(min=0)] = 2,
) -> None:
    """Render a request using supplied graph fixtures. NO HTTP or LLM calls."""

    class FixtureGraph:
        def push(self, problem_id: str, turns: list[dict]) -> dict:
            value = json.loads(snapshot.read_text(encoding="utf-8"))
            return {**value, "problem_id": problem_id}

        def select(
            self, problem_id: str, query: str, question: str, budget: int
        ) -> dict:
            return json.loads(selection.read_text(encoding="utf-8"))

    try:
        body = json.loads(request.read_text(encoding="utf-8"))
        engine = ContextEngine(
            ProxyConfig(
                upstream_base_url="http://offline.invalid/v1",
                recent_turns=recent_turns,
                audit_dir=None,
            ),
            FixtureGraph(),
        )
        outgoing, metadata = engine.prepare(body, Session("offline-preview"))
    except (ValueError, OSError, TypeError, RequestError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {"offline_fixture": True, "metadata": metadata, "request": outgoing},
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    app(args=argv, prog_name="bcg codex-proxy")
