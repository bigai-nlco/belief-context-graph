"""``bcg config`` subcommands: show effective settings and migrate legacy JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from bcg.config import load_settings, locate_config_files
from bcg.config.migration import migrate_to_yaml

app = typer.Typer(
    name="config",
    help="Inspect the unified configuration or migrate legacy JSON files.",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("show")
def show(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Explicit YAML config file."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the effective settings as JSON.")
    ] = False,
) -> None:
    """Print the effective configuration and each field's source."""

    settings, sources = load_settings(explicit=str(config) if config else None)
    if json_output:
        typer.echo(json.dumps(settings.model_dump(), ensure_ascii=False, indent=2))
        return
    typer.echo(
        f"Loaded files: {[str(p) for p in locate_config_files(explicit=str(config) if config else None)] or 'packaged defaults only'}"
    )
    for section, value in settings.model_dump().items():
        if isinstance(value, dict):
            typer.echo(f"[{section}]")
            for key, item in value.items():
                if isinstance(item, dict):
                    typer.echo(f"  {key}: <{len(item)} entries>")
                else:
                    typer.echo(f"  {key}: {item}")
        else:
            typer.echo(f"{section}: {value}")


@app.command("migrate")
def migrate(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Destination YAML file (default: ~/.bcg/config.yaml).",
        ),
    ] = None,
) -> None:
    """Convert legacy model_config.json / ~/.bcg/config.json to YAML."""

    from bcg.setup import state_root

    dest = output or (state_root() / "config.yaml")
    try:
        written = migrate_to_yaml(dest)
    except FileNotFoundError as exc:
        typer.echo(f"bcg config migrate: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote {written}")
    typer.echo("Remove the legacy JSON files once the YAML config is verified.")


__all__ = ["app"]
