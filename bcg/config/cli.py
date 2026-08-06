"""``bcg config`` subcommands: show effective settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from bcg.config import load_settings, locate_config_files

app = typer.Typer(
    name="config",
    help="Inspect the unified configuration.",
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


_all__ = ["app"]
