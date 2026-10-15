"""Project-local gateway setup. Never edit Codex configuration or credentials."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import Config, default_graph_config, private_write


def _require_gateway_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("invalid BCG installation manifest")
    if (
        manifest.get("mode") != "gateway"
        or manifest.get("blocks")
        or manifest.get("hooks")
    ):
        raise ValueError(
            "This project has a legacy BCG installation. Use its archived installer "
            "to uninstall the old wiring before gateway initialization; memories are retained."
        )


def install(
    project: Path,
    *,
    env_file: Path,
    embedding_model: str,
    allow_upload: bool,
    mode: str = "gateway",
) -> Config:
    if not allow_upload:
        raise ValueError(
            "Graph construction sends captured conversation/tool text to the configured "
            "provider. Review this, then pass --allow-graph-upload."
        )
    if mode != "gateway":
        raise ValueError("only gateway mode is available")
    project = project.expanduser().resolve()
    if not project.is_dir():
        raise ValueError("project directory does not exist")
    root = project / ".bcg-codex"
    config_path, manifest_path = root / "config.json", root / "install.json"
    graph_path, ignore_path = root / "graph-model.json", root / ".gitignore"
    for path in (root, config_path, manifest_path, graph_path, ignore_path):
        if path.is_symlink():
            raise ValueError("refusing symlinked integration/config paths")
    if manifest_path.exists():
        _require_gateway_manifest(json.loads(manifest_path.read_text()))
    if config_path.exists():
        existing = Config.load(config_path)
        if Path(existing.project).resolve() != project:
            raise ValueError(
                "existing BCG configuration belongs to a different project"
            )
        config = replace(
            existing, enabled=True, env_file=str(env_file.expanduser().resolve())
        )
    else:
        config = Config(
            str(project),
            str(root / "state"),
            str(graph_path),
            str(env_file.expanduser().resolve()),
        )
    selected_graph = Path(config.graph_config)
    if selected_graph.is_symlink():
        raise ValueError("refusing symlinked Graph configuration")
    if selected_graph.exists() and not selected_graph.is_file():
        raise ValueError("Graph configuration must be a file")
    if selected_graph != graph_path and not selected_graph.is_file():
        raise ValueError(
            "existing custom Graph configuration is missing; restore it first"
        )
    ignore = ignore_path.read_text() if ignore_path.exists() else ""
    ignore_block = "# Private BCG configuration and session records\n*\n!.gitignore\n"
    if not ignore.endswith(ignore_block):
        ignore = (ignore.rstrip() + "\n\n" if ignore else "") + ignore_block
    root.mkdir(mode=0o700, exist_ok=True)
    if not selected_graph.exists():
        private_write(selected_graph, default_graph_config(embedding_model))
    config.save(config_path)
    private_write(ignore_path, ignore)
    private_write(manifest_path, {"mode": "gateway", "blocks": [], "hooks": {}})
    return config


def uninstall(project: Path) -> None:
    root = project.expanduser().resolve() / ".bcg-codex"
    config_path, manifest_path = root / "config.json", root / "install.json"
    for path in (root, manifest_path, config_path):
        if path.is_symlink():
            raise ValueError("refusing symlinked integration/config paths")
    if not manifest_path.exists():
        raise ValueError("No BCG installation manifest in this project")
    _require_gateway_manifest(json.loads(manifest_path.read_text()))
    replace(Config.load(config_path), enabled=False).save(config_path)
    manifest_path.unlink()
