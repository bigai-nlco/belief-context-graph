"""Lazy Graph SDK initialization for the gateway, without hook queues or workers."""

from __future__ import annotations

import os
from pathlib import Path

from .config import Config


def create_graph_manager(config: Config):
    # Keep initialization and CLI help free of model loading or API requests.
    os.environ["BCG_ENV_FILE"] = config.env_file
    from bcg.core.env import load_project_env

    load_project_env(config.env_file, override=True)
    from bcg.construct.unified.online import SessionManager
    from bcg.construct.unified.stream import StreamOptions

    return SessionManager(
        config_path=config.graph_config,
        model_key="graph-model",
        output_root=Path(config.state_dir) / "graphs",
        options=StreamOptions(
            context_chars=12000, extraction_history_turns=2, max_previous_windows=3
        ),
    )
