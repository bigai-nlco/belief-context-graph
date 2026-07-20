"""Named, version-controlled parameter bundles for common agent runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


# Preset arguments are prepended to user arguments. Argparse therefore keeps
# its normal "last occurrence wins" behavior, so a caller can override any
# preset value without a second merging implementation.
_PRESET_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "averitec-hero4": (
        "--backend",
        "api",
        "--tasks",
        "averitec",
        "--tools",
        "averitec_search",
        "--retrieval-method",
        "hero4",
        "--retrieval-max-results",
        "10",
        "--hero-bm25-top-k",
        "10",
        "--hero-embedding-device",
        "cpu",
        "--stage1-bm25-k",
        "1000",
        "--stage2-embed-k",
        "32",
        "--stage3-rerank-k",
        "5",
        "--enable-archive",
        "--recent-turns",
        "2",
        "--belief-graph-mode",
        "augment",
        "--belief-graph-timeout",
        "600",
        "--graph-format",
        "deepseek_v4",
        "--belief-graph-placement",
        "system",
        "--file-tool-root",
        "ai_workspace",
        "--max-problems",
        "10",
        "--n-parallel-tasks",
        "1",
        "--enable-thinking",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--num-samples",
        "1",
        "--passk",
        "1",
        "--max-steps",
        "12",
        "--max-new-tokens",
        "32768",
        "--max-response-length",
        "32768",
        "--max-prompt-length",
        "32768",
        "--prompt",
        "averitec_nohyde.txt",
        "--no-hyde",
        "--no-shuffle",
        "--shuffle-seed",
        "0",
        "--output-dir",
        "output",
        "--save-alias",
        "augment_dsv4_system",
        "--overwrite",
    ),
}


def preset_names() -> tuple[str, ...]:
    """Return stable preset names suitable for CLI choices and documentation."""

    return tuple(_PRESET_ARGUMENTS)


def expand_preset_args(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Remove ``--preset`` and prepend its argument bundle when requested."""

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--preset", choices=preset_names())
    selected, remaining = selector.parse_known_args(argv)
    if selected.preset is None:
        return None, list(argv)
    return selected.preset, [*_PRESET_ARGUMENTS[selected.preset], *remaining]


__all__ = ["expand_preset_args", "preset_names"]
