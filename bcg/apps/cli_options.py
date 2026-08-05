"""Shared CLI option declarations for construct entry points (step 6).

``add_run_options`` centralizes the run knobs previously copy-pasted across
``run.py`` / ``online_server.py`` / ``online_driver.py``. Defaults
deliberately stay at their pre-unification values (0.86, 100000, verify
ON); step 7 consolidates them through the YAML defaults.
"""

from __future__ import annotations

import argparse


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared api_based run knobs (evidence mode / merge / context)."""

    parser.add_argument(
        "--evidence-mode",
        choices=["sentence", "excerpt"],
        default="sentence",
        help="'sentence' = evidence is always a complete sentence; "
        "'excerpt' = model quotes verbatim spans. Default: sentence.",
    )
    parser.add_argument(
        "--incremental-merge",
        dest="incremental_merge",
        default=True,
        action="store_true",
        help="Run an embedding-only per-turn merge (needs the embedding entry). "
        "Default: ON.",
    )
    parser.add_argument(
        "--no-incremental-merge",
        dest="incremental_merge",
        action="store_false",
        help="Disable the per-turn incremental merge.",
    )
    parser.add_argument(
        "--incremental-merge-threshold",
        type=float,
        default=0.86,
        help="Cosine threshold for the per-turn incremental merge. Default: 0.86.",
    )
    parser.add_argument(
        "--verify-merge",
        dest="verify_merge",
        default=True,
        action="store_true",
        help="LLM-verify and rewrite per-turn embedding merge groups. Default: ON.",
    )
    parser.add_argument(
        "--no-verify-merge",
        dest="verify_merge",
        action="store_false",
        help="Disable LLM verification for the per-turn incremental merge.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=100000,
        help="Char budget of the existing-nodes context block. Default: 100000.",
    )
    parser.add_argument(
        "--min-content-len",
        type=int,
        default=0,
        help="Skip turns whose content is shorter than this. Default: 0.",
    )


__all__ = ["add_run_options"]
