"""Shared CLI option declarations for construct entry points (step 6).

``add_run_options`` centralizes the run knobs previously copy-pasted across
``run.py`` / ``online_server.py`` / ``online_driver.py``. Defaults come from
the packaged ``defaults.yaml`` (``runner`` section), the single behavioral
source after the step-7 unification.
"""

from __future__ import annotations

import argparse

from bcg.config import defaults_dict


def _runner_defaults() -> dict:
    runner = defaults_dict().get("runner") or {}
    return {
        "evidence_mode": runner.get("evidence_mode", "sentence"),
        "incremental_merge": runner.get("incremental_merge", True),
        "incremental_merge_threshold": runner.get("incremental_merge_threshold", 0.8),
        "verify_merge": runner.get("verify_merge", False),
        "context_chars": runner.get("context_chars", 9000),
        "min_content_len": runner.get("min_content_len", 0),
    }


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared api_based run knobs (evidence mode / merge / context)."""
    defaults = _runner_defaults()

    parser.add_argument(
        "--evidence-mode",
        choices=["sentence", "excerpt"],
        default=defaults["evidence_mode"],
        help="'sentence' = evidence is always a complete sentence; "
        "'excerpt' = model quotes verbatim spans. Default: sentence.",
    )
    parser.add_argument(
        "--incremental-merge",
        dest="incremental_merge",
        default=defaults["incremental_merge"],
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
        default=defaults["incremental_merge_threshold"],
        help=f"Cosine threshold for the per-turn incremental merge. "
        f"Default: {defaults['incremental_merge_threshold']}.",
    )
    parser.add_argument(
        "--verify-merge",
        dest="verify_merge",
        default=defaults["verify_merge"],
        action="store_true",
        help="LLM-verify and rewrite per-turn embedding merge groups. "
        "Default: ON.",
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
        default=defaults["context_chars"],
        help=f"Char budget of the existing-nodes context block. "
        f"Default: {defaults['context_chars']}.",
    )
    parser.add_argument(
        "--min-content-len",
        type=int,
        default=defaults["min_content_len"],
        help="Skip turns whose content is shorter than this. Default: 0.",
    )


__all__ = ["add_run_options"]
