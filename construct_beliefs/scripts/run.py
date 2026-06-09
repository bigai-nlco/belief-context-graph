#!/usr/bin/env python3
"""
scripts/run.py
==============
Command-line driver for the `construct_beliefs` pipeline.

Examples
--------
  # Run all stages end-to-end:
  python scripts/run.py --input traj.json

  # Run a single stage (useful when iterating on prompts):
  python scripts/run.py --stage segment       --input traj.json
  python scripts/run.py --stage io
  python scripts/run.py --stage reasoning
  python scripts/run.py --stage forward       # NEW: derivation edges (informs)
  python scripts/run.py --stage backward      # renamed from `link`: confirms/contradicts/extends
  python scripts/run.py --stage finalize

Each stage reads / writes a file in `--output-dir`, so partial runs resume
automatically.
"""

import argparse
import sys
from pathlib import Path

# Make the package importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P
from src.llm import load_config, make_client


def main():
    parser = argparse.ArgumentParser(description="construct_beliefs pipeline driver")
    parser.add_argument(
        "--input", "-i",
        default="D:\\a_projects\\belief_memory\\BGM_example.txt",
        help="Trajectory JSON/TXT file (required for stage=segment / all).",
    )
    parser.add_argument("--config", "-c", default="D:\\a_projects\\belief_memory\\construct_beliefs\\model_config.json",
                        help="Model config (base_url / api_key / model).")
    parser.add_argument("--output-dir", "-o", default="D:\\a_projects\\belief_memory\\construct_beliefs\\outputs",
                        help="Output directory for layered JSON files.")
    parser.add_argument(
        "--stage", "-s",
        choices=["all", "segment", "io", "reasoning", "forward", "backward", "finalize"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--model-key", default="deepseek-v4-flash-260425",
        help="When the config nests entries by model name, pick which one to use. Defaults to the first key.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        P.run_all(args.input, args.config, out_dir, model_key=args.model_key)
        return

    # Single-stage runs reuse files from previous stages.
    if args.stage == "segment":
        P.stage_segment(args.input, out_dir)
        return

    # All later stages need an LLM client.
    cfg = load_config(args.config, model_key=args.model_key)
    client = make_client(cfg)
    model = cfg.get("model") or cfg.get("model_name") or "gpt-4o-mini"
    max_tokens = cfg.get("max_tokens")

    if args.stage == "io":
        segments, _ = P._load_segments(out_dir)
        P.stage_io(client, model, segments, out_dir, max_tokens=max_tokens)
        return

    if args.stage == "reasoning":
        segments, _ = P._load_segments(out_dir)
        io_payload = P._load_json(out_dir / "02_io_beliefs.json")
        P.stage_reasoning(client, model, segments, io_payload["beliefs"], out_dir, max_tokens=max_tokens)
        return

    if args.stage == "forward":
        io_payload  = P._load_json(out_dir / "02_io_beliefs.json")
        rea_payload = P._load_json(out_dir / "03_reasoning_beliefs.json")
        combined = P.renumber_chronologically(io_payload["beliefs"] + rea_payload["beliefs"])
        P.stage_forward(client, model, combined, out_dir, max_tokens=max_tokens)
        return

    if args.stage == "backward":
        io_payload  = P._load_json(out_dir / "02_io_beliefs.json")
        rea_payload = P._load_json(out_dir / "03_reasoning_beliefs.json")
        combined = P.renumber_chronologically(io_payload["beliefs"] + rea_payload["beliefs"])
        P.stage_backward(client, model, combined, out_dir, max_tokens=max_tokens)
        return

    if args.stage == "finalize":
        seg_payload = P._load_json(out_dir / "01_segments.json")
        io_payload  = P._load_json(out_dir / "02_io_beliefs.json")
        rea_payload = P._load_json(out_dir / "03_reasoning_beliefs.json")
        # Renumber to keep ids consistent with what forward/backward stages saw.
        combined = P.renumber_chronologically(io_payload["beliefs"] + rea_payload["beliefs"])
        io_chrono = [b for b in combined if b.get("layer") == "io"]
        rea_chrono = [b for b in combined if b.get("layer") == "reasoning"]
        # Forward/backward files are optional — default to empty lists if missing.
        try:
            fwd = P._load_json(out_dir / "04_forward_relations.json")["forward_relations"]
        except FileNotFoundError:
            fwd = []
            print("[warn] 04_forward_relations.json not found, using []", file=sys.stderr)
        try:
            bwd = P._load_json(out_dir / "05_backward_relations.json")["relations"]
        except FileNotFoundError:
            bwd = []
            print("[warn] 05_backward_relations.json not found, using []", file=sys.stderr)
        P.stage_finalize(
            seg_payload["trajectory"],
            io_chrono, rea_chrono,
            fwd, bwd,
            model=model,
            output_dir=out_dir,
            extra_meta={"input_path": seg_payload.get("input_path")},
        )
        return


if __name__ == "__main__":
    main()
