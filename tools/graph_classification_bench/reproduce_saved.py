"""Offline/read-only check for the self-contained result bundle with repo deps installed."""
import json
import sys
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[2]
    bundle = source.parent
    if not (bundle/"dataset/blind.lock.json").is_file():
        raise SystemExit("Run the copy inside the extracted bundle: python reproduce/tools/graph_classification_bench/reproduce_saved.py")
    sys.path.insert(0,str(source))
    from tools.graph_classification_bench import freeze
    freeze.BASE = bundle
    from tools.graph_classification_bench import preflight, score
    from tools.graph_classification_bench.summarize import ARMS
    preflight.main()
    for a in ARMS:
        for s in ("dev","test"):
            actual = score.score(a,s,"r1",write=False)
            saved = json.loads((bundle/f"experiments/{a}/{s}/r1/score.json").read_text())
            q = actual["quality"]
            assert (actual["failed"],q["correct"],q["n"]) == (saved["failed"],saved["quality"]["correct"],saved["quality"]["n"])
            assert abs(q["group_balanced_macro_f1"] - saved["quality"]["group_balanced_macro_f1"]) < 1e-12
            print(f"{a:22s} {s:4s} {q['correct']:2d}/{q['n']}   Macro-F1={q['group_balanced_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
