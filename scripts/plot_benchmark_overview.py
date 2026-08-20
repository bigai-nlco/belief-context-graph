#!/usr/bin/env python3
"""Compose the README benchmark charts into one compact SVG canvas."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NAMESPACE)


def _tag(name: str) -> str:
    return f"{{{SVG_NAMESPACE}}}{name}"


def _visible_group(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    group = root.find(_tag("g"))
    if group is None:
        raise ValueError(f"No visible SVG group found in {path}")
    return deepcopy(group)


def compose(summary: Path, horizon: Path, output: Path) -> None:
    width, height = 1280, 1145
    horizon_offset = 540
    root = ET.Element(
        _tag("svg"),
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "role": "img",
            "aria-labelledby": "title desc",
        },
    )
    title = ET.SubElement(root, _tag("title"), {"id": "title"})
    title.text = "Benchmark accuracy and token-cost overview"
    description = ET.SubElement(root, _tag("desc"), {"id": "desc"})
    description.text = (
        "Full-dataset accuracy and mean token cost for BrowseComp and "
        "BrowseComp-ZH with GPT-5.6-luna, plus BrowseComp with Kimi K3; "
        "cumulative token cost by Agent model-call depth is shown for the "
        "GPT-5.6-luna evaluations."
    )
    ET.SubElement(
        root,
        _tag("rect"),
        {"width": str(width), "height": str(height), "fill": "#ffffff"},
    )

    summary_group = _visible_group(summary)
    horizon_group = _visible_group(horizon)
    horizon_group.set("transform", f"translate(0 {horizon_offset})")
    root.extend((summary_group, horizon_group))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=Path("assert/benchmark_summary.svg")
    )
    parser.add_argument(
        "--horizon", type=Path, default=Path("assert/token_cost.svg")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("assert/benchmark_overview.svg")
    )
    args = parser.parse_args()
    compose(args.summary, args.horizon, args.output)


if __name__ == "__main__":
    main()
