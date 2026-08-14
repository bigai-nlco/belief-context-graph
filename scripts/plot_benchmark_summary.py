#!/usr/bin/env python3
"""Render the full-dataset benchmark summary used in the README."""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

TEXT_COLOR = "#0f172a"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"
DEFAULT_COLOR = "#64748b"
BCG_COLOR = "#5b5bd6"
GRAPH_COLOR = "#aaa8f4"


@dataclass(frozen=True)
class Benchmark:
    title: str
    task_count: int
    accuracy_default: float
    accuracy_bcg: float
    accuracy_max: float
    accuracy_ticks: tuple[float, ...]
    tokens_default: float
    tokens_bcg: float
    graph_tokens: float
    token_max: float
    token_ticks: tuple[float, ...]


BENCHMARKS = (
    Benchmark(
        title="BrowseComp",
        task_count=1266,
        accuracy_default=33.33,
        accuracy_bcg=37.12,
        accuracy_max=50,
        accuracy_ticks=(0, 10, 20, 30, 40, 50),
        tokens_default=35_390,
        tokens_bcg=29_640,
        graph_tokens=7_100,
        token_max=50_000,
        token_ticks=(0, 10_000, 20_000, 30_000, 40_000, 50_000),
    ),
    Benchmark(
        title="BrowseComp-ZH",
        task_count=289,
        accuracy_default=49.48,
        accuracy_bcg=59.17,
        accuracy_max=75,
        accuracy_ticks=(0, 25, 50, 75),
        tokens_default=30_840,
        tokens_bcg=27_900,
        graph_tokens=6_350,
        token_max=40_000,
        token_ticks=(0, 10_000, 20_000, 30_000, 40_000),
    ),
)


def _y(value: float, maximum: float, top: float, bottom: float) -> float:
    return bottom - (bottom - top) * value / maximum


def _rounded_bar(x: float, y: float, width: float, bottom: float, color: str) -> str:
    radius = 7
    return (
        f'<path d="M{x:.1f} {bottom:.1f}V{y + radius:.1f}'
        f'a{radius} {radius} 0 0 1 {radius} -{radius}h{width - 2 * radius:.1f}'
        f'a{radius} {radius} 0 0 1 {radius} {radius}V{bottom:.1f}Z" fill="{color}"/>'
    )


def _format_tokens(value: float) -> str:
    return f"{value / 1000:.2f}K"


def render_svg(output: Path) -> None:
    width, height = 1280, 620
    plot_top, plot_bottom = 190.0, 475.0
    panel_lefts = (88.0, 703.0)
    panel_width = 525.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Accuracy and mean token cost by benchmark</title>',
        '<desc id="desc">Full-dataset Default and BCG accuracy and mean token cost for BrowseComp and BrowseComp-ZH. BCG token cost is divided into Agent and Graph Construction tokens.</desc>',
        '<rect width="1280" height="620" fill="#ffffff"/>',
        '<g font-family="Inter,Arial,sans-serif">',
        f'<text x="640" y="39" text-anchor="middle" fill="{TEXT_COLOR}" font-size="27" font-weight="700">Accuracy and mean token cost by benchmark</text>',
        f'<rect x="438" y="62" width="14" height="14" rx="3" fill="{DEFAULT_COLOR}"/><text x="462" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Default</text>',
        f'<rect x="548" y="62" width="14" height="14" rx="3" fill="{BCG_COLOR}"/><text x="572" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">BCG</text>',
        f'<rect x="638" y="62" width="14" height="14" rx="3" fill="{GRAPH_COLOR}"/><text x="662" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Graph portion</text>',
    ]

    for panel_index, item in enumerate(BENCHMARKS):
        left = panel_lefts[panel_index]
        right = left + panel_width
        axis_left = left + 48
        axis_right = right - 48
        accuracy_x = (left + 89, left + 161)
        token_x = (left + 305, left + 377)
        bar_width = 60.0

        parts.extend(
            [
                f'<text x="{left:.0f}" y="116" fill="{TEXT_COLOR}" font-size="21" font-weight="700">{html.escape(item.title)}</text>',
                f'<text x="{right:.0f}" y="116" text-anchor="end" fill="{MUTED_COLOR}" font-size="13">{item.task_count:,} tasks / mode</text>',
                f'<text x="{axis_left:.1f}" y="171" fill="#475569" font-size="14" font-weight="700">Accuracy</text>',
                f'<text x="{axis_right:.1f}" y="171" text-anchor="end" fill="#475569" font-size="14" font-weight="700">Mean tokens / task</text>',
            ]
        )

        for tick in item.token_ticks:
            tick_y = _y(tick, item.token_max, plot_top, plot_bottom)
            parts.append(
                f'<path d="M{axis_left:.1f} {tick_y:.1f}H{axis_right:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            token_label = "0" if tick == 0 else f"{tick / 1000:g}K"
            parts.append(
                f'<text x="{axis_right + 10:.1f}" y="{tick_y + 5:.1f}" fill="{MUTED_COLOR}" font-size="13">{token_label}</text>'
            )

        for tick in item.accuracy_ticks:
            tick_y = _y(tick, item.accuracy_max, plot_top, plot_bottom)
            parts.extend(
                [
                    f'<path d="M{axis_left - 7:.1f} {tick_y:.1f}H{axis_left:.1f}" stroke="{MUTED_COLOR}" stroke-width="1"/>',
                    f'<text x="{axis_left - 12:.1f}" y="{tick_y + 5:.1f}" text-anchor="end" fill="{MUTED_COLOR}" font-size="13">{tick:g}%</text>',
                ]
            )

        parts.append(
            f'<path d="M{axis_left:.1f} {plot_top:.1f}V{plot_bottom:.1f}H{axis_right:.1f}V{plot_top:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.2"/>'
        )

        accuracy_values = (item.accuracy_default, item.accuracy_bcg)
        for index, (x, value, color, mode) in enumerate(
            zip(
                accuracy_x,
                accuracy_values,
                (DEFAULT_COLOR, BCG_COLOR),
                ("Default", "BCG"),
                strict=True,
            )
        ):
            del index
            bar_y = _y(value, item.accuracy_max, plot_top, plot_bottom)
            parts.extend(
                [
                    _rounded_bar(x, bar_y, bar_width, plot_bottom, color),
                    f'<text x="{x + bar_width / 2:.1f}" y="{bar_y - 12:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{value:.2f}%</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="500" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13" font-weight="600">{mode}</text>',
                ]
            )

        default_y = _y(item.tokens_default, item.token_max, plot_top, plot_bottom)
        bcg_y = _y(item.tokens_bcg, item.token_max, plot_top, plot_bottom)
        agent_tokens = item.tokens_bcg - item.graph_tokens
        graph_boundary_y = _y(agent_tokens, item.token_max, plot_top, plot_bottom)
        parts.extend(
            [
                _rounded_bar(
                    token_x[0], default_y, bar_width, plot_bottom, DEFAULT_COLOR
                ),
                f'<rect x="{token_x[1]:.1f}" y="{graph_boundary_y:.1f}" width="{bar_width:.1f}" height="{plot_bottom - graph_boundary_y:.1f}" fill="{BCG_COLOR}"/>',
                _rounded_bar(
                    token_x[1], bcg_y, bar_width, graph_boundary_y, GRAPH_COLOR
                ),
                f'<text x="{token_x[0] + bar_width / 2:.1f}" y="{default_y - 12:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{_format_tokens(item.tokens_default)}</text>',
                f'<text x="{token_x[1] + bar_width / 2:.1f}" y="{bcg_y - 12:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{_format_tokens(item.tokens_bcg)}</text>',
                f'<text x="{token_x[1] + bar_width / 2:.1f}" y="{(graph_boundary_y + plot_bottom) / 2 + 5:.1f}" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="700">{_format_tokens(agent_tokens)}</text>',
                f'<text x="{token_x[1] + bar_width / 2:.1f}" y="{(bcg_y + graph_boundary_y) / 2 + 5:.1f}" text-anchor="middle" fill="#312e81" font-size="13" font-weight="700">{_format_tokens(item.graph_tokens)}</text>',
                f'<text x="{token_x[0] + bar_width / 2:.1f}" y="500" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13" font-weight="600">Default</text>',
                f'<text x="{token_x[1] + bar_width / 2:.1f}" y="500" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13" font-weight="600">BCG</text>',
                f'<text x="{sum(accuracy_x) / 2 + bar_width / 2:.1f}" y="536" text-anchor="middle" fill="{TEXT_COLOR}" font-size="16" font-weight="700">Accuracy</text>',
                f'<text x="{sum(token_x) / 2 + bar_width / 2:.1f}" y="536" text-anchor="middle" fill="{TEXT_COLOR}" font-size="16" font-weight="700">Token cost</text>',
            ]
        )

    parts.extend(["</g>", "</svg>"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("assert/benchmark_summary.svg")
    )
    args = parser.parse_args()
    render_svg(args.output)


if __name__ == "__main__":
    main()
