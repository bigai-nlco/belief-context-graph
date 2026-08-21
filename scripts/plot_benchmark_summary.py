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
SUMMARY_COLOR = "#0f9f8f"
SUMMARY_MODEL_COLOR = "#99ded3"
IMPROVEMENT_COLOR = "#4338ca"


@dataclass(frozen=True)
class Benchmark:
    title: str
    task_count: int
    accuracy_default: float
    accuracy_bcg: float
    accuracy_summary: float
    accuracy_max: float
    accuracy_ticks: tuple[float, ...]
    tokens_default: float
    tokens_bcg: float
    graph_tokens: float
    tokens_summary: float
    summary_tokens: float
    token_max: float
    token_ticks: tuple[float, ...]


BENCHMARKS = (
    Benchmark(
        title="BrowseComp",
        task_count=1266,
        accuracy_default=33.33,
        accuracy_bcg=37.12,
        accuracy_summary=34.76,
        accuracy_max=50,
        accuracy_ticks=(0, 10, 20, 30, 40, 50),
        tokens_default=35_388.39,
        tokens_bcg=29_643.74,
        graph_tokens=7_098.74,
        tokens_summary=31_810.35,
        summary_tokens=10_450.00,
        token_max=50_000,
        token_ticks=(0, 10_000, 20_000, 30_000, 40_000, 50_000),
    ),
    Benchmark(
        title="BrowseComp-ZH",
        task_count=289,
        accuracy_default=49.48,
        accuracy_bcg=59.17,
        accuracy_summary=52.94,
        accuracy_max=75,
        accuracy_ticks=(0, 25, 50, 75),
        tokens_default=30_839.81,
        tokens_bcg=27_900.56,
        graph_tokens=6_353.08,
        tokens_summary=32_347.75,
        summary_tokens=9_670.52,
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


def _change_arrow(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    label: str,
    *,
    plot_top: float,
) -> list[str]:
    """Draw a compact curved comparison arrow above two adjacent bars."""
    # Value labels sit 12 px above each bar. Keep the complete arrow another
    # small step above those labels instead of terminating on the bar tops.
    arrow_y = max(plot_top + 48, min(start_y, end_y) - 34)
    tilt = max(-16.0, min(16.0, (end_y - start_y) * 0.45))
    arrow_start_y = arrow_y - tilt / 2
    arrow_end_y = arrow_y + tilt / 2
    curve_y = max(plot_top + 34, min(arrow_start_y, arrow_end_y) - 14)
    label_x = (start_x + end_x) / 2
    label_y = plot_top + 8
    label_width = max(68.0, len(label) * 7.2 + 18)
    return [
        f'<path d="M{start_x:.1f} {arrow_start_y:.1f} C{start_x + 12:.1f} {curve_y:.1f}, {end_x - 12:.1f} {curve_y:.1f}, {end_x:.1f} {arrow_end_y:.1f}" fill="none" stroke="{IMPROVEMENT_COLOR}" stroke-width="1.8" stroke-linecap="round" marker-end="url(#improvement-arrow)"/>',
        f'<rect x="{label_x - label_width / 2:.1f}" y="{label_y:.1f}" width="{label_width:.1f}" height="20" rx="10" fill="#ffffff" stroke="#c7d2fe"/>',
        f'<text x="{label_x:.1f}" y="{label_y + 15:.1f}" text-anchor="middle" fill="{IMPROVEMENT_COLOR}" font-size="11.5" font-weight="700">{html.escape(label)}</text>',
    ]


def render_svg(output: Path) -> None:
    width, height = 1280, 620
    plot_top, plot_bottom = 185.0, 480.0
    panel_lefts = (88.0, 703.0)
    panel_width = 525.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Accuracy and mean token cost by benchmark</title>',
        '<desc id="desc">Full-dataset Default, BCG, and Summary accuracy and mean token cost for BrowseComp and BrowseComp-ZH. The light BCG segment is Graph Construction, and the light Summary segment is Summary Generation.</desc>',
        '<rect width="1280" height="620" fill="#ffffff"/>',
        '<g font-family="Inter,Arial,sans-serif">',
        f'<defs><marker id="improvement-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="{IMPROVEMENT_COLOR}"/></marker></defs>',
        f'<text x="640" y="39" text-anchor="middle" fill="{TEXT_COLOR}" font-size="27" font-weight="700">Accuracy and mean token cost by benchmark</text>',
        f'<rect x="296" y="62" width="14" height="14" rx="3" fill="{DEFAULT_COLOR}"/><text x="320" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Default</text>',
        f'<rect x="400" y="62" width="14" height="14" rx="3" fill="{BCG_COLOR}"/><text x="424" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">BCG</text>',
        f'<rect x="480" y="62" width="14" height="14" rx="3" fill="{GRAPH_COLOR}"/><text x="504" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">BCG Graph Construction</text>',
        f'<rect x="704" y="62" width="14" height="14" rx="3" fill="{SUMMARY_COLOR}"/><text x="728" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Summary</text>',
        f'<rect x="816" y="62" width="14" height="14" rx="3" fill="{SUMMARY_MODEL_COLOR}"/><text x="840" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Summary Generation</text>',
    ]

    for panel_index, item in enumerate(BENCHMARKS):
        left = panel_lefts[panel_index]
        right = left + panel_width
        axis_left = left + 48
        axis_right = right - 48
        accuracy_x = (left + 80, left + 136, left + 192)
        token_x = (left + 286, left + 342, left + 398)
        bar_width = 48.0

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

        accuracy_values = (
            item.accuracy_default,
            item.accuracy_bcg,
            item.accuracy_summary,
        )
        for index, (x, value, color, mode) in enumerate(
            zip(
                accuracy_x,
                accuracy_values,
                (DEFAULT_COLOR, BCG_COLOR, SUMMARY_COLOR),
                ("Default", "BCG", "Summary"),
                strict=True,
            )
        ):
            del index
            bar_y = _y(value, item.accuracy_max, plot_top, plot_bottom)
            parts.extend(
                [
                    _rounded_bar(x, bar_y, bar_width, plot_bottom, color),
                    f'<text x="{x + bar_width / 2:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{value:.2f}%</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="500" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13" font-weight="600">{mode}</text>',
                ]
            )

        accuracy_tops = tuple(
            _y(value, item.accuracy_max, plot_top, plot_bottom)
            for value in accuracy_values
        )
        parts.extend(
            _change_arrow(
                accuracy_x[0] + bar_width / 2,
                accuracy_tops[0],
                accuracy_x[1] + bar_width / 2,
                accuracy_tops[1],
                f"+{item.accuracy_bcg - item.accuracy_default:.2f} pp",
                plot_top=plot_top,
            )
        )

        default_y = _y(item.tokens_default, item.token_max, plot_top, plot_bottom)
        parts.extend(
            [
                _rounded_bar(
                    token_x[0], default_y, bar_width, plot_bottom, DEFAULT_COLOR
                ),
                f'<text x="{token_x[0] + bar_width / 2:.1f}" y="{default_y - 6:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{_format_tokens(item.tokens_default)}</text>',
                f'<text x="{token_x[0] + bar_width / 2:.1f}" y="500" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13" font-weight="600">Default</text>',
                f'<text x="{sum(accuracy_x) / len(accuracy_x) + bar_width / 2:.1f}" y="536" text-anchor="middle" fill="{TEXT_COLOR}" font-size="16" font-weight="700">Accuracy</text>',
                f'<text x="{sum(token_x) / len(token_x) + bar_width / 2:.1f}" y="536" text-anchor="middle" fill="{TEXT_COLOR}" font-size="16" font-weight="700">Token cost</text>',
            ]
        )

        for x, total, memory, color, memory_color, mode in zip(
            token_x[1:],
            (item.tokens_bcg, item.tokens_summary),
            (item.graph_tokens, item.summary_tokens),
            (BCG_COLOR, SUMMARY_COLOR),
            (GRAPH_COLOR, SUMMARY_MODEL_COLOR),
            ("BCG", "Summary"),
            strict=True,
        ):
            total_y = _y(total, item.token_max, plot_top, plot_bottom)
            agent_tokens = total - memory
            boundary_y = _y(agent_tokens, item.token_max, plot_top, plot_bottom)
            memory_text_color = "#312e81" if mode == "BCG" else "#115e59"
            parts.extend(
                [
                    f'<rect x="{x:.1f}" y="{boundary_y:.1f}" width="{bar_width:.1f}" height="{plot_bottom - boundary_y:.1f}" fill="{color}"/>',
                    _rounded_bar(x, total_y, bar_width, boundary_y, memory_color),
                    f'<text x="{x + bar_width / 2:.1f}" y="{total_y - 6:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{_format_tokens(total)}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{(boundary_y + plot_bottom) / 2 + 5:.1f}" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="700">{_format_tokens(agent_tokens)}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{(total_y + boundary_y) / 2 + 5:.1f}" text-anchor="middle" fill="{memory_text_color}" font-size="12" font-weight="700">{_format_tokens(memory)}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="500" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13" font-weight="600">{mode}</text>',
                ]
            )

        token_tops = (
            _y(item.tokens_default, item.token_max, plot_top, plot_bottom),
            _y(item.tokens_bcg, item.token_max, plot_top, plot_bottom),
        )
        token_reduction = 1 - item.tokens_bcg / item.tokens_default
        parts.extend(
            _change_arrow(
                token_x[0] + bar_width / 2,
                token_tops[0],
                token_x[1] + bar_width / 2,
                token_tops[1],
                f"−{token_reduction:.1%}",
                plot_top=plot_top,
            )
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
