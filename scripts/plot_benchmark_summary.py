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
    tokens_default: float
    tokens_bcg: float
    graph_tokens: float
    tokens_summary: float
    summary_tokens: float


BENCHMARKS = (
    Benchmark(
        title="BrowseComp",
        task_count=1266,
        accuracy_default=33.33,
        accuracy_bcg=37.12,
        accuracy_summary=34.76,
        tokens_default=35_388.39,
        tokens_bcg=29_643.74,
        graph_tokens=7_098.74,
        tokens_summary=31_810.35,
        summary_tokens=10_450.00,
    ),
    Benchmark(
        title="BrowseComp-ZH",
        task_count=289,
        accuracy_default=49.48,
        accuracy_bcg=59.17,
        accuracy_summary=52.94,
        tokens_default=30_839.81,
        tokens_bcg=27_900.56,
        graph_tokens=6_353.08,
        tokens_summary=32_347.75,
        summary_tokens=9_670.52,
    ),
)


def _y(value: float, maximum: float, top: float, bottom: float) -> float:
    return bottom - (bottom - top) * value / maximum


def _rounded_bar(x: float, y: float, width: float, bottom: float, color: str) -> str:
    radius = 7
    return (
        f'<path d="M{x:.1f} {bottom:.1f}V{y + radius:.1f}'
        f"a{radius} {radius} 0 0 1 {radius} -{radius}h{width - 2 * radius:.1f}"
        f'a{radius} {radius} 0 0 1 {radius} {radius}V{bottom:.1f}Z" fill="{color}"/>'
    )


def _format_tokens(value: float) -> str:
    return f"{value / 1000:.2f}K"


def _benchmark_tick(center: float, item: Benchmark) -> list[str]:
    return [
        f'<text x="{center:.1f}" y="516" text-anchor="middle" fill="{TEXT_COLOR}" font-size="15" font-weight="700">{html.escape(item.title)}</text>',
        f'<text x="{center:.1f}" y="537" text-anchor="middle" fill="{MUTED_COLOR}" font-size="12">{item.task_count:,} tasks / mode</text>',
    ]


def _change_arrow(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    label: str,
    *,
    plot_top: float,
    middle_y: float | None = None,
) -> list[str]:
    """Draw a compact curved comparison arrow above two adjacent bars."""
    # Value labels sit 12 px above each bar. Keep the complete arrow another
    # small step above those labels instead of terminating on the bar tops.
    arrow_y = max(plot_top + 48, min(start_y, end_y) - 34)
    tilt = max(-16.0, min(16.0, (end_y - start_y) * 0.45))
    arrow_start_y = arrow_y - tilt / 2
    arrow_end_y = arrow_y + tilt / 2
    curve_y = max(plot_top + 34, min(arrow_start_y, arrow_end_y) - 14)
    if middle_y is not None:
        # Default and BCG surround Summary in the bar order. Lift the arc when
        # the middle bar is tall enough for its value label to intersect it.
        curve_y = min(curve_y, middle_y - 32)
    label_x = (start_x + end_x) / 2
    label_y = max(plot_top + 4, curve_y - 27)
    label_width = max(68.0, len(label) * 7.2 + 18)
    return [
        f'<path d="M{start_x:.1f} {arrow_start_y:.1f} C{start_x + 12:.1f} {curve_y:.1f}, {end_x - 12:.1f} {curve_y:.1f}, {end_x:.1f} {arrow_end_y:.1f}" fill="none" stroke="{IMPROVEMENT_COLOR}" stroke-width="1.8" stroke-linecap="round" marker-end="url(#improvement-arrow)"/>',
        f'<rect x="{label_x - label_width / 2:.1f}" y="{label_y:.1f}" width="{label_width:.1f}" height="20" rx="10" fill="#ffffff" stroke="#c7d2fe"/>',
        f'<text x="{label_x:.1f}" y="{label_y + 15:.1f}" text-anchor="middle" fill="{IMPROVEMENT_COLOR}" font-size="11.5" font-weight="700">{html.escape(label)}</text>',
    ]


def render_svg(output: Path) -> None:
    width, height = 1280, 620
    plot_top, plot_bottom = 185.0, 485.0
    bar_width = 44.0
    accuracy_max = 70.0
    accuracy_ticks = tuple(range(0, 71, 10))
    token_max = 40_000.0
    token_ticks = (0, 10_000, 20_000, 30_000, 40_000)
    accuracy_axis = (135.0, 600.0)
    token_axis = (745.0, 1210.0)
    accuracy_centers = (260.0, 475.0)
    token_centers = (870.0, 1085.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Accuracy and mean token cost by benchmark</title>',
        '<desc id="desc">Full-dataset Default, Summary, and BCG accuracy and mean token cost for BrowseComp and BrowseComp-ZH. The light Summary segment is Summary Generation, and the light BCG segment is Graph Construction.</desc>',
        '<rect width="1280" height="620" fill="#ffffff"/>',
        '<g font-family="Inter,Arial,sans-serif">',
        f'<defs><marker id="improvement-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="{IMPROVEMENT_COLOR}"/></marker></defs>',
        f'<text x="640" y="39" text-anchor="middle" fill="{TEXT_COLOR}" font-size="27" font-weight="700">Accuracy and mean token cost by benchmark</text>',
        f'<rect x="296" y="62" width="14" height="14" rx="3" fill="{DEFAULT_COLOR}"/><text x="320" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Default</text>',
        f'<rect x="400" y="62" width="14" height="14" rx="3" fill="{SUMMARY_COLOR}"/><text x="424" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Summary</text>',
        f'<rect x="512" y="62" width="14" height="14" rx="3" fill="{SUMMARY_MODEL_COLOR}"/><text x="536" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">Summary Generation</text>',
        f'<rect x="704" y="62" width="14" height="14" rx="3" fill="{BCG_COLOR}"/><text x="728" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">BCG</text>',
        f'<rect x="784" y="62" width="14" height="14" rx="3" fill="{GRAPH_COLOR}"/><text x="808" y="75" fill="{TEXT_COLOR}" font-size="15" font-weight="600">BCG Graph Construction</text>',
        f'<text x="90" y="130" fill="{TEXT_COLOR}" font-size="21" font-weight="700">Accuracy</text>',
        f'<text x="700" y="130" fill="{TEXT_COLOR}" font-size="21" font-weight="700">Mean tokens / task</text>',
    ]

    for tick in accuracy_ticks:
        tick_y = _y(tick, accuracy_max, plot_top, plot_bottom)
        parts.extend(
            [
                f'<path d="M{accuracy_axis[0]:.1f} {tick_y:.1f}H{accuracy_axis[1]:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>',
                f'<text x="{accuracy_axis[0] - 12:.1f}" y="{tick_y + 5:.1f}" text-anchor="end" fill="{MUTED_COLOR}" font-size="13">{tick}%</text>',
            ]
        )
    for tick in token_ticks:
        tick_y = _y(tick, token_max, plot_top, plot_bottom)
        token_label = "0" if tick == 0 else f"{tick / 1000:g}K"
        parts.extend(
            [
                f'<path d="M{token_axis[0]:.1f} {tick_y:.1f}H{token_axis[1]:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>',
                f'<text x="{token_axis[0] - 12:.1f}" y="{tick_y + 5:.1f}" text-anchor="end" fill="{MUTED_COLOR}" font-size="13">{token_label}</text>',
            ]
        )
    parts.extend(
        [
            f'<path d="M{accuracy_axis[0]:.1f} {plot_top:.1f}V{plot_bottom:.1f}H{accuracy_axis[1]:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.2"/>',
            f'<path d="M{token_axis[0]:.1f} {plot_top:.1f}V{plot_bottom:.1f}H{token_axis[1]:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.2"/>',
        ]
    )

    for item, accuracy_center, token_center in zip(
        BENCHMARKS, accuracy_centers, token_centers, strict=True
    ):
        accuracy_x = tuple(accuracy_center - 74 + index * 52 for index in range(3))
        token_x = tuple(token_center - 74 + index * 52 for index in range(3))
        accuracy_values = (
            item.accuracy_default,
            item.accuracy_summary,
            item.accuracy_bcg,
        )
        accuracy_tops: list[float] = []
        for x, value, color in zip(
            accuracy_x,
            accuracy_values,
            (DEFAULT_COLOR, SUMMARY_COLOR, BCG_COLOR),
            strict=True,
        ):
            bar_y = _y(value, accuracy_max, plot_top, plot_bottom)
            accuracy_tops.append(bar_y)
            parts.extend(
                [
                    _rounded_bar(x, bar_y, bar_width, plot_bottom, color),
                    f'<text x="{x + bar_width / 2:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="14" font-weight="700">{value:.2f}%</text>',
                ]
            )
        parts.extend(_benchmark_tick(accuracy_center, item))
        parts.extend(
            _change_arrow(
                accuracy_x[0] + bar_width / 2,
                accuracy_tops[0],
                accuracy_x[2] + bar_width / 2,
                accuracy_tops[2],
                f"+{item.accuracy_bcg - item.accuracy_default:.2f} pp",
                plot_top=plot_top,
                middle_y=accuracy_tops[1],
            )
        )

        default_y = _y(item.tokens_default, token_max, plot_top, plot_bottom)
        parts.extend(
            [
                _rounded_bar(
                    token_x[0], default_y, bar_width, plot_bottom, DEFAULT_COLOR
                ),
                f'<text x="{token_x[0] + bar_width / 2:.1f}" y="{default_y - 6:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="14" font-weight="700">{_format_tokens(item.tokens_default)}</text>',
            ]
        )
        for x, total, memory, color, memory_color, mode in zip(
            token_x[1:],
            (item.tokens_summary, item.tokens_bcg),
            (item.summary_tokens, item.graph_tokens),
            (SUMMARY_COLOR, BCG_COLOR),
            (SUMMARY_MODEL_COLOR, GRAPH_COLOR),
            ("Summary", "BCG"),
            strict=True,
        ):
            total_y = _y(total, token_max, plot_top, plot_bottom)
            agent_tokens = total - memory
            boundary_y = _y(agent_tokens, token_max, plot_top, plot_bottom)
            memory_text_color = "#312e81" if mode == "BCG" else "#115e59"
            parts.extend(
                [
                    f'<rect x="{x:.1f}" y="{boundary_y:.1f}" width="{bar_width:.1f}" height="{plot_bottom - boundary_y:.1f}" fill="{color}"/>',
                    _rounded_bar(x, total_y, bar_width, boundary_y, memory_color),
                    f'<text x="{x + bar_width / 2:.1f}" y="{total_y - 6:.1f}" text-anchor="middle" fill="{TEXT_COLOR}" font-size="14" font-weight="700">{_format_tokens(total)}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{(boundary_y + plot_bottom) / 2 + 5:.1f}" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">{_format_tokens(agent_tokens)}</text>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{(total_y + boundary_y) / 2 + 5:.1f}" text-anchor="middle" fill="{memory_text_color}" font-size="11" font-weight="700">{_format_tokens(memory)}</text>',
                ]
            )
        parts.extend(_benchmark_tick(token_center, item))
        token_tops = (
            default_y,
            _y(item.tokens_summary, token_max, plot_top, plot_bottom),
            _y(item.tokens_bcg, token_max, plot_top, plot_bottom),
        )
        parts.extend(
            _change_arrow(
                token_x[0] + bar_width / 2,
                token_tops[0],
                token_x[2] + bar_width / 2,
                token_tops[2],
                f"−{1 - item.tokens_bcg / item.tokens_default:.1%}",
                plot_top=plot_top,
                middle_y=token_tops[1],
            )
        )

    parts.extend(
        [
            f'<text x="{sum(accuracy_axis) / 2:.1f}" y="575" text-anchor="middle" fill="{MUTED_COLOR}" font-size="14" font-weight="600">Benchmark</text>',
            f'<text x="{sum(token_axis) / 2:.1f}" y="575" text-anchor="middle" fill="{MUTED_COLOR}" font-size="14" font-weight="600">Benchmark</text>',
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
