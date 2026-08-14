#!/usr/bin/env python3
"""Render the README token-horizon chart from benchmark trajectories."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_COLOR = "#64748b"
BCG_COLOR = "#5b5bd6"
SAVINGS_COLOR = "#aaa8f4"
TEXT_COLOR = "#0f172a"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"


@dataclass(frozen=True)
class BenchmarkSeries:
    title: str
    task_count: int
    default: tuple[float, ...]
    bcg: tuple[float, ...]


def _call_tokens(path: Path, *, include_graph: bool) -> list[int]:
    tokens: list[int] = []
    timestamps_ms: list[float] = []
    graph_by_label: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            message = event.get("message") or {}
            if event.get("type") == "message_end" and message.get("role") == "assistant":
                usage = message.get("usage") or {}
                value = usage.get("totalTokens")
                timestamp = message.get("timestamp")
                if isinstance(value, (int, float)) and value > 0:
                    if not isinstance(timestamp, (int, float)):
                        raise ValueError(f"Assistant call is missing a timestamp in {path}")
                    tokens.append(int(value))
                    timestamps_ms.append(float(timestamp))
            elif event.get("type") == "graph_usage":
                graph_by_label = (event.get("usage") or {}).get("by_label") or {}

    if not include_graph:
        return tokens
    if not graph_by_label:
        raise ValueError(f"No Graph usage event found in {path}")

    trace_path = path.parent.parent / "graph-contexts" / path.name
    trace_times_ms: dict[int, float] = {}
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            turn_index = int(event["streamTurnIndex"])
            timestamp_ms = (
                datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
                * 1000
            )
            trace_times_ms[turn_index] = min(
                trace_times_ms.get(turn_index, timestamp_ms), timestamp_ms
            )

    trace_indices = sorted(trace_times_ms)
    graph_tokens_by_call = [0] * len(tokens)
    for label, usage in graph_by_label.items():
        batch_match = re.match(r"^t(\d+)\.extract:tool_batch:(\d+)$", label)
        if batch_match:
            source_turn = int(batch_match.group(1)) + int(batch_match.group(2)) - 1
        else:
            turn_match = re.match(r"^t(\d+)\.", label)
            if not turn_match:
                # Embedding usage is not an LLM token count; final post-task
                # construction is intentionally excluded from benchmark totals.
                continue
            source_turn = int(turn_match.group(1))

        trace_position = bisect_left(trace_indices, source_turn)
        if trace_position == len(trace_indices):
            raise ValueError(f"Cannot align Graph label {label!r} in {path}")
        graph_timestamp_ms = trace_times_ms[trace_indices[trace_position]]
        call_index = bisect_left(timestamps_ms, graph_timestamp_ms)
        if call_index == len(tokens):
            raise ValueError(f"Graph label {label!r} occurs after the last Agent call in {path}")
        graph_tokens_by_call[call_index] += int(usage.get("total_tokens", 0))

    return [agent + graph for agent, graph in zip(tokens, graph_tokens_by_call, strict=True)]


def _mean_cumulative_tokens(
    trajectory_dir: Path, horizon: int, *, include_graph: bool = False
) -> tuple[tuple[float, ...], int]:
    trajectories = sorted(trajectory_dir.glob("*.jsonl"))
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found under {trajectory_dir}")

    cumulative_by_task: list[list[int]] = []
    for path in trajectories:
        running = 0
        cumulative: list[int] = []
        for value in _call_tokens(path, include_graph=include_graph):
            running += value
            cumulative.append(running)
        if not cumulative:
            raise ValueError(f"No completed assistant model calls found in {path}")
        cumulative_by_task.append(cumulative)

    means: list[float] = []
    for call_index in range(horizon):
        total = sum(
            cumulative[min(call_index, len(cumulative) - 1)]
            for cumulative in cumulative_by_task
        )
        means.append(total / len(cumulative_by_task))
    return tuple(means), len(cumulative_by_task)


def load_benchmark(result_root: Path, benchmark: str, title: str, horizon: int) -> BenchmarkSeries:
    default, default_count = _mean_cumulative_tokens(
        result_root / benchmark / "default" / "trajectories", horizon
    )
    bcg, bcg_count = _mean_cumulative_tokens(
        result_root / benchmark / "bcg" / "trajectories",
        horizon,
        include_graph=True,
    )
    if default_count != bcg_count:
        raise ValueError(
            f"{title} mode counts differ: Default={default_count}, BCG={bcg_count}"
        )
    return BenchmarkSeries(title, default_count, default, bcg)


def _points(
    values: tuple[float, ...],
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    y_max: float,
) -> list[tuple[float, float]]:
    denominator = max(1, len(values) - 1)
    return [
        (
            left + (right - left) * index / denominator,
            bottom - (bottom - top) * value / y_max,
        )
        for index, value in enumerate(values)
    ]


def _smooth_path(
    points: list[tuple[float, float]], *, move_command: str = "M"
) -> str:
    """Render a shape-preserving cubic curve through every observed point."""
    if not points:
        return ""
    if len(points) == 1:
        x, y = points[0]
        return f"{move_command}{x:.1f},{y:.1f}"

    deltas = [
        (right_y - left_y) / (right_x - left_x)
        for (left_x, left_y), (right_x, right_y) in zip(
            points, points[1:], strict=False
        )
    ]
    tangents = [deltas[0]]
    for left_delta, right_delta in zip(deltas, deltas[1:], strict=False):
        if left_delta * right_delta <= 0:
            tangents.append(0.0)
        else:
            tangents.append((left_delta + right_delta) / 2)
    tangents.append(deltas[-1])

    # Fritsch-Carlson limiting prevents the interpolant from overshooting a
    # monotone interval, which matters for cumulative-token curves.
    for index, delta in enumerate(deltas):
        if delta == 0:
            tangents[index] = tangents[index + 1] = 0.0
            continue
        alpha = tangents[index] / delta
        beta = tangents[index + 1] / delta
        magnitude = alpha * alpha + beta * beta
        if magnitude > 9:
            scale = 3 / math.sqrt(magnitude)
            tangents[index] = scale * alpha * delta
            tangents[index + 1] = scale * beta * delta

    first_x, first_y = points[0]
    commands = [f"{move_command}{first_x:.1f},{first_y:.1f}"]
    for index, ((left_x, left_y), (right_x, right_y)) in enumerate(
        zip(points, points[1:], strict=False)
    ):
        width = right_x - left_x
        commands.append(
            "C"
            f"{left_x + width / 3:.1f},{left_y + tangents[index] * width / 3:.1f} "
            f"{right_x - width / 3:.1f},{right_y - tangents[index + 1] * width / 3:.1f} "
            f"{right_x:.1f},{right_y:.1f}"
        )
    return " ".join(commands)


def _format_k(value: float) -> str:
    return f"{value / 1000:.1f}K"


def _break_even_call(default: tuple[float, ...], bcg: tuple[float, ...]) -> float:
    """Return the interpolated call where BCG moves below Default."""
    previous_delta = bcg[0] - default[0]
    for index in range(1, min(len(default), len(bcg))):
        delta = bcg[index] - default[index]
        if previous_delta > 0 >= delta:
            fraction = previous_delta / (previous_delta - delta)
            return index + fraction
        previous_delta = delta
    raise ValueError("BCG does not reach break-even within the plotted horizon")


def _axis_scale(values: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
    """Choose a compact, panel-local axis with readable round-number ticks."""
    maximum = max(values)
    raw_step = maximum * 1.08 / 7
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    multiple = next(value for value in (1, 2, 2.5, 5, 10) if normalized <= value)
    step = multiple * magnitude
    y_max = math.ceil(maximum * 1.08 / step) * step
    ticks = tuple(float(value) for value in range(0, int(y_max) + 1, int(step)))
    return y_max, ticks


def render_svg(series: tuple[BenchmarkSeries, ...], output: Path) -> None:
    width, height = 1280, 620
    plot_top, plot_bottom = 166.0, 510.0
    plot_width = 525.0
    plot_lefts = (88.0, 703.0)
    plot_rights = tuple(left + plot_width for left in plot_lefts)
    horizon = len(series[0].default)
    x_ticks = (1, 5, 10, 15, 20, horizon)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Cumulative total tokens across the task horizon</title>',
        '<desc id="desc">Full-dataset comparison of Default and BCG cumulative total tokens by model call for BrowseComp and BrowseComp-ZH. BCG includes Agent and Graph Construction tokens.</desc>',
        '<rect width="1280" height="620" fill="#ffffff"/>',
        '<g font-family="Inter,Arial,sans-serif">',
        f'<text x="640" y="39" text-anchor="middle" fill="{TEXT_COLOR}" font-size="27" font-weight="700">Cumulative total tokens across the task horizon</text>',
        f'<text x="640" y="66" text-anchor="middle" fill="{MUTED_COLOR}" font-size="14">Full-dataset mean · BCG includes Agent + Graph Construction · completed task totals carried forward</text>',
        f'<path d="M474 96H510" stroke="{DEFAULT_COLOR}" stroke-width="5" stroke-linecap="round"/><text x="522" y="102" fill="{TEXT_COLOR}" font-size="16" font-weight="600">Default</text>',
        f'<path d="M644 96H680" stroke="{BCG_COLOR}" stroke-width="5" stroke-linecap="round"/><text x="692" y="102" fill="{TEXT_COLOR}" font-size="16" font-weight="600">BCG</text>',
        f'<text x="24" y="338" transform="rotate(-90 24 338)" text-anchor="middle" fill="{TEXT_COLOR}" font-size="17" font-weight="600">Mean cumulative total tokens / task</text>',
    ]

    for panel_index, item in enumerate(series):
        left = plot_lefts[panel_index]
        right = plot_rights[panel_index]
        y_max, y_ticks = _axis_scale(item.default + item.bcg)
        default_points = _points(
            item.default,
            left=left,
            right=right,
            top=plot_top,
            bottom=plot_bottom,
            y_max=y_max,
        )
        bcg_points = _points(
            item.bcg,
            left=left,
            right=right,
            top=plot_top,
            bottom=plot_bottom,
            y_max=y_max,
        )
        reduction = 1 - item.bcg[-1] / item.default[-1]
        break_even_call = _break_even_call(item.default, item.bcg)
        break_even_x = left + (right - left) * (break_even_call - 1) / max(
            1, horizon - 1
        )

        parts.extend(
            [
                f'<text x="{left:.0f}" y="132" fill="{TEXT_COLOR}" font-size="21" font-weight="700">{html.escape(item.title)}</text>',
                f'<text x="{right:.0f}" y="132" text-anchor="end" fill="{MUTED_COLOR}" font-size="13">{item.task_count:,} tasks / mode</text>',
            ]
        )

        for tick in y_ticks:
            y = plot_bottom - (plot_bottom - plot_top) * tick / y_max
            parts.append(
                f'<path d="M{left:.1f} {y:.1f}H{right:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            label = "0" if tick == 0 else f"{tick / 1000:g}K"
            parts.append(
                f'<text x="{left - 13:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{MUTED_COLOR}" font-size="13">{label}</text>'
            )

        parts.append(
            f'<path d="M{left:.1f} {plot_top:.1f}V{plot_bottom:.1f}H{right:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.2"/>'
        )
        for tick in x_ticks:
            x = left + (right - left) * (tick - 1) / max(1, horizon - 1)
            parts.extend(
                [
                    f'<path d="M{x:.1f} {plot_bottom:.1f}v7" stroke="{MUTED_COLOR}" stroke-width="1"/>',
                    f'<text x="{x:.1f}" y="{plot_bottom + 27:.1f}" text-anchor="middle" fill="{MUTED_COLOR}" font-size="13">{tick}</text>',
                ]
            )

        break_even_label = f"Break-even at ~{round(break_even_call)} calls"
        parts.extend(
            [
                f'<path d="M{break_even_x:.1f} {plot_top:.1f}V{plot_bottom:.1f}" stroke="{BCG_COLOR}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="5 5"/>',
                f'<rect x="{break_even_x - 79:.1f}" y="{plot_top + 13:.1f}" width="158" height="26" rx="13" fill="#ffffff" fill-opacity="0.94" stroke="#c7d2fe"/>',
                f'<text x="{break_even_x:.1f}" y="{plot_top + 31:.1f}" text-anchor="middle" fill="#4338ca" font-size="12" font-weight="700">{break_even_label}</text>',
            ]
        )

        savings_path = (
            _smooth_path(default_points)
            + " "
            + _smooth_path(list(reversed(bcg_points)), move_command="L")
            + " Z"
        )
        parts.extend(
            [
                f'<path d="{savings_path}" fill="{SAVINGS_COLOR}" fill-opacity="0.20"/>',
                f'<path d="{_smooth_path(default_points)}" fill="none" stroke="{DEFAULT_COLOR}" stroke-width="4.5" stroke-linecap="round"/>',
                f'<path d="{_smooth_path(bcg_points)}" fill="none" stroke="{BCG_COLOR}" stroke-width="4.5" stroke-linecap="round"/>',
            ]
        )

        for values, points, color, label_offset in (
            (item.default, default_points, DEFAULT_COLOR, -15),
            (item.bcg, bcg_points, BCG_COLOR, 25),
        ):
            x, y = points[-1]
            parts.extend(
                [
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>',
                    f'<text x="{x - 10:.1f}" y="{y + label_offset:.1f}" text-anchor="end" fill="{color}" font-size="16" font-weight="700">{_format_k(values[-1])}</text>',
                ]
            )

        badge_x = left + 334
        badge_y = plot_bottom - 58
        parts.extend(
            [
                f'<rect x="{badge_x:.1f}" y="{badge_y:.1f}" width="174" height="31" rx="15.5" fill="#eef2ff"/>',
                f'<text x="{badge_x + 87:.1f}" y="{badge_y + 21:.1f}" text-anchor="middle" fill="#4338ca" font-size="13" font-weight="700">{reduction:.0%} fewer total tokens</text>',
            ]
        )

        parts.append(
            f'<text x="{(left + right) / 2:.1f}" y="580" text-anchor="middle" fill="{TEXT_COLOR}" font-size="17" font-weight="600">Agent model calls</text>'
        )

    parts.extend(["</g>", "</svg>"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browsecomp-results", type=Path, required=True)
    parser.add_argument("--browsecomp-zh-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assert/token_cost.svg"))
    parser.add_argument("--horizon", type=int, default=22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = (
        load_benchmark(args.browsecomp_results, "browsecomp", "BrowseComp", args.horizon),
        load_benchmark(
            args.browsecomp_zh_results,
            "browsecomp_zh",
            "BrowseComp-ZH",
            args.horizon,
        ),
    )
    render_svg(series, args.output)


if __name__ == "__main__":
    main()
