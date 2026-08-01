#!/usr/bin/env python3
"""Plot every recorded loss report from the 5,000-step showcase run."""

from html import escape
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "book" / "logs" / "train-cyberpunk-5000.log"
OUTPUT = ROOT / "book" / "figures" / "01-training-loss.svg"

WIDTH = 960
HEIGHT = 640
LEFT = 88
RIGHT = 40
TOP = 150
BOTTOM = 510
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = BOTTOM - TOP

FIRST_STEP = 1
FINAL_STEP = 5000
Y_MIN = 0.5
Y_MAX = 4.5

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d6dce5"
BLUE = "#225fc7"
GREEN = "#147d64"
PLOT_BACKGROUND = "#f8fafc"
PANEL = "#ffffff"

REPORT = re.compile(
    r"^step\s+(\d+)/5000 \| loss ([0-9]+\.[0-9]+) \|",
    re.MULTILINE,
)


def text(x, y, value, css_class, anchor="middle"):
    return (
        f'  <text x="{x:.2f}" y="{y:.2f}" class="{css_class}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, css_class):
    return (
        f'  <line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" class="{css_class}"/>'
    )


def rect(x, y, width, height, fill, stroke, radius=8):
    return (
        f'  <rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="{radius}" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )


def read_reports():
    reports = [
        (int(step), float(loss))
        for step, loss in REPORT.findall(LOG.read_text(encoding="utf-8"))
    ]
    expected_steps = [FIRST_STEP, *range(50, FINAL_STEP + 1, 50)]

    assert len(reports) == 101
    assert [step for step, _ in reports] == expected_steps
    assert reports[0] == (FIRST_STEP, 4.4395)
    assert reports[-1] == (FINAL_STEP, 0.7668)
    assert all(Y_MIN <= loss <= Y_MAX for _, loss in reports)
    return reports


def plot_x(step):
    fraction = (step - FIRST_STEP) / (FINAL_STEP - FIRST_STEP)
    return LEFT + fraction * PLOT_WIDTH


def plot_y(loss):
    fraction = (Y_MAX - loss) / (Y_MAX - Y_MIN)
    return TOP + fraction * PLOT_HEIGHT


def start_svg():
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        (
            '  <title id="title">Recorded training loss across five thousand '
            'steps</title>'
        ),
        (
            '  <desc id="desc">A line connects all one hundred one loss '
            'reports in the committed showcase log. Loss starts at 4.4395 '
            'on step 1 and ends at 0.7668 on step 5000. The reports cover '
            'training only. Training ends by saving a checkpoint. A later '
            'sample command loads it and does not update the parameters.'
            '</desc>'
        ),
        "  <style>",
        (
            "    text { font-family: system-ui, -apple-system, BlinkMacSystemFont, "
            "'Segoe UI', sans-serif; fill: "
            + INK
            + "; font-style: normal; }"
        ),
        "    .title { font-size: 28px; font-weight: 700; }",
        "    .subtitle { font-size: 16px; fill: " + MUTED + "; }",
        "    .tick { font-size: 16px; fill: " + MUTED + "; }",
        "    .axis-label { font-size: 17px; font-weight: 650; }",
        "    .callout-title { font-size: 17px; font-weight: 700; }",
        "    .callout-detail { font-size: 16px; fill: " + MUTED + "; }",
        "    .source { font-size: 16px; fill: " + MUTED + "; }",
        f"    .grid {{ stroke: {GRID}; stroke-width: 1; }}",
        f"    .axis {{ stroke: {INK}; stroke-width: 2; }}",
        f"    .curve {{ fill: none; stroke: {BLUE}; stroke-width: 3; "
        "stroke-linecap: round; stroke-linejoin: round; }",
        f"    .callout-line {{ stroke: {GREEN}; stroke-width: 2; }}",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            44,
            42,
            "The training penalty falls over 5,000 steps",
            "title",
            "start",
        ),
        text(
            44,
            73,
            "Each dot is one recorded report. The line connects all 101 reports.",
            "subtitle",
            "start",
        ),
        text(
            44,
            99,
            (
                "Training ends by saving the checkpoint. A later sample "
                "command loads it without updating the model."
            ),
            "subtitle",
            "start",
        ),
        rect(
            LEFT,
            TOP,
            PLOT_WIDTH,
            PLOT_HEIGHT,
            PLOT_BACKGROUND,
            GRID,
            4,
        ),
    ]


def draw_axes(svg):
    for loss in (0.5, 1.0, 2.0, 3.0, 4.0, 4.5):
        y = plot_y(loss)
        svg.append(line(LEFT, y, WIDTH - RIGHT, y, "grid"))
        svg.append(text(LEFT - 14, y + 5, f"{loss:.1f}", "tick", "end"))

    for step in (FIRST_STEP, 1000, 2000, 3000, 4000, FINAL_STEP):
        x = plot_x(step)
        svg.append(line(x, TOP, x, BOTTOM, "grid"))
        svg.append(text(x, BOTTOM + 27, f"{step:,}", "tick"))

    svg.extend(
        (
            line(LEFT, BOTTOM, WIDTH - RIGHT, BOTTOM, "axis"),
            line(LEFT, TOP, LEFT, BOTTOM, "axis"),
            text(
                WIDTH / 2,
                558,
                "training step",
                "axis-label",
            ),
            text(
                LEFT,
                TOP - 17,
                "recorded loss",
                "axis-label",
                "start",
            ),
        )
    )


def draw_loss_curve(svg, points):
    point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    svg.append(f'  <polyline points="{point_string}" class="curve"/>')
    for x, y in points:
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" '
            f'fill="{BLUE}" stroke="{PANEL}" stroke-width="0.8"/>'
        )


def draw_callouts(svg, points):
    first_x, first_y = points[0]
    svg.append(line(first_x + 4, first_y + 3, 116, 164, "callout-line"))
    svg.append(rect(116, 146, 190, 64, PANEL, GREEN, 8))
    svg.append(text(132, 173, "before adjustment 1", "callout-title", "start"))
    svg.append(text(132, 197, "step 1: loss 4.4395", "callout-detail", "start"))

    final_x, final_y = points[-1]
    svg.append(line(866, 400, final_x - 4, final_y - 3, "callout-line"))
    svg.append(rect(668, 352, 198, 64, PANEL, GREEN, 8))
    svg.append(text(684, 379, "step 5,000 report", "callout-title", "start"))
    svg.append(text(684, 403, "final loss 0.7668", "callout-detail", "start"))


def write_svg(svg):
    svg.append(
        text(
            44,
            612,
            "Source: book/logs/train-cyberpunk-5000.log",
            "source",
            "start",
        )
    )
    svg.append("</svg>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(svg) + "\n")


def main():
    reports = read_reports()
    points = [(plot_x(step), plot_y(loss)) for step, loss in reports]

    assert all(LEFT <= x <= WIDTH - RIGHT for x, _ in points)
    assert all(TOP <= y <= BOTTOM for _, y in points)

    svg = start_svg()
    draw_axes(svg)
    draw_loss_curve(svg, points)
    draw_callouts(svg, points)
    write_svg(svg)


if __name__ == "__main__":
    main()
