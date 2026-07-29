#!/usr/bin/env python3
"""Plot the selected training and validation reports from the held-out run."""

from html import escape
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "book" / "logs" / "validation-cyberpunk-5000-summary.log"
OUTPUT = ROOT / "book" / "figures" / "15-validation-losses.svg"

WIDTH = 960
HEIGHT = 620
LEFT = 92
RIGHT = 50
TOP = 160
BOTTOM = 500
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = BOTTOM - TOP

FIRST_STEP = 1
FINAL_STEP = 5000
Y_MIN = 0.5
Y_MAX = 4.6

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d6dce5"
TRAINING = "#225fc7"
VALIDATION = "#b84e16"
PLOT_BACKGROUND = "#f8fafc"
PANEL = "#ffffff"

REPORT = re.compile(
    r"^step\s+(\d+)/5000 \| loss ([0-9]+\.[0-9]+) "
    r"\| val ([0-9]+\.[0-9]+)$",
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
        (int(step), float(training), float(validation))
        for step, training, validation in REPORT.findall(
            LOG.read_text(encoding="utf-8")
        )
    ]
    expected = [
        (1, 4.4463, 4.0430),
        (50, 2.4230, 2.4210),
        (250, 1.8567, 1.8301),
        (1000, 1.1218, 1.1182),
        (2000, 0.9470, 0.9790),
        (3000, 0.8414, 0.9320),
        (4000, 0.7987, 0.9105),
        (5000, 0.7638, 0.9023),
    ]

    assert reports == expected
    assert all(
        Y_MIN <= value <= Y_MAX
        for _, training, validation in reports
        for value in (training, validation)
    )
    return reports


def plot_x(step):
    fraction = (step - FIRST_STEP) / (FINAL_STEP - FIRST_STEP)
    return LEFT + fraction * PLOT_WIDTH


def plot_y(loss):
    fraction = (Y_MAX - loss) / (Y_MAX - Y_MIN)
    return TOP + fraction * PLOT_HEIGHT


def points_string(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def main():
    reports = read_reports()
    training_points = [
        (plot_x(step), plot_y(training))
        for step, training, _ in reports
    ]
    validation_points = [
        (plot_x(step), plot_y(validation))
        for step, _, validation in reports
    ]

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        (
            '  <title id="title">Training and held-out losses converge '
            'by step one thousand, then separate</title>'
        ),
        (
            '  <desc id="desc">Eight reports from the committed validation '
            'run. Training loss falls from 4.4463 to 0.7638. Fixed '
            'validation loss falls from 4.0430 to 0.9023. The curves are '
            'nearly equal at step 1000, then training loss keeps falling '
            'faster.</desc>'
        ),
        "  <style>",
        (
            "    text { font-family: system-ui, -apple-system, "
            "BlinkMacSystemFont, 'Segoe UI', sans-serif; "
            f"fill: {INK}; font-style: normal; }}"
        ),
        "    .title { font-size: 28px; font-weight: 700; }",
        f"    .subtitle {{ font-size: 16px; fill: {MUTED}; }}",
        f"    .tick {{ font-size: 15px; fill: {MUTED}; }}",
        "    .axis-label { font-size: 17px; font-weight: 650; }",
        "    .legend { font-size: 16px; font-weight: 650; }",
        "    .callout-title { font-size: 16px; font-weight: 700; }",
        f"    .callout-detail {{ font-size: 15px; fill: {MUTED}; }}",
        f"    .source {{ font-size: 15px; fill: {MUTED}; }}",
        f"    .grid {{ stroke: {GRID}; stroke-width: 1; }}",
        f"    .axis {{ stroke: {INK}; stroke-width: 2; }}",
        (
            f"    .training {{ fill: none; stroke: {TRAINING}; "
            "stroke-width: 3; stroke-linecap: round; "
            "stroke-linejoin: round; }"
        ),
        (
            f"    .validation {{ fill: none; stroke: {VALIDATION}; "
            "stroke-width: 3; stroke-linecap: round; "
            "stroke-linejoin: round; }"
        ),
        f"    .callout-line {{ stroke: {VALIDATION}; stroke-width: 2; }}",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            44,
            42,
            "Two grades converge, then separate",
            "title",
            "start",
        ),
        text(
            44,
            72,
            "Training: one random batch before the update.",
            "subtitle",
            "start",
        ),
        text(
            44,
            98,
            "Held out: the same four fixed batches after the update.",
            "subtitle",
            "start",
        ),
        line(600, 71, 644, 71, "training"),
        text(654, 77, "training", "legend", "start"),
        line(760, 71, 804, 71, "validation"),
        text(814, 77, "held out", "legend", "start"),
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

    for loss in (0.5, 1.0, 2.0, 3.0, 4.0):
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
            text(WIDTH / 2, 558, "training step", "axis-label"),
            text(
                LEFT,
                TOP - 17,
                "mean cross-entropy",
                "axis-label",
                "start",
            ),
            (
                f'  <polyline points="{points_string(training_points)}" '
                'class="training"/>'
            ),
            (
                f'  <polyline points="{points_string(validation_points)}" '
                'class="validation"/>'
            ),
        )
    )

    for x, y in training_points:
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4" '
            f'fill="{TRAINING}" stroke="{PANEL}" stroke-width="1"/>'
        )
    for x, y in validation_points:
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4" '
            f'fill="{VALIDATION}" stroke="{PANEL}" stroke-width="1"/>'
        )

    final_training_x, final_training_y = training_points[-1]
    final_validation_x, final_validation_y = validation_points[-1]
    callout_x = 560
    callout_y = 318

    svg.append(
        line(
            callout_x + 300,
            callout_y + 42,
            final_validation_x - 5,
            final_validation_y,
            "callout-line",
        )
    )
    svg.append(rect(callout_x, callout_y, 300, 66, PANEL, VALIDATION, 8))
    svg.append(
        text(
            callout_x + 16,
            callout_y + 27,
            "Step 5,000",
            "callout-title",
            "start",
        )
    )
    svg.append(
        text(
            callout_x + 16,
            callout_y + 52,
            "training 0.7638, held out 0.9023",
            "callout-detail",
            "start",
        )
    )
    svg.append(
        line(
            final_training_x,
            final_training_y,
            final_validation_x,
            final_validation_y,
            "callout-line",
        )
    )
    svg.append(
        text(
            44,
            598,
            "Source: book/logs/validation-cyberpunk-5000-summary.log",
            "source",
            "start",
        )
    )
    svg.append("</svg>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
