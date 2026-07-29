#!/usr/bin/env python3
"""Generate the deterministic softmax saturation plot used in Chapter 5."""

from math import exp
from pathlib import Path


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "book"
    / "figures"
    / "05-softmax-saturation.svg"
)

WIDTH = 960
HEIGHT = 640
LEFT = 86
RIGHT = 34
TOP = 122
PLOT_BOTTOM = 444
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = PLOT_BOTTOM - TOP
GAPS = (0, 1, 2, 4, 8, 12)

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d8dee9"
AXIS = "#7a8699"
BLUE = "#225fc7"
ORANGE = "#c66708"
PANEL = "#ffffff"
TABLE_FILL = "#f6f8fb"


def softmax_pair(gap):
    """Return stable softmax probabilities for the scores [gap, 0]."""
    high_exp = exp(float(gap) - float(gap))
    low_exp = exp(0.0 - float(gap))
    total = high_exp + low_exp
    return high_exp / total, low_exp / total


def plot_x(gap):
    return LEFT + gap / GAPS[-1] * PLOT_WIDTH


def plot_y(probability):
    return TOP + (1.0 - probability) * PLOT_HEIGHT


def line(x1, y1, x2, y2, css_class):
    return (
        f'  <line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" class="{css_class}"/>'
    )


def text(x, y, value, css_class, anchor="middle"):
    return (
        f'  <text x="{x:.2f}" y="{y:.2f}" '
        f'class="{css_class}" text-anchor="{anchor}">{value}</text>'
    )


def polyline(points, color):
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'  <polyline points="{coordinates}" fill="none" stroke="{color}" '
        'stroke-width="3.25" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
    )


def main():
    probabilities = [softmax_pair(gap) for gap in GAPS]
    high_points = [
        (plot_x(gap), plot_y(high)) for gap, (high, _) in zip(GAPS, probabilities)
    ]
    low_points = [
        (plot_x(gap), plot_y(low)) for gap, (_, low) in zip(GAPS, probabilities)
    ]

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        '  <title id="title">Softmax saturation as a score gap grows</title>',
        (
            '  <desc id="desc">For the two scores gap and zero, the plot '
            'shows the higher score probability approaching one and the '
            'lower score probability approaching zero at six fixed gaps.'
            "</desc>"
        ),
        "  <style>",
        (
            "    text { font-family: system-ui, -apple-system, BlinkMacSystemFont, "
            "'Segoe UI', sans-serif; fill: "
            + INK
            + "; }"
        ),
        "    .title { font-size: 24px; font-weight: 700; }",
        "    .subtitle { font-size: 14px; fill: " + MUTED + "; }",
        "    .tick { font-size: 12px; fill: " + MUTED + "; }",
        "    .axis-label { font-size: 14px; font-weight: 650; }",
        "    .legend { font-size: 13px; font-weight: 650; }",
        "    .table-label { font-size: 12px; font-weight: 650; }",
        "    .table-value { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 12px; }",
        "    .grid { stroke: " + GRID + "; stroke-width: 1; }",
        "    .axis { stroke: " + AXIS + "; stroke-width: 1.5; }",
        "    .table-line { stroke: " + GRID + "; stroke-width: 1; }",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            LEFT,
            38,
            "A larger score gap pushes softmax toward 0 and 1",
            "title",
            "start",
        ),
        text(
            LEFT,
            67,
            "Fixed inputs [gap, 0]; each pair of probabilities sums to 1",
            "subtitle",
            "start",
        ),
        (
            f'  <line x1="{LEFT:.2f}" y1="94" x2="{LEFT + 34:.2f}" '
            f'y2="94" stroke="{BLUE}" stroke-width="3.25"/>'
        ),
        text(LEFT + 43, 98, "higher score", "legend", "start"),
        (
            f'  <line x1="{LEFT + 178:.2f}" y1="94" '
            f'x2="{LEFT + 212:.2f}" y2="94" stroke="{ORANGE}" '
            'stroke-width="3.25"/>'
        ),
        text(LEFT + 221, 98, "lower score", "legend", "start"),
        text(WIDTH - RIGHT, 112, "probability", "axis-label", "end"),
    ]

    for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_y(probability)
        svg.append(line(LEFT, y, WIDTH - RIGHT, y, "grid"))
        svg.append(text(LEFT - 12, y + 4, f"{probability:g}", "tick", "end"))

    for gap in GAPS:
        x = plot_x(gap)
        svg.append(line(x, TOP, x, PLOT_BOTTOM, "grid"))
        svg.append(text(x, PLOT_BOTTOM + 23, str(gap), "tick"))

    svg.append(line(LEFT, PLOT_BOTTOM, WIDTH - RIGHT, PLOT_BOTTOM, "axis"))
    svg.append(line(LEFT, TOP, LEFT, PLOT_BOTTOM, "axis"))
    svg.append(polyline(high_points, BLUE))
    svg.append(polyline(low_points, ORANGE))

    for points, color in ((high_points, BLUE), (low_points, ORANGE)):
        for x, y in points:
            svg.append(
                f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
                f'fill="{PANEL}" stroke="{color}" stroke-width="2.5"/>'
            )

    svg.append(
        text(WIDTH - RIGHT, PLOT_BOTTOM + 52, "score gap", "axis-label", "end")
    )

    table_x = LEFT
    table_y = 512
    label_width = 132
    value_width = (PLOT_WIDTH - label_width) / len(GAPS)
    row_height = 34
    table_height = 3 * row_height
    svg.append(
        f'  <rect x="{table_x:.2f}" y="{table_y:.2f}" '
        f'width="{PLOT_WIDTH:.2f}" height="{table_height:.2f}" '
        f'rx="5" fill="{TABLE_FILL}" stroke="{GRID}"/>'
    )

    for row in range(1, 3):
        y = table_y + row * row_height
        svg.append(line(table_x, y, table_x + PLOT_WIDTH, y, "table-line"))
    svg.append(
        line(
            table_x + label_width,
            table_y,
            table_x + label_width,
            table_y + table_height,
            "table-line",
        )
    )
    for column in range(1, len(GAPS)):
        x = table_x + label_width + column * value_width
        svg.append(line(x, table_y, x, table_y + table_height, "table-line"))

    row_labels = ("gap", "higher p", "lower p")
    for row, label in enumerate(row_labels):
        svg.append(
            text(
                table_x + 12,
                table_y + row * row_height + 22,
                label,
                "table-label",
                "start",
            )
        )

    for column, (gap, (high, low)) in enumerate(zip(GAPS, probabilities)):
        x = table_x + label_width + (column + 0.5) * value_width
        svg.append(text(x, table_y + 22, str(gap), "table-value"))
        svg.append(text(x, table_y + row_height + 22, f"{high:.6f}", "table-value"))
        svg.append(
            text(x, table_y + 2 * row_height + 22, f"{low:.6f}", "table-value")
        )

    svg.append("</svg>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
