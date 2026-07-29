#!/usr/bin/env python3
"""Generate the deterministic GELU curve used in Chapter 5."""

from math import tanh
from pathlib import Path


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "book"
    / "figures"
    / "05-gelu.svg"
)

GELU_SQRT_2_OVER_PI = 0.7978845608
GELU_CUBIC_COEFF = 0.044715

WIDTH = 960
HEIGHT = 560
LEFT = 86
RIGHT = 34
TOP = 104
BOTTOM = 68
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = HEIGHT - TOP - BOTTOM
X_MIN = -4.0
X_MAX = 4.0
Y_MIN = -0.5
Y_MAX = 4.2

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d8dee9"
AXIS = "#7a8699"
BLUE = "#225fc7"
PANEL = "#ffffff"


def gelu(value):
    """Match the tanh approximation and constants in src/ops.c."""
    inner = GELU_SQRT_2_OVER_PI * (
        value + GELU_CUBIC_COEFF * value * value * value
    )
    return 0.5 * value * (1.0 + tanh(inner))


def plot_x(value):
    return LEFT + (value - X_MIN) / (X_MAX - X_MIN) * PLOT_WIDTH


def plot_y(value):
    return TOP + (Y_MAX - value) / (Y_MAX - Y_MIN) * PLOT_HEIGHT


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


def main():
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        '  <title id="title">The GELU activation curve</title>',
        (
            '  <desc id="desc">A plot of the GELU tanh approximation from '
            'negative four to four, with the lab inputs negative one, zero, '
            'and one marked.</desc>'
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
        "    .point-label { font-size: 13px; font-weight: 650; }",
        "    .grid { stroke: " + GRID + "; stroke-width: 1; }",
        "    .axis { stroke: " + AXIS + "; stroke-width: 1.5; }",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            LEFT,
            38,
            "GELU bends near zero and approaches the input for positive x",
            "title",
            "start",
        ),
        text(
            LEFT,
            67,
            (
                "0.5x(1 + tanh(0.7978845608"
                "(x + 0.044715x^3)))"
            ),
            "subtitle",
            "start",
        ),
        text(LEFT, TOP - 15, "output GELU(x)", "axis-label", "start"),
    ]

    for value in (-0.5, 0.0, 1.0, 2.0, 3.0, 4.0):
        y = plot_y(value)
        css_class = "axis" if value == 0.0 else "grid"
        svg.append(line(LEFT, y, WIDTH - RIGHT, y, css_class))
        label = f"{value:g}"
        svg.append(text(LEFT - 12, y + 4, label, "tick", "end"))

    for value in range(-4, 5):
        x = plot_x(float(value))
        css_class = "axis" if value == 0 else "grid"
        svg.append(line(x, TOP, x, HEIGHT - BOTTOM, css_class))
        svg.append(text(x, HEIGHT - BOTTOM + 23, str(value), "tick"))

    samples = 400
    path_parts = []
    for index in range(samples + 1):
        value = X_MIN + (X_MAX - X_MIN) * index / samples
        command = "M" if index == 0 else "L"
        path_parts.append(
            f"{command}{plot_x(value):.2f},{plot_y(gelu(value)):.2f}"
        )
    svg.append(
        f'  <path d="{" ".join(path_parts)}" fill="none" stroke="{BLUE}" '
        'stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )

    lab_points = [
        (-1.0, -32.0, 34.0, "middle"),
        (0.0, 14.0, -14.0, "start"),
        (1.0, 16.0, -12.0, "start"),
    ]
    for value, dx, dy, anchor in lab_points:
        result = gelu(value)
        x = plot_x(value)
        y = plot_y(result)
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" '
            f'fill="{PANEL}" stroke="{BLUE}" stroke-width="3"/>'
        )
        display_result = "0" if value == 0.0 else f"{result:.6f}"
        svg.append(
            text(
                x + dx,
                y + dy,
                f"({value:g}, {display_result})",
                "point-label",
                anchor,
            )
        )

    svg.extend(
        [
            text(WIDTH - RIGHT, HEIGHT - 20, "input x", "axis-label", "end"),
            "</svg>",
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
