#!/usr/bin/env python3
"""Plot how temperature changes softmax probabilities for fixed logits."""

from math import exp
from pathlib import Path


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "book"
    / "figures"
    / "16-temperature-distribution.svg"
)

WIDTH = 960
HEIGHT = 700
LEFT = 88
RIGHT = 42
TOP = 146
PLOT_BOTTOM = 462
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = PLOT_BOTTOM - TOP

TEMPERATURE_MIN = 0.25
TEMPERATURE_MAX = 3.0
TEMPERATURE_STEP = 0.025
LOGITS = (2.0, 1.0, 0.0)
MARKED_TEMPERATURES = (0.5, 1.0, 2.0)
TOKEN_COUNT = len(LOGITS)

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d6dce5"
AXIS = "#7a8699"
BLUE = "#225fc7"
ORANGE = "#b84e16"
GREEN = "#2d7a55"
PANEL = "#ffffff"
TABLE_FILL = "#f6f8fb"
CURVE_COLORS = (BLUE, ORANGE, GREEN)
CURVE_CLASSES = ("id-zero", "id-one", "id-two")


def softmax_at_temperature(temperature):
    maximum = max(LOGITS)
    exponentials = [
        exp((logit - maximum) / temperature) for logit in LOGITS
    ]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def plot_x(temperature):
    fraction = (
        (temperature - TEMPERATURE_MIN)
        / (TEMPERATURE_MAX - TEMPERATURE_MIN)
    )
    return LEFT + fraction * PLOT_WIDTH


def plot_y(probability):
    return TOP + (1.0 - probability) * PLOT_HEIGHT


def line(x1, y1, x2, y2, css_class):
    return (
        f'  <line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" class="{css_class}"/>'
    )


def text(x, y, value, css_class, anchor="middle"):
    return (
        f'  <text x="{x:.2f}" y="{y:.2f}" class="{css_class}" '
        f'text-anchor="{anchor}">{value}</text>'
    )


def polyline(points, css_class):
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'  <polyline points="{coordinates}" class="{css_class}"/>'
    )


def temperatures():
    count = round(
        (TEMPERATURE_MAX - TEMPERATURE_MIN) / TEMPERATURE_STEP
    )
    return [
        TEMPERATURE_MIN + index * TEMPERATURE_STEP
        for index in range(count + 1)
    ]


def probability_curves():
    samples = [
        (temperature, softmax_at_temperature(temperature))
        for temperature in temperatures()
    ]
    curves = []

    for token_id in range(TOKEN_COUNT):
        curves.append(
            [
                (plot_x(temperature), plot_y(probabilities[token_id]))
                for temperature, probabilities in samples
            ]
        )
    return curves


def start_svg():
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        (
            '  <title id="title">Temperature changes the probability '
            'spread for logits two, one, and zero</title>'
        ),
        (
            '  <desc id="desc">Three deterministic softmax curves from '
            'temperature 0.25 through 3. The highest-logit probability '
            'falls as temperature rises, while the other two rise. The '
            'ranking never changes.</desc>'
        ),
        "  <style>",
        (
            "    text { font-family: system-ui, -apple-system, "
            "BlinkMacSystemFont, 'Segoe UI', sans-serif; "
            f"fill: {INK}; }}"
        ),
        "    .title { font-size: 27px; font-weight: 700; }",
        f"    .subtitle {{ font-size: 15px; fill: {MUTED}; }}",
        f"    .tick {{ font-size: 14px; fill: {MUTED}; }}",
        "    .axis-label { font-size: 16px; font-weight: 650; }",
        "    .legend { font-size: 15px; font-weight: 650; }",
        "    .table-label { font-size: 14px; font-weight: 650; }",
        (
            "    .table-value { font-family: ui-monospace, "
            "SFMono-Regular, Menlo, Consolas, monospace; "
            "font-size: 14px; }"
        ),
        f"    .grid {{ stroke: {GRID}; stroke-width: 1; }}",
        (
            f"    .marker {{ stroke: {AXIS}; stroke-width: 1; "
            "stroke-dasharray: 5 5; }"
        ),
        f"    .axis {{ stroke: {AXIS}; stroke-width: 1.5; }}",
        (
            f"    .id-zero {{ fill: none; stroke: {BLUE}; "
            "stroke-width: 3.2; stroke-linecap: round; }"
        ),
        (
            f"    .id-one {{ fill: none; stroke: {ORANGE}; "
            "stroke-width: 3.2; stroke-linecap: round; }"
        ),
        (
            f"    .id-two {{ fill: none; stroke: {GREEN}; "
            "stroke-width: 3.2; stroke-linecap: round; }"
        ),
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            LEFT,
            40,
            "A positive divisor changes the probability spread",
            "title",
            "start",
        ),
        text(
            LEFT,
            69,
            "Illustrative logits [2, 1, 0]; centered softmax at each temperature",
            "subtitle",
            "start",
        ),
    ]


def draw_legend_and_axes(svg):
    legend_x = (LEFT, LEFT + 176, LEFT + 352)
    legend_labels = (
        "id 0, logit 2",
        "id 1, logit 1",
        "id 2, logit 0",
    )

    for x, label, color in zip(legend_x, legend_labels, CURVE_COLORS):
        svg.append(
            f'  <line x1="{x:.2f}" y1="104" x2="{x + 38:.2f}" '
            f'y2="104" stroke="{color}" stroke-width="3.2"/>'
        )
        svg.append(text(x + 47, 109, label, "legend", "start"))

    for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_y(probability)
        svg.append(line(LEFT, y, WIDTH - RIGHT, y, "grid"))
        svg.append(
            text(LEFT - 13, y + 5, f"{probability:g}", "tick", "end")
        )

    tick_temperatures = (0.25, 0.5, 1.0, 2.0, 3.0)
    for temperature in tick_temperatures:
        x = plot_x(temperature)
        css_class = (
            "marker"
            if temperature in MARKED_TEMPERATURES
            else "grid"
        )
        svg.append(line(x, TOP, x, PLOT_BOTTOM, css_class))
        svg.append(
            text(x, PLOT_BOTTOM + 25, f"{temperature:g}", "tick")
        )

    svg.extend(
        (
            line(LEFT, PLOT_BOTTOM, WIDTH - RIGHT, PLOT_BOTTOM, "axis"),
            line(LEFT, TOP, LEFT, PLOT_BOTTOM, "axis"),
            text(
                LEFT,
                TOP - 16,
                "probability",
                "axis-label",
                "start",
            ),
            text(
                WIDTH - RIGHT,
                PLOT_BOTTOM + 52,
                "temperature",
                "axis-label",
                "end",
            ),
        )
    )


def draw_curves_and_markers(svg, curves):
    for points, css_class in zip(curves, CURVE_CLASSES):
        svg.append(polyline(points, css_class))

    for temperature in MARKED_TEMPERATURES:
        probabilities = softmax_at_temperature(temperature)
        for probability, color in zip(probabilities, CURVE_COLORS):
            svg.append(
                f'  <circle cx="{plot_x(temperature):.2f}" '
                f'cy="{plot_y(probability):.2f}" r="4.5" '
                f'fill="{PANEL}" stroke="{color}" stroke-width="2.5"/>'
            )


def draw_probability_table(svg):
    table_x = LEFT
    table_y = 540
    table_width = PLOT_WIDTH
    row_height = 32
    label_width = 174
    value_width = (table_width - label_width) / len(MARKED_TEMPERATURES)
    table_height = row_height * (TOKEN_COUNT + 1)
    svg.append(
        f'  <rect x="{table_x:.2f}" y="{table_y:.2f}" '
        f'width="{table_width:.2f}" height="{table_height:.2f}" '
        f'rx="5" fill="{TABLE_FILL}" stroke="{GRID}"/>'
    )

    for row in range(1, TOKEN_COUNT + 1):
        y = table_y + row * row_height
        svg.append(line(table_x, y, table_x + table_width, y, "grid"))

    svg.append(
        line(
            table_x + label_width,
            table_y,
            table_x + label_width,
            table_y + table_height,
            "grid",
        )
    )
    for column in range(1, len(MARKED_TEMPERATURES)):
        x = table_x + label_width + column * value_width
        svg.append(
            line(x, table_y, x, table_y + table_height, "grid")
        )

    row_labels = ("temperature", "p(id 0)", "p(id 1)", "p(id 2)")
    for row, label in enumerate(row_labels):
        svg.append(
            text(
                table_x + 13,
                table_y + row * row_height + 21,
                label,
                "table-label",
                "start",
            )
        )

    marked_probabilities = [
        softmax_at_temperature(temperature)
        for temperature in MARKED_TEMPERATURES
    ]
    for column, temperature in enumerate(MARKED_TEMPERATURES):
        x = (
            table_x
            + label_width
            + (column + 0.5) * value_width
        )
        svg.append(
            text(
                x,
                table_y + 21,
                f"{temperature:.1f}",
                "table-value",
            )
        )
        for token_id in range(TOKEN_COUNT):
            svg.append(
                text(
                    x,
                    table_y + (token_id + 1) * row_height + 21,
                    f"{marked_probabilities[column][token_id]:.4f}",
                    "table-value",
                )
            )


def write_svg(svg):
    svg.append("</svg>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("\n".join(svg) + "\n")


def main():
    curves = probability_curves()
    svg = start_svg()

    draw_legend_and_axes(svg)
    draw_curves_and_markers(svg, curves)
    draw_probability_table(svg)
    write_svg(svg)


if __name__ == "__main__":
    main()
