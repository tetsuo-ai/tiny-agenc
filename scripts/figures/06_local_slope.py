#!/usr/bin/env python3
"""Generate the deterministic local-slope plot used in Chapter 6."""

from pathlib import Path


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "book"
    / "figures"
    / "06-local-slope.svg"
)

WIDTH = 960
HEIGHT = 690
LEFT = 86
RIGHT = 34
TOP = 142
PLOT_BOTTOM = 510
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = PLOT_BOTTOM - TOP
X_MIN = 0.0
X_MAX = 3.2
Y_MIN = 0.0
Y_MAX = 10.5

ANCHOR_X = 2.0
ANCHOR_Y = 4.0
SECANTS = (
    (3.0, 9.0, 5.0),
    (2.5, 6.25, 4.5),
)
LOCAL_SLOPE = 4.0

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d8dee9"
AXIS = "#7a8699"
BLUE = "#225fc7"
ORANGE = "#c66708"
PURPLE = "#7856b3"
GREEN = "#147d64"
PANEL = "#ffffff"
CARD_FILL = "#f6f8fb"


def square(value):
    return value * value


def secant_slope(x0, y0, x1, y1):
    return (y1 - y0) / (x1 - x0)


def tangent_y(value):
    return ANCHOR_Y + LOCAL_SLOPE * (value - ANCHOR_X)


def verify_fixture():
    """Keep every labeled number tied to y=x^2 and the stated slopes."""
    assert square(ANCHOR_X) == ANCHOR_Y
    assert LOCAL_SLOPE == 2.0 * ANCHOR_X
    assert tangent_y(ANCHOR_X) == ANCHOR_Y

    for endpoint_x, endpoint_y, expected_slope in SECANTS:
        assert square(endpoint_x) == endpoint_y
        measured_slope = secant_slope(
            ANCHOR_X,
            ANCHOR_Y,
            endpoint_x,
            endpoint_y,
        )
        assert measured_slope == expected_slope

    assert SECANTS[0][2] > SECANTS[1][2] > LOCAL_SLOPE
    assert square(2.5) - tangent_y(2.5) == 0.25


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


def legend_item(x, y, color, label, dash=None):
    dash_attribute = (
        f' stroke-dasharray="{dash}"' if dash is not None else ""
    )
    return [
        (
            f'  <line x1="{x:.2f}" y1="{y:.2f}" '
            f'x2="{x + 32:.2f}" y2="{y:.2f}" stroke="{color}" '
            f'stroke-width="3.25"{dash_attribute}/>'
        ),
        text(x + 41, y + 4, label, "legend", "start"),
    ]


def start_svg():
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        (
            '  <title id="title">Two-point slopes close in on the local '
            'slope at (2, 4)</title>'
        ),
        (
            '  <desc id="desc">A graph of y equals x squared with the '
            'point (2, 4) fixed. The line to (3, 9) has slope 5. '
            'The closer line to (2.5, 6.25) has slope 4.5. The line '
            'through the fixed point has local slope 4.</desc>'
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
        "    .legend { font-size: 12px; font-weight: 650; }",
        "    .point-label { font-size: 12px; font-weight: 650; }",
        "    .curve-label { font-size: 13px; font-weight: 700; fill: "
        + BLUE
        + "; }",
        "    .card-title { font-size: 12px; font-weight: 650; }",
        "    .card-value { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 13px; font-weight: 700; }",
        "    .grid { stroke: " + GRID + "; stroke-width: 1; }",
        "    .axis { stroke: " + AXIS + "; stroke-width: 1.5; }",
        "    .far-secant { stroke: "
        + ORANGE
        + "; stroke-width: 3.25; stroke-dasharray: 10 7; }",
        "    .near-secant { stroke: "
        + PURPLE
        + "; stroke-width: 3.25; stroke-dasharray: 5 5; }",
        "    .tangent { stroke: "
        + GREEN
        + "; stroke-width: 3.5; stroke-linecap: round; }",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            LEFT,
            38,
            "Move a second point toward (2, 4) and watch the slope settle",
            "title",
            "start",
        ),
        text(
            LEFT,
            67,
            "Curve y = x^2; each line compares rise in y with run in x",
            "subtitle",
            "start",
        ),
    ]


def draw_legend_and_axes(svg):
    svg.extend(legend_item(LEFT, 102, BLUE, "y = x^2"))
    svg.extend(
        legend_item(215, 102, ORANGE, "h = 1: slope 5", "10 7")
    )
    svg.extend(
        legend_item(414, 102, PURPLE, "h = 0.5: slope 4.5", "5 5")
    )
    svg.extend(legend_item(665, 102, GREEN, "local line: slope 4"))
    svg.append(text(LEFT, TOP - 13, "output y", "axis-label", "start"))

    for value in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        y = plot_y(value)
        svg.append(line(LEFT, y, WIDTH - RIGHT, y, "grid"))
        svg.append(text(LEFT - 12, y + 4, f"{value:g}", "tick", "end"))

    for value in (0.0, 1.0, 2.0, 3.0):
        x = plot_x(value)
        svg.append(line(x, TOP, x, PLOT_BOTTOM, "grid"))
        svg.append(text(x, PLOT_BOTTOM + 23, f"{value:g}", "tick"))

    svg.append(line(LEFT, PLOT_BOTTOM, WIDTH - RIGHT, PLOT_BOTTOM, "axis"))
    svg.append(line(LEFT, TOP, LEFT, PLOT_BOTTOM, "axis"))


def draw_square_curve(svg):
    samples = 400
    path_parts = []
    for index in range(samples + 1):
        value = X_MIN + (X_MAX - X_MIN) * index / samples
        command = "M" if index == 0 else "L"
        path_parts.append(
            f"{command}{plot_x(value):.2f},{plot_y(square(value)):.2f}"
        )
    svg.append(
        f'  <path d="{" ".join(path_parts)}" fill="none" stroke="{BLUE}" '
        'stroke-width="3.5" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
    )


def draw_slope_lines(svg):
    far_x, far_y, _ = SECANTS[0]
    near_x, near_y, _ = SECANTS[1]
    svg.append(
        line(
            plot_x(ANCHOR_X),
            plot_y(ANCHOR_Y),
            plot_x(far_x),
            plot_y(far_y),
            "far-secant",
        )
    )
    svg.append(
        line(
            plot_x(ANCHOR_X),
            plot_y(ANCHOR_Y),
            plot_x(near_x),
            plot_y(near_y),
            "near-secant",
        )
    )
    svg.append(
        line(
            plot_x(1.0),
            plot_y(tangent_y(1.0)),
            plot_x(X_MAX),
            plot_y(tangent_y(X_MAX)),
            "tangent",
        )
    )


def draw_labeled_points(svg):
    far_x, far_y, _ = SECANTS[0]
    near_x, near_y, _ = SECANTS[1]
    points = (
        (far_x, far_y, ORANGE),
        (near_x, near_y, PURPLE),
        (ANCHOR_X, ANCHOR_Y, GREEN),
    )
    for point_x, point_y, color in points:
        svg.append(
            f'  <circle cx="{plot_x(point_x):.2f}" '
            f'cy="{plot_y(point_y):.2f}" r="5.5" fill="{PANEL}" '
            f'stroke="{color}" stroke-width="3"/>'
        )

    svg.append(
        text(
            plot_x(far_x) - 12,
            plot_y(far_y) - 13,
            "(3, 9)",
            "point-label",
            "end",
        )
    )
    svg.append(
        text(
            plot_x(near_x) + 13,
            plot_y(near_y) - 12,
            "(2.5, 6.25)",
            "point-label",
            "start",
        )
    )
    svg.append(
        text(
            plot_x(ANCHOR_X) - 13,
            plot_y(ANCHOR_Y) + 27,
            "(2, 4) fixed",
            "point-label",
            "end",
        )
    )
    svg.append(
        text(
            plot_x(0.30),
            plot_y(0.65),
            "y = x^2",
            "curve-label",
            "start",
        )
    )
    svg.append(
        text(WIDTH - RIGHT, PLOT_BOTTOM + 46, "input x", "axis-label", "end")
    )


def draw_slope_cards(svg):
    card_y = 578
    card_gap = 12
    card_width = (PLOT_WIDTH - 2 * card_gap) / 3
    card_height = 78
    cards = (
        (
            "second point (3, 9)",
            "(9 - 4) / (3 - 2) = 5",
        ),
        (
            "second point (2.5, 6.25)",
            "(6.25 - 4) / (2.5 - 2) = 4.5",
        ),
        (
            "at the fixed point (2, 4)",
            "local slope = 4",
        ),
    )
    for index, (heading, value) in enumerate(cards):
        card_x = LEFT + index * (card_width + card_gap)
        svg.append(
            f'  <rect x="{card_x:.2f}" y="{card_y:.2f}" '
            f'width="{card_width:.2f}" height="{card_height:.2f}" '
            f'rx="5" fill="{CARD_FILL}" stroke="{GRID}"/>'
        )
        svg.append(
            text(
                card_x + card_width / 2,
                card_y + 28,
                heading,
                "card-title",
            )
        )
        svg.append(
            text(
                card_x + card_width / 2,
                card_y + 55,
                value,
                "card-value",
            )
        )


def write_svg(svg):
    svg.append("</svg>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("\n".join(svg) + "\n")


def main():
    verify_fixture()
    svg = start_svg()

    draw_legend_and_axes(svg)
    draw_square_curve(svg)
    draw_slope_lines(svg)
    draw_labeled_points(svg)
    draw_slope_cards(svg)
    write_svg(svg)


if __name__ == "__main__":
    main()
