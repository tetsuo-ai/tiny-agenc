#!/usr/bin/env python3
"""Generate the deterministic causal-attention heatmap used in Chapter 5."""

from math import exp, sqrt
from pathlib import Path


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "book"
    / "figures"
    / "05-causal-attention.svg"
)

WIDTH = 1040
HEIGHT = 650
TABLE_X = 40
TABLE_Y = 112
ROW_LABEL_WIDTH = 180
CELL_WIDTH = 190
OUTPUT_WIDTH = 210
HEADER_HEIGHT = 94
ROW_HEIGHT = 126

QUERIES = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
KEYS = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
VALUES = ((2.0, 1.0), (4.0, 3.0), (8.0, 7.0))

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#cbd3df"
BLUE_LIGHT = "#eff6ff"
BLUE_DARK = "#1d4ed8"
PANEL = "#ffffff"
HEADER_FILL = "#f4f7fb"
OUTPUT_FILL = "#f1f8f5"
UNUSED_FILL = "#f5f6f8"
UNUSED_LINE = "#d7dce4"


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def softmax(values):
    maximum = max(values)
    exponentials = [exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def attention_fixture():
    """Compute every visible weight and output from labs/check05.c."""
    scale = 1.0 / sqrt(len(QUERIES[0]))
    all_weights = []
    all_outputs = []
    for time_index, query in enumerate(QUERIES):
        raw_scores = [
            scale * dot(query, KEYS[key_index])
            for key_index in range(time_index + 1)
        ]
        weights = softmax(raw_scores)
        output = [
            sum(
                weights[key_index] * VALUES[key_index][channel]
                for key_index in range(time_index + 1)
            )
            for channel in range(len(query))
        ]
        all_weights.append(weights)
        all_outputs.append(output)
    return all_weights, all_outputs


def blend_hex(low, high, amount):
    low_rgb = tuple(int(low[index : index + 2], 16) for index in (1, 3, 5))
    high_rgb = tuple(int(high[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(low_value + (high_value - low_value) * amount)
        for low_value, high_value in zip(low_rgb, high_rgb)
    )
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def text(x, y, value, css_class, anchor="middle"):
    return (
        f'  <text x="{x:.2f}" y="{y:.2f}" '
        f'class="{css_class}" text-anchor="{anchor}">{value}</text>'
    )


def rect(x, y, width, height, fill, stroke=GRID, radius=0):
    return (
        f'  <rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="{radius}" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )


def unused_mark(x, y, width, height):
    inset = 10
    return [
        (
            f'  <line x1="{x + inset:.2f}" y1="{y + inset:.2f}" '
            f'x2="{x + width - inset:.2f}" y2="{y + height - inset:.2f}" '
            f'stroke="{UNUSED_LINE}" stroke-width="2"/>'
        ),
        (
            f'  <line x1="{x + width - inset:.2f}" y1="{y + inset:.2f}" '
            f'x2="{x + inset:.2f}" y2="{y + height - inset:.2f}" '
            f'stroke="{UNUSED_LINE}" stroke-width="2"/>'
        ),
    ]


def vector(values, digits=0):
    if digits == 0:
        members = ", ".join(f"{value:g}" for value in values)
    else:
        members = ", ".join(f"{value:.{digits}f}" for value in values)
    return f"[{members}]"


def main():
    weights, outputs = attention_fixture()
    table_width = ROW_LABEL_WIDTH + 3 * CELL_WIDTH + OUTPUT_WIDTH
    table_height = HEADER_HEIGHT + 3 * ROW_HEIGHT

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        '  <title id="title">Causal attention weights for three positions</title>',
        (
            '  <desc id="desc">A lower triangular heatmap computed from the '
            'three-position query, key, and value fixture in labs/check05.c. '
            'Future cells are marked not read.</desc>'
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
        "    .header-main { font-size: 14px; font-weight: 700; }",
        "    .header-detail { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 12px; fill: " + MUTED + "; }",
        "    .row-main { font-size: 14px; font-weight: 700; }",
        "    .row-detail { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 13px; }",
        "    .cell-label { font-size: 12px; font-weight: 650; }",
        "    .cell-value { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 16px; font-weight: 700; }",
        "    .cell-label-light { font-size: 12px; font-weight: 650; "
        "fill: #ffffff; }",
        "    .cell-value-light { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 16px; font-weight: 700; "
        "fill: #ffffff; }",
        "    .unused { font-size: 12px; font-weight: 650; fill: " + MUTED + "; }",
        "    .output-label { font-size: 12px; fill: " + MUTED + "; }",
        "    .output-value { font-family: ui-monospace, SFMono-Regular, "
        "Menlo, Consolas, monospace; font-size: 13px; font-weight: 700; }",
        "    .footer { font-size: 13px; fill: " + MUTED + "; }",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            TABLE_X,
            39,
            "A position can mix values only from itself and its past",
            "title",
            "start",
        ),
        text(
            TABLE_X,
            69,
            "Fixture head size 2; every query-key dot product is scaled by 1/sqrt(2)",
            "subtitle",
            "start",
        ),
        rect(
            TABLE_X,
            TABLE_Y,
            table_width,
            table_height,
            PANEL,
            GRID,
            6,
        ),
    ]

    header_cells = [
        (
            TABLE_X,
            ROW_LABEL_WIDTH,
            "query position",
            "Q vector",
        )
    ]
    for key_index in range(3):
        header_cells.append(
            (
                TABLE_X + ROW_LABEL_WIDTH + key_index * CELL_WIDTH,
                CELL_WIDTH,
                f"source t2 = {key_index}",
                f"K={vector(KEYS[key_index])}  V={vector(VALUES[key_index])}",
            )
        )
    header_cells.append(
        (
            TABLE_X + ROW_LABEL_WIDTH + 3 * CELL_WIDTH,
            OUTPUT_WIDTH,
            "weighted output",
            "sum(weight * V)",
        )
    )

    for x, width, heading, detail in header_cells:
        svg.append(rect(x, TABLE_Y, width, HEADER_HEIGHT, HEADER_FILL))
        svg.append(text(x + width / 2, TABLE_Y + 37, heading, "header-main"))
        svg.append(text(x + width / 2, TABLE_Y + 63, detail, "header-detail"))

    for time_index in range(3):
        row_y = TABLE_Y + HEADER_HEIGHT + time_index * ROW_HEIGHT
        svg.append(
            rect(TABLE_X, row_y, ROW_LABEL_WIDTH, ROW_HEIGHT, HEADER_FILL)
        )
        svg.append(
            text(
                TABLE_X + 18,
                row_y + 49,
                f"query t = {time_index}",
                "row-main",
                "start",
            )
        )
        svg.append(
            text(
                TABLE_X + 18,
                row_y + 77,
                f"Q={vector(QUERIES[time_index])}",
                "row-detail",
                "start",
            )
        )

        for key_index in range(3):
            cell_x = TABLE_X + ROW_LABEL_WIDTH + key_index * CELL_WIDTH
            if key_index <= time_index:
                weight = weights[time_index][key_index]
                fill = blend_hex(BLUE_LIGHT, BLUE_DARK, weight)
                svg.append(rect(cell_x, row_y, CELL_WIDTH, ROW_HEIGHT, fill))
                label_class = (
                    "cell-label-light" if weight >= 0.62 else "cell-label"
                )
                value_class = (
                    "cell-value-light" if weight >= 0.62 else "cell-value"
                )
                svg.append(
                    text(
                        cell_x + CELL_WIDTH / 2,
                        row_y + 49,
                        "softmax weight",
                        label_class,
                    )
                )
                svg.append(
                    text(
                        cell_x + CELL_WIDTH / 2,
                        row_y + 78,
                        f"{weight:.6f}",
                        value_class,
                    )
                )
            else:
                svg.append(
                    rect(
                        cell_x,
                        row_y,
                        CELL_WIDTH,
                        ROW_HEIGHT,
                        UNUSED_FILL,
                    )
                )
                svg.extend(
                    unused_mark(cell_x, row_y, CELL_WIDTH, ROW_HEIGHT)
                )
                svg.append(
                    text(
                        cell_x + CELL_WIDTH / 2,
                        row_y + 55,
                        "future position",
                        "unused",
                    )
                )
                svg.append(
                    text(
                        cell_x + CELL_WIDTH / 2,
                        row_y + 78,
                        "not read",
                        "unused",
                    )
                )

        output_x = TABLE_X + ROW_LABEL_WIDTH + 3 * CELL_WIDTH
        svg.append(
            rect(output_x, row_y, OUTPUT_WIDTH, ROW_HEIGHT, OUTPUT_FILL)
        )
        svg.append(
            text(
                output_x + OUTPUT_WIDTH / 2,
                row_y + 49,
                f"output t = {time_index}",
                "output-label",
            )
        )
        svg.append(
            text(
                output_x + OUTPUT_WIDTH / 2,
                row_y + 78,
                vector(outputs[time_index], 5),
                "output-value",
            )
        )

    svg.extend(
        [
            text(
                TABLE_X,
                TABLE_Y + table_height + 35,
                (
                    "Each used row sums to 1. Future cells are never scored "
                    "or passed to softmax."
                ),
                "footer",
                "start",
            ),
            "</svg>",
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
