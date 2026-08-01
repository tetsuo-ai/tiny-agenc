#!/usr/bin/env python3
"""Plot every recorded loss report from the 5,000-step showcase run."""

from html import escape
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "book" / "logs" / "train-cyberpunk-5000.log"
OUTPUT = ROOT / "book" / "figures" / "17-showcase-loss.svg"

WIDTH = 960
HEIGHT = 720
LEFT = 86
RIGHT = 42
TOP = 158
BOTTOM = 488
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = BOTTOM - TOP

FIRST_STEP = 1
FINAL_STEP = 5000
Y_MIN = 0.5
Y_MAX = 4.5

INSET_LEFT = 516
INSET_RIGHT = 898
INSET_TOP = 184
INSET_BOTTOM = 356
INSET_PLOT_LEFT = 556
INSET_PLOT_RIGHT = 880
INSET_PLOT_TOP = 222
INSET_PLOT_BOTTOM = 332
INSET_FIRST_STEP = 3900
INSET_FINAL_STEP = 4500
INSET_Y_MIN = 0.70
INSET_Y_MAX = 0.95

SELECTED_SAMPLE_STEPS = (250, 500, 1000, 1750, 4750)

INK = "#17233d"
MUTED = "#5d6878"
GRID = "#d6dce5"
AXIS = "#7a8699"
BLUE = "#225fc7"
ORANGE = "#b84e16"
GREEN = "#147d64"
PLOT_BACKGROUND = "#f8fafc"
PANEL = "#ffffff"

REPORT = re.compile(
    r"^step\s+(\d+)/5000 \| loss ([0-9]+\.[0-9]+) \|",
    re.MULTILINE,
)
SAMPLE = re.compile(
    r"^---- sample at step (\d+) ----$",
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


def rect(x, y, width, height, fill, stroke, radius=6, css_class=None):
    class_attribute = (
        f' class="{css_class}"'
        if css_class is not None
        else ""
    )
    return (
        f'  <rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="{radius}" fill="{fill}" '
        f'stroke="{stroke}"{class_attribute}/>'
    )


def polyline(points, css_class):
    coordinates = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in points
    )
    return (
        f'  <polyline points="{coordinates}" class="{css_class}"/>'
    )


def diamond(x, y, radius, fill, stroke, stroke_width=2):
    points = (
        f"{x:.2f},{y - radius:.2f} "
        f"{x + radius:.2f},{y:.2f} "
        f"{x:.2f},{y + radius:.2f} "
        f"{x - radius:.2f},{y:.2f}"
    )
    return (
        f'  <polygon points="{points}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def read_log():
    contents = LOG.read_text(encoding="utf-8")
    reports = [
        (int(step), float(loss))
        for step, loss in REPORT.findall(contents)
    ]
    sample_steps = [
        int(step) for step in SAMPLE.findall(contents)
    ]
    expected_steps = [FIRST_STEP, *range(50, FINAL_STEP + 1, 50)]
    expected_sample_steps = list(range(250, FINAL_STEP + 1, 250))

    assert len(reports) == 101
    assert [step for step, _ in reports] == expected_steps
    assert reports[0] == (FIRST_STEP, 4.4395)
    assert reports[-1] == (FINAL_STEP, 0.7668)
    assert min(reports, key=lambda report: report[1]) == (4150, 0.7281)
    assert sum(
        current[1] > previous[1]
        for previous, current in zip(reports, reports[1:])
    ) == 42
    assert sample_steps == expected_sample_steps
    assert set(SELECTED_SAMPLE_STEPS) <= set(sample_steps)
    assert all(Y_MIN <= loss <= Y_MAX for _, loss in reports)
    return reports


def plot_x(step):
    fraction = (step - FIRST_STEP) / (FINAL_STEP - FIRST_STEP)
    return LEFT + fraction * PLOT_WIDTH


def plot_y(loss):
    fraction = (Y_MAX - loss) / (Y_MAX - Y_MIN)
    return TOP + fraction * PLOT_HEIGHT


def inset_x(step):
    fraction = (
        (step - INSET_FIRST_STEP)
        / (INSET_FINAL_STEP - INSET_FIRST_STEP)
    )
    return (
        INSET_PLOT_LEFT
        + fraction * (INSET_PLOT_RIGHT - INSET_PLOT_LEFT)
    )


def inset_y(loss):
    fraction = (
        (INSET_Y_MAX - loss) / (INSET_Y_MAX - INSET_Y_MIN)
    )
    return (
        INSET_PLOT_TOP
        + fraction * (INSET_PLOT_BOTTOM - INSET_PLOT_TOP)
    )


def prepare_plot_data():
    reports = read_log()
    report_by_step = dict(reports)
    points = [
        (plot_x(step), plot_y(loss))
        for step, loss in reports
    ]
    inset_reports = [
        (step, loss)
        for step, loss in reports
        if INSET_FIRST_STEP <= step <= INSET_FINAL_STEP
    ]
    inset_points = [
        (inset_x(step), inset_y(loss))
        for step, loss in inset_reports
    ]

    assert len(inset_reports) == 13
    assert inset_reports[0] == (3900, 0.8755)
    assert inset_reports[-1] == (4500, 0.7881)
    assert report_by_step[4150] == 0.7281
    assert report_by_step[4200] == 0.8766
    assert all(
        LEFT <= x <= WIDTH - RIGHT and TOP <= y <= BOTTOM
        for x, y in points
    )
    assert all(
        INSET_PLOT_LEFT <= x <= INSET_PLOT_RIGHT
        and INSET_PLOT_TOP <= y <= INSET_PLOT_BOTTOM
        for x, y in inset_points
    )

    return report_by_step, points, inset_points


def start_svg():
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        (
            '  <title id="title">One hundred one random-batch loss '
            'reports across the showcase training run</title>'
        ),
        (
            '  <desc id="desc">The recorded training loss falls from '
            '4.4395 at step 1 to 0.7668 at step 5000, but neighboring '
            'reports bounce because each dot grades one random training '
            'batch. An inset magnifies steps 3900 through 4500. Orange '
            'diamonds mark five pre-update loss reports whose completed '
            'updates are followed by progress samples. This plot contains '
            'no held-out measurements.</desc>'
        ),
        "  <style>",
        (
            "    text { font-family: system-ui, -apple-system, "
            "BlinkMacSystemFont, 'Segoe UI', sans-serif; "
            f"fill: {INK}; font-style: normal; }}"
        ),
        "    .title { font-size: 27px; font-weight: 700; }",
        f"    .subtitle {{ font-size: 15px; fill: {MUTED}; }}",
        f"    .tick {{ font-size: 14px; fill: {MUTED}; }}",
        "    .axis-label { font-size: 16px; font-weight: 650; }",
        "    .legend { font-size: 15px; font-weight: 650; }",
        "    .inset-title { font-size: 15px; font-weight: 700; }",
        f"    .note {{ font-size: 15px; fill: {MUTED}; }}",
        f"    .source {{ font-size: 14px; fill: {MUTED}; }}",
        f"    .grid {{ stroke: {GRID}; stroke-width: 1; }}",
        f"    .axis {{ stroke: {AXIS}; stroke-width: 1.5; }}",
        (
            f"    .curve {{ fill: none; stroke: {BLUE}; "
            "stroke-width: 2.6; stroke-linecap: round; "
            "stroke-linejoin: round; }"
        ),
        (
            f"    .inset-curve {{ fill: none; stroke: {BLUE}; "
            "stroke-width: 2.4; stroke-linecap: round; "
            "stroke-linejoin: round; }"
        ),
        (
            f"    .selection {{ stroke: {GREEN}; stroke-width: 1.5; "
            "stroke-dasharray: 4 3; }"
        ),
        f"    .connector {{ stroke: {GREEN}; stroke-width: 1.5; }}",
        "  </style>",
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{PANEL}"/>',
        text(
            LEFT,
            39,
            "One random batch at a time makes a jagged curve",
            "title",
            "start",
        ),
        text(
            LEFT,
            68,
            "All 101 reports from the uninterrupted 5,000-step showcase run.",
            "subtitle",
            "start",
        ),
        text(
            LEFT,
            93,
            (
                "The long-run direction falls; neighboring reports "
                "can still rise because the batch changes."
            ),
            "subtitle",
            "start",
        ),
        line(LEFT, 123, LEFT + 42, 123, "curve"),
        text(
            LEFT + 52,
            128,
            "loss on one random training batch",
            "legend",
            "start",
        ),
        diamond(453, 123, 6, PANEL, ORANGE),
        text(
            470,
            128,
            "report step with post-update sample",
            "legend",
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


def draw_main_plot(svg, points, report_by_step):
    for loss in (0.5, 1.0, 2.0, 3.0, 4.0, 4.5):
        y = plot_y(loss)
        svg.append(line(LEFT, y, WIDTH - RIGHT, y, "grid"))
        svg.append(
            text(LEFT - 13, y + 5, f"{loss:.1f}", "tick", "end")
        )

    for step in (FIRST_STEP, 1000, 2000, 3000, 4000, FINAL_STEP):
        x = plot_x(step)
        svg.append(line(x, TOP, x, BOTTOM, "grid"))
        svg.append(
            text(x, BOTTOM + 25, f"{step:,}", "tick")
        )

    svg.extend(
        (
            line(LEFT, BOTTOM, WIDTH - RIGHT, BOTTOM, "axis"),
            line(LEFT, TOP, LEFT, BOTTOM, "axis"),
            text(
                LEFT,
                TOP - 14,
                "recorded training loss",
                "axis-label",
                "start",
            ),
            text(
                WIDTH - RIGHT,
                BOTTOM + 51,
                "training step",
                "axis-label",
                "end",
            ),
            polyline(points, "curve"),
        )
    )

    for x, y in points:
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="2.3" '
            f'fill="{BLUE}" stroke="{PANEL}" stroke-width="0.7"/>'
        )

    for step in SELECTED_SAMPLE_STEPS:
        svg.append(
            diamond(
                plot_x(step),
                plot_y(report_by_step[step]),
                6,
                PANEL,
                ORANGE,
                2.2,
            )
        )


def draw_inset_selection(svg):
    selection_x = plot_x(INSET_FIRST_STEP)
    selection_y = plot_y(INSET_Y_MAX)
    selection_width = (
        plot_x(INSET_FINAL_STEP) - plot_x(INSET_FIRST_STEP)
    )
    selection_height = plot_y(INSET_Y_MIN) - plot_y(INSET_Y_MAX)
    svg.append(
        rect(
            selection_x,
            selection_y,
            selection_width,
            selection_height,
            "none",
            GREEN,
            2,
            "selection",
        )
    )
    svg.append(
        line(
            selection_x + selection_width,
            selection_y,
            INSET_RIGHT - 8,
            INSET_BOTTOM,
            "connector",
        )
    )


def draw_inset_frame(svg):
    svg.append(
        rect(
            INSET_LEFT,
            INSET_TOP,
            INSET_RIGHT - INSET_LEFT,
            INSET_BOTTOM - INSET_TOP,
            PANEL,
            GREEN,
            7,
        )
    )
    svg.append(
        text(
            INSET_LEFT + 15,
            INSET_TOP + 25,
            "Late reports, magnified",
            "inset-title",
            "start",
        )
    )


def draw_inset_grid(svg):
    for loss in (0.70, 0.80, 0.90):
        y = inset_y(loss)
        svg.append(
            line(
                INSET_PLOT_LEFT,
                y,
                INSET_PLOT_RIGHT,
                y,
                "grid",
            )
        )
        svg.append(
            text(
                INSET_PLOT_LEFT - 8,
                y + 4,
                f"{loss:.2f}",
                "tick",
                "end",
            )
        )

    for step in (3900, 4200, 4500):
        x = inset_x(step)
        svg.append(
            line(
                x,
                INSET_PLOT_TOP,
                x,
                INSET_PLOT_BOTTOM,
                "grid",
            )
        )
        svg.append(
            text(
                x,
                INSET_PLOT_BOTTOM + 18,
                f"{step:,}",
                "tick",
            )
        )


def draw_inset_curve(svg, inset_points, report_by_step):
    svg.append(polyline(inset_points, "inset-curve"))
    for x, y in inset_points:
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" '
            f'fill="{BLUE}" stroke="{PANEL}" stroke-width="0.7"/>'
        )

    for step in (4150, 4200):
        x = inset_x(step)
        y = inset_y(report_by_step[step])
        svg.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4.6" '
            f'fill="{PANEL}" stroke="{GREEN}" stroke-width="2"/>'
        )


def finish_svg(svg):
    svg.extend(
        (
            text(
                LEFT,
                572,
                (
                    "Measured rise: step 4,150 is 0.7281; "
                    "step 4,200 is 0.8766."
                ),
                "note",
                "start",
            ),
            text(
                LEFT,
                601,
                (
                    "Diamond report steps: 250, 500, 1,000, "
                    "1,750, and 4,750."
                ),
                "note",
                "start",
            ),
            text(
                LEFT,
                627,
                (
                    "Each loss dot is pre-update; its sample follows "
                    "that step's update."
                ),
                "note",
                "start",
            ),
            text(
                LEFT,
                687,
                "Source: book/logs/train-cyberpunk-5000.log",
                "source",
                "start",
            ),
            "</svg>",
        )
    )


def write_svg(svg):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(svg) + "\n")


def main():
    report_by_step, points, inset_points = prepare_plot_data()
    svg = start_svg()
    draw_main_plot(svg, points, report_by_step)
    draw_inset_selection(svg)
    draw_inset_frame(svg)
    draw_inset_grid(svg)
    draw_inset_curve(svg, inset_points, report_by_step)
    finish_svg(svg)
    write_svg(svg)


if __name__ == "__main__":
    main()
