#!/usr/bin/env python3
"""Check book/knowledge-graph.json against the book.

Detects, per AGENTS.md P6:
  1. dangling depends_on references;
  2. circular definitions (cycles in depends_on);
  3. forward teaching: a concept whose owner-level definition depends
     on a concept that is neither taught nor previewed by then;
  4. use-before-teach: an alias appearing in a written unit earlier than
     the concept's owner or earliest preview;
  5. intuition gaps: a concept reachable in a written unit with no
     recorded intuition (definitions alone do not teach);
  6. unanchored metaphors: an intuition whose anchor is neither
     "self" nor a registered root metaphor, and root metaphors whose
     markers never appear in the written units.

Standalone report tool; not wired into make. Exit 1 on violations.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "book" / "knowledge-graph.json"

UNIT_ORDER = {str(chapter): chapter for chapter in range(19)}
UNIT_ORDER.update({
    "appendix-a": 19,
    "appendix-b": 20,
    "appendix-c": 21,
})
RANK_UNIT = {rank: unit for unit, rank in UNIT_ORDER.items()}
APPENDIX_PATHS = {
    "appendix-a": ROOT / "book" / "appendix-a-derivations.md",
    "appendix-b": ROOT / "book" / "appendix-b-sources.md",
    "appendix-c": ROOT / "book" / "appendix-c-capstone-labs.md",
}


def unit_rank(value):
    """Return the book-order rank for a chapter number or appendix id."""
    unit = str(value)

    if unit not in UNIT_ORDER:
        raise ValueError(f"unknown book unit '{unit}'")
    return UNIT_ORDER[unit]


def unit_label_from_rank(rank):
    unit = RANK_UNIT[rank]

    return f"ch {int(unit):02d}" if unit.isdigit() else unit


graph = json.loads(GRAPH.read_text())
concepts = graph["concepts"]
metaphors = graph.get("metaphors", {})
written_units = {str(chapter) for chapter in graph.get("written_chapters", [])}
written_units.update(graph.get("written_appendices", []))
written = {unit_rank(unit) for unit in written_units}
problems = []

# 1. dangling references
for name, c in concepts.items():
    for dep in c.get("depends_on", []):
        if dep not in concepts:
            problems.append(f"dangling: {name} depends on unknown '{dep}'")

# 2. cycles in depends_on
WHITE, GRAY, BLACK = 0, 1, 2
color = {n: WHITE for n in concepts}


def visit(node, path):
    color[node] = GRAY
    for dep in concepts[node].get("depends_on", []):
        if dep not in concepts:
            continue
        if color[dep] == GRAY:
            cycle = path[path.index(dep):] + [dep] if dep in path else [node, dep]
            problems.append("circular definition: " + " -> ".join(cycle))
        elif color[dep] == WHITE:
            visit(dep, path + [dep])
    color[node] = BLACK


for n in concepts:
    if color[n] == WHITE:
        visit(n, [n])


def available(name):
    """Earliest book unit where a concept may be used."""
    c = concepts[name]
    units = [unit_rank(c["owner"])]
    units += [unit_rank(k) for k in c.get("previews", {})]
    return min(units)


# 3. forward teaching
for name, c in concepts.items():
    owner = unit_rank(c["owner"])
    for dep in c.get("depends_on", []):
        if dep in concepts and available(dep) > owner:
            problems.append(
                f"forward teaching: {name} "
                f"({unit_label_from_rank(owner)}) is defined via '{dep}', "
                f"not available until "
                f"{unit_label_from_rank(available(dep))}"
            )

# 4. use-before-teach scan over written chapter and appendix files
chapter_files = sorted(ROOT.glob("book/[0-1][0-9]-*.md"))
texts = {
    unit_rank(str(int(path.name[:2]))): path.read_text()
    for path in chapter_files
}
for appendix in graph.get("written_appendices", []):
    if appendix not in APPENDIX_PATHS:
        raise ValueError(f"no Markdown path registered for '{appendix}'")
    texts[unit_rank(appendix)] = APPENDIX_PATHS[appendix].read_text()

for name, c in concepts.items():
    if c.get("scan") is False or not c.get("aliases"):
        continue
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(a) for a in c["aliases"]) + r")\b",
        re.IGNORECASE,
    )
    first_hit = next(
        (ch for ch in sorted(texts) if pattern.search(texts[ch])), None
    )
    if first_hit is not None and first_hit < available(name):
        problems.append(
            f"use-before-teach: '{name}' first appears in "
            f"{unit_label_from_rank(first_hit)} but is not taught/previewed "
            f"until {unit_label_from_rank(available(name))} "
            f"(add a gloss+pointer there, or record the preview)"
        )

# 5. intuition gaps in written units
for name, c in concepts.items():
    if available(name) in written or unit_rank(c["owner"]) in written:
        if not c.get("intuition"):
            problems.append(
                f"intuition gap: '{name}' is reachable in a written "
                f"unit but has no recorded intuition"
            )

# 6. metaphor anchoring
census = {}
for name, c in concepts.items():
    anchor = c.get("anchor", "")
    if not c.get("intuition"):
        continue
    if anchor != "self" and anchor not in metaphors:
        problems.append(
            f"unanchored metaphor: '{name}' anchors to '{anchor}', "
            f"which is not a registered root"
        )
    census[anchor] = census.get(anchor, 0) + 1

all_written_text = "".join(texts[ch] for ch in sorted(texts) if ch in written)
for mname, m in metaphors.items():
    established = m.get("established")

    if established is not None and unit_rank(established) in written:
        if not any(mk.lower() in all_written_text.lower() for mk in m.get("markers", [])):
            problems.append(
                f"ghost metaphor: root '{mname}' claims establishment in a "
                f"written chapter but no marker appears in the prose"
            )

if problems:
    for p in problems:
        print("check-graph:", p)
    print(f"check-graph: {len(problems)} problem(s)")
    sys.exit(1)

parts = ", ".join(f"{k}:{v}" for k, v in sorted(census.items(), key=lambda x: -x[1]))
print(f"check-graph: {len(concepts)} concepts, {len(metaphors)} root "
      f"metaphors, no gaps, no cycles, no forward teaching, no "
      f"use-before-teach, no intuition gaps")
print(f"check-graph: metaphor census: {parts}")
