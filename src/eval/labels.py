"""
Label placement for the accuracy-vs-cost figures. Each label has candidate
positions around its own point, in order of preference: under its error bar,
over it, beside the point, then shifted sideways, further along the bar and
further out. A label may take a candidate only if it stays inside the axes,
covers no obstacle (error bars, markers, the legend) and no other label, and
is nearer its own error bar than any other model's, so it cannot be read as
a neighbour's. Where the neighbours leave no such spot, the label takes the
clear one that is closest to its own bar and least nearer another. Pure
geometry on display-space boxes, tested without drawing.
"""

from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 in display units, y up

SEARCH_BUDGET = 20_000  # partial placements tried before settling for a greedy one
SIDE_HEIGHTS = (0.5, -0.5, 1.0, -1.0)  # beside the bar, away from the point, by preference
AMBIGUITY_WEIGHT = 2  # in a fallback spot, each unit another bar is nearer costs this much distance


class Position(NamedTuple):
    """
    Where a label goes: anchored on its error bar at `along` (-1 the lower
    end, 0 the point, 1 the upper end, linear in between), offset by dx, dy
    points, with the text aligned ha / va to that spot.
    """
    along: float
    dx: float
    dy: float
    ha: str
    va: str


def overlaps(a: Box, b: Box) -> bool:
    """True if the boxes share area; touching edges do not count."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def inside(box: Box, bounds: Box) -> bool:
    return (bounds[0] <= box[0] and box[2] <= bounds[2]
            and bounds[1] <= box[1] and box[3] <= bounds[3])


def gap(a: Box, b: Box) -> float:
    """Shortest distance between two boxes (0 if they touch or overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def candidate_positions(gap_pt: float, side_pt: float, step_pt: float, shift_pt: float,
                        max_steps: int, max_shifts: int) -> List[Position]:
    """
    Every position a label may take, in order of preference. Tier k holds
    the spots under and over the bar k steps further out, each centred and
    then shifted sideways; tier 0 adds the spots beside the point, and
    tiers 1..len(SIDE_HEIGHTS) the spots beside the bar at other heights.
    """
    shifts = [0.0] + [sign * shift_pt * n for n in range(1, max_shifts + 1) for sign in (-1, 1)]
    beside = [(0.0,), *((h,) for h in SIDE_HEIGHTS)]
    positions = []
    for k in range(max_steps + 1):
        out = gap_pt + step_pt * k
        vertical = [[Position(-1, dx, -out, "center", "top"),
                     Position(1, dx, out, "center", "bottom")] for dx in shifts]
        side = [Position(h, sign * side_pt, 0, "left" if sign > 0 else "right", "center")
                for h in (beside[k] if k < len(beside) else ()) for sign in (1, -1)]
        positions += [*vertical[0], *side, *(p for pair in vertical[1:] for p in pair)]
    return positions


def label_box(anchor: Tuple[float, float], size: Tuple[float, float], position: Position,
              px_per_pt: float) -> Box:
    """Display box of a w x h label placed at `position` around the display point `anchor`."""
    (x, y), (w, h) = anchor, size
    x, y = x + position.dx * px_per_pt, y + position.dy * px_per_pt
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[position.ha]
    y0 = {"bottom": y, "center": y - h / 2, "top": y - h}[position.va]
    return (x0, y0, x0 + w, y0 + h)


def _clear(box: Box, obstacles: Sequence[Box], bounds: Box) -> bool:
    """Inside the axes and clear of every obstacle."""
    return inside(box, bounds) and not any(overlaps(box, o) for o in obstacles)


def _nearest_own(box: Box, own: int, bars: Sequence[Box]) -> bool:
    """Nearer its own bar than any other (always, if there are no bars)."""
    mine = gap(box, bars[own]) if bars else 0
    return all(mine < gap(box, bar) for j, bar in enumerate(bars) if j != own)


def _ambiguity_cost(box: Box, own: int, bars: Sequence[Box]) -> float:
    """Distance to its own bar, plus AMBIGUITY_WEIGHT x how much nearer the nearest other is."""
    mine = gap(box, bars[own])
    nearest_other = min(gap(box, bar) for j, bar in enumerate(bars) if j != own)
    return mine + AMBIGUITY_WEIGHT * max(0.0, mine - nearest_other)


def _options(boxes: Sequence[Box], own: int, obstacles: Sequence[Box], bars: Sequence[Box],
             bounds: Box) -> List[int]:
    """
    A label's usable candidates: the clear ones nearest its own bar, in order
    of preference, or if there are none, every clear one by _ambiguity_cost.
    """
    clear = [i for i, box in enumerate(boxes) if _clear(box, obstacles, bounds)]
    nearest = [i for i in clear if _nearest_own(boxes[i], own, bars)]
    if nearest or len(bars) < 2:
        return nearest
    return sorted(clear, key=lambda i: _ambiguity_cost(boxes[i], own, bars))


def _search(order: Sequence[int], allowed: Sequence[Sequence[int]],
            candidates: Sequence[Sequence[Box]], placed: dict, budget: Iterator) -> Optional[dict]:
    """
    Depth-first: the first choice per label (in `order`) overlapping no label
    placed; None once `budget` (an iterator, one item per choice tried) runs out.
    """
    if len(placed) == len(order):
        return placed
    label = order[len(placed)]
    for index in allowed[label]:
        if next(budget, None) is None:
            return None
        box = candidates[label][index]
        if any(overlaps(box, candidates[j][i]) for j, i in placed.items()):
            continue
        found = _search(order, allowed, candidates, {**placed, label: index}, budget)
        if found is not None:
            return found
    return None


def _greedy(allowed: Sequence[Sequence[int]], candidates: Sequence[Sequence[Box]]) -> List[int]:
    """Each label in turn: its first allowed candidate clear of those before it, else its first."""
    placed: List[Box] = []
    chosen = []
    for options, boxes in zip(allowed, candidates):
        clear = [i for i in options if not any(overlaps(boxes[i], p) for p in placed)]
        index = clear[0] if clear else (options[0] if options else 0)
        placed.append(boxes[index])
        chosen.append(index)
    return chosen


def choose_placements(candidates: Sequence[Sequence[Box]], obstacles: Sequence[Box],
                      bounds: Box, bars: Sequence[Box] = ()) -> List[int]:
    """
    Index of the candidate box each label takes. candidates[i] are label i's
    boxes by preference; bars[i], if bars are given, is its own error bar,
    and a label may only sit nearer its own bar than any other (see
    _options for a label with no such spot). Labels with no such spot
    choose first, then those with the fewest allowed candidates, each taking
    its most preferred one clear of the labels already placed, backtracking
    when a later label would have none. If no such placement exists, each label (in order)
    takes its first allowed candidate clear of the ones before it, else its
    first candidate.
    """
    allowed = [_options(boxes, label, obstacles, bars, bounds)
               for label, boxes in enumerate(candidates)]
    fallback = [bool(options) and not _nearest_own(candidates[label][options[0]], label, bars)
                for label, options in enumerate(allowed)]
    order = sorted(range(len(candidates)),
                   key=lambda label: (not fallback[label], len(allowed[label])))
    found = _search(order, allowed, candidates, {}, iter(range(SEARCH_BUDGET)))
    if found is None:
        return _greedy(allowed, candidates)
    return [found[label] for label in range(len(candidates))]
