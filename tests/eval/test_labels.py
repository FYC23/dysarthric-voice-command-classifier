"""Label placement for the accuracy-vs-cost figures."""

import pytest

from src.eval.labels import (
    Position, candidate_positions, choose_placements, gap, label_box, overlaps,
)

BOUNDS = (0, 0, 100, 100)


def test_boxes_that_only_touch_do_not_overlap():
    assert overlaps((0, 0, 10, 10), (5, 5, 15, 15))
    assert not overlaps((0, 0, 10, 10), (10, 0, 20, 10))


def test_gap_is_zero_for_touching_boxes_and_euclidean_across_a_corner():
    assert gap((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert gap((0, 0, 10, 10), (5, 15, 6, 20)) == 5
    assert gap((0, 0, 10, 10), (13, 14, 20, 20)) == pytest.approx(5)


@pytest.mark.parametrize("position, box", [
    (Position(0, 0, -6, "center", "top"), (95, 38, 105, 44)),
    (Position(0, 0, 6, "center", "bottom"), (95, 56, 105, 62)),
    (Position(0, 11, 0, "left", "center"), (111, 47, 121, 53)),
    (Position(0, -11, 0, "right", "center"), (79, 47, 89, 53)),
])
def test_label_box_offsets_and_aligns_the_text_around_its_anchor(position, box):
    assert label_box((100, 50), (10, 6), position, px_per_pt=1) == pytest.approx(box)


def test_candidates_start_under_the_bar_then_over_it_then_beside_the_point():
    first = candidate_positions(gap_pt=6, side_pt=11, step_pt=6, shift_pt=4,
                                max_steps=2, max_shifts=2)
    assert first[:4] == [Position(-1, 0, -6, "center", "top"),
                         Position(1, 0, 6, "center", "bottom"),
                         Position(0, 11, 0, "left", "center"),
                         Position(0, -11, 0, "right", "center")]
    assert Position(-1, -8, -18, "center", "top") in first  # two steps down, two shifts left
    assert len(set(first)) == len(first)


def test_each_label_takes_its_first_candidate_that_is_clear():
    candidates = [
        [(10, 10, 20, 20), (30, 30, 40, 40)],  # first is clear -> 0
        [(12, 12, 22, 22), (50, 50, 60, 60)],  # first hits label 0 -> 1
        [(70, 70, 80, 80), (85, 85, 95, 95)],  # first hits the obstacle -> 1
    ]
    assert choose_placements(candidates, [(65, 65, 75, 75)], BOUNDS) == [0, 1, 1]


def test_a_candidate_outside_the_axes_is_skipped():
    candidates = [[(95, 95, 105, 105), (40, 40, 50, 50)]]
    assert choose_placements(candidates, [], BOUNDS) == [1]


def test_a_candidate_nearer_another_models_bar_than_its_own_is_skipped():
    bars = [(10, 0, 10, 50), (40, 0, 40, 50)]
    candidates = [[(30, 20, 38, 30), (14, 20, 22, 30)],  # first is 2 from bar 1, 20 from its own
                  [(44, 20, 52, 30)]]
    assert choose_placements(candidates, [], BOUNDS, bars) == [1, 0]


def test_an_earlier_label_gives_up_its_first_choice_when_a_later_one_has_no_other():
    candidates = [[(10, 10, 20, 20), (30, 30, 40, 40)],
                  [(15, 15, 25, 25)]]  # only option overlaps label 0's first choice
    assert choose_placements(candidates, [], BOUNDS) == [1, 0]


def test_with_no_clear_candidate_the_first_one_is_used():
    candidates = [[(10, 10, 20, 20), (30, 30, 40, 40)]]
    assert choose_placements(candidates, [(0, 0, 100, 100)], BOUNDS) == [0]


def test_with_no_spot_nearest_its_own_bar_a_label_takes_the_least_ambiguous_clear_one():
    bars = [(50, 40, 50, 60), (37, 30, 37, 70), (66, 30, 66, 70)]  # own bar between longer ones
    candidates = [[(40, 20, 60, 34),   # 6 from its own bar, 3 from bar 1
                   (42, 20, 62, 34),   # 6 from its own, 5 from bar 1, 4 from bar 2
                   (40, 0, 60, 10)],   # far below: 30 from its own, ~20 from bar 1
                  [(10, 80, 20, 90)], [(80, 80, 90, 90)]]
    # own gap + 2 x (own gap - nearest other): 6 + 2 x 3 = 12, 6 + 2 x 2 = 10, 30 + 2 x ~10 = ~50
    assert choose_placements(candidates, [], BOUNDS, bars) == [1, 0, 0]


def test_a_label_with_no_spot_nearest_its_own_bar_chooses_before_its_neighbours():
    bars = [(50, 40, 50, 60), (37, 30, 37, 70), (66, 30, 66, 70)]
    candidates = [[(40, 20, 60, 34), (42, 20, 62, 34), (40, 0, 60, 10)],  # as above: 1 best
                  [(10, 80, 20, 90)],
                  [(55, 20, 64, 28),  # nearest its own bar, but would push label 0 far below
                   (68, 40, 78, 50)]]
    assert choose_placements(candidates, [], BOUNDS, bars) == [1, 0, 1]
