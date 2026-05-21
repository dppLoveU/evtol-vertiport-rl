"""Unit tests for the Stage-6 PR3 greedy marginal-coverage baseline.

All tests use small hand-crafted synthetic data so the bilateral OD
coverage math is verifiable by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.baselines.greedy import greedy_marginal_coverage

# OD demand matrix (rows = origin, cols = destination); total = 38.
#   [[ 0, 10,  0,  0],
#    [ 5,  0,  0,  0],
#    [ 0,  0,  0, 20],
#    [ 0,  0,  3,  0]]
_M = np.array(
    [
        [0, 10, 0, 0],
        [5, 0, 0, 0],
        [0, 0, 0, 20],
        [0, 0, 3, 0],
    ],
    dtype=np.int32,
)
_TOTAL = 38

# Candidate -> covered zones:
#   cand 0 -> {0}, cand 1 -> {1}, cand 2 -> {2, 3}, cand 3 -> {0, 1}
_COV = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
        [True, True, False, False],
    ],
    dtype=bool,
)


def test_greedy_picks_handcomputed_optimum() -> None:
    # Step 1: cand 2 covers {2,3} -> bilateral demand 23 (best); others
    #         give 0 (c0/c1) or 15 (c3).
    # Step 2: cand 3 adds zones {0,1} -> full total 38, gain 15.
    # Step 3: nothing left to gain; tie at gain 0 -> smallest id, cand 0.
    res = greedy_marginal_coverage(_M, _COV, k_select=3)
    assert res["selected_candidates"] == [2, 3, 0]
    assert res["gains"] == [23, 15, 0]


def test_greedy_coverage_ratio_and_demand() -> None:
    res = greedy_marginal_coverage(_M, _COV, k_select=3)
    assert res["total_covered_demand"] == _TOTAL
    assert res["coverage_ratio"] == pytest.approx(1.0)

    # k=1: best single candidate is cand 2 with 23 covered demand.
    res1 = greedy_marginal_coverage(_M, _COV, k_select=1)
    assert res1["total_covered_demand"] == 23
    assert res1["coverage_ratio"] == pytest.approx(23.0 / _TOTAL)


def test_greedy_selection_has_no_duplicates() -> None:
    res = greedy_marginal_coverage(_M, _COV, k_select=4)
    selected = res["selected_candidates"]
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_greedy_gains_are_nonnegative() -> None:
    res = greedy_marginal_coverage(_M, _COV, k_select=4)
    assert all(g >= 0 for g in res["gains"])


def test_greedy_zero_total_demand_does_not_crash() -> None:
    od_zero = np.zeros((4, 4), dtype=np.int32)
    res = greedy_marginal_coverage(od_zero, _COV, k_select=3)
    assert res["coverage_ratio"] == 0.0
    assert res["total_covered_demand"] == 0
    assert len(res["selected_candidates"]) == 3
    assert len(set(res["selected_candidates"])) == 3
    assert res["gains"] == [0, 0, 0]


def test_greedy_tie_breaks_to_smaller_cand_id() -> None:
    # cand 0 covers {0}, cand 1 covers {1}; demand is symmetric so both
    # give an equal step-1 gain of 5 -- greedy must pick the smaller id.
    od = np.array([[5, 0], [0, 5]], dtype=np.int32)
    cov = np.array([[True, False], [False, True]], dtype=bool)
    res = greedy_marginal_coverage(od, cov, k_select=1)
    assert res["selected_candidates"] == [0]
    assert res["gains"] == [5]
