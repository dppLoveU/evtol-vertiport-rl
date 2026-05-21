"""Synthetic tests for Stage-6 PR4 formulation-audit helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.baselines.greedy import (
    evaluate_fixed_selection,
    greedy_blind_mean,
    robust_cvar_greedy,
)


def test_evaluate_fixed_selection_coverage_correct() -> None:
    od = np.array(
        [
            [0, 10, 0],
            [5, 0, 7],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    cov = np.array(
        [
            [True, True, False],
            [False, True, True],
        ],
        dtype=bool,
    )

    res = evaluate_fixed_selection(od, cov, [0])
    assert res["total_covered_demand"] == 15
    assert res["coverage_ratio"] == pytest.approx(15.0 / 22.0)


def test_greedy_blind_mean_returns_fixed_candidate_set() -> None:
    od = np.zeros((2, 3, 3), dtype=np.int32)
    od[0, 0, 1] = 10
    od[1, 1, 2] = 10
    cov = np.array(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ],
        dtype=bool,
    )

    res = greedy_blind_mean(od, cov, k_select=1)
    assert len(res["selected_candidates"]) == 1
    assert len(set(res["selected_candidates"])) == 1
    assert len(res["coverage_ratio_per_scenario"]) == 2


def test_robust_cvar_greedy_improves_worst_case_in_constructed_case() -> None:
    od = np.zeros((2, 4, 4), dtype=np.int32)
    # Scenario 0: candidate 0 is strong, candidate 1 is moderate.
    od[0, 0, 1] = 90
    od[0, 2, 3] = 30
    od[0, 0, 3] = 80
    # Scenario 1: candidate 0 is useless, candidate 1 remains moderate.
    od[1, 2, 3] = 30
    od[1, 0, 3] = 70
    cov = np.array(
        [
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    blind = greedy_blind_mean(od, cov, k_select=1)
    robust = robust_cvar_greedy(od, cov, k_select=1, alpha=0.5)

    assert blind["selected_candidates"] == [0]
    assert robust["selected_candidates"] == [1]
    assert min(robust["coverage_ratio_per_scenario"]) > min(
        blind["coverage_ratio_per_scenario"]
    )


def test_zero_demand_scenario_does_not_crash() -> None:
    od = np.zeros((2, 3, 3), dtype=np.int32)
    od[1, 0, 1] = 5
    cov = np.array(
        [
            [True, True, False],
            [False, True, True],
        ],
        dtype=bool,
    )

    fixed = evaluate_fixed_selection(od[0], cov, [0])
    robust = robust_cvar_greedy(od, cov, k_select=1, alpha=0.5)

    assert fixed["coverage_ratio"] == 0.0
    assert len(robust["selected_candidates"]) == 1
    assert np.isfinite(robust["coverage_ratio_per_scenario"]).all()
