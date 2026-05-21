"""Greedy marginal-coverage baseline for vertiport placement.

At each of ``k_select`` steps the greedy heuristic adds the single
candidate that maximizes the *marginal* gain in bilateral OD coverage.
Bilateral coverage is identical to ``VertiportEnv``'s reward: an OD pair
``(i, j)`` counts as covered iff both origin zone ``i`` and destination
zone ``j`` lie in the currently covered zone set.

This is a diagnostic / paper baseline (the A1-family "greedy" reference)
-- it does not use the RL environment's ``step`` so it can scan every
unselected candidate's marginal gain at each step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def greedy_marginal_coverage(
    od: NDArray[np.integer],
    cand_covers_zones: NDArray[np.bool_],
    k_select: int,
) -> dict[str, Any]:
    """Greedily place ``k_select`` vertiports by marginal bilateral coverage.

    Parameters
    ----------
    od:
        ``[Z, Z]`` nonnegative integer OD demand matrix for one scenario
        (rows = origin zone, cols = destination zone).
    cand_covers_zones:
        ``[C, Z]`` bool coverage mask -- ``True`` where a candidate
        covers a zone.
    k_select:
        Number of vertiports to place.

    Returns
    -------
    dict with keys:
        ``selected_candidates`` -- list of ``k_select`` candidate ids in
        selection order; ``coverage_ratio`` -- covered bilateral demand
        divided by total demand (0.0 if total demand is 0);
        ``total_covered_demand`` -- covered bilateral demand (int);
        ``gains`` -- per-step marginal demand gain (list of ints).

    Notes
    -----
    Deterministic: ties in marginal gain are broken toward the smaller
    candidate id. The marginal gain for adding candidate ``c`` decomposes
    so that no ``[C, Z, Z]`` tensor is ever materialized -- only a
    ``[C, Z]`` delta matrix and a single ``[C, Z] @ [Z, Z]`` matmul per
    step.
    """
    od_f = np.ascontiguousarray(od).astype(np.float64)
    cov = np.ascontiguousarray(cand_covers_zones).astype(bool)
    if od_f.ndim != 2 or od_f.shape[0] != od_f.shape[1]:
        raise ValueError(f"od must be [Z, Z]; got shape {od_f.shape}")
    if cov.ndim != 2 or cov.shape[1] != od_f.shape[0]:
        raise ValueError(
            f"cand_covers_zones must be [C, Z={od_f.shape[0]}]; "
            f"got shape {cov.shape}"
        )
    n_candidates = cov.shape[0]
    if not 1 <= k_select <= n_candidates:
        raise ValueError(
            f"k_select must be in [1, n_candidates={n_candidates}]; "
            f"got {k_select}"
        )

    total_demand = int(od_f.sum())

    covered = np.zeros(od_f.shape[0], dtype=bool)
    selected_mask = np.zeros(n_candidates, dtype=bool)
    selected: list[int] = []
    gains: list[int] = []

    for _ in range(k_select):
        # Per-candidate delta zone indicators: zones a candidate would
        # newly cover. delta is disjoint from `covered`, so the marginal
        # gain of adding c is
        #   gain_c = (p + q) . delta_c + delta_c @ od @ delta_c
        # where p = od @ covered and q = covered @ od.
        delta = (cov & ~covered).astype(np.float64)  # [C, Z]
        u = covered.astype(np.float64)
        p = od_f @ u  # [Z]: outflow from each origin to covered zones
        q = u @ od_f  # [Z]: inflow to each dest from covered zones
        linear = delta @ (p + q)  # [C]
        delta_od = delta @ od_f  # [C, Z]
        quadratic = np.einsum("cz,cz->c", delta_od, delta)  # [C]
        marginal = linear + quadratic  # [C]

        # Exclude already-selected candidates; argmax then breaks ties
        # toward the smaller candidate id (first occurrence).
        marginal[selected_mask] = -np.inf
        best_c = int(np.argmax(marginal))
        best_gain = int(round(float(marginal[best_c])))

        selected.append(best_c)
        selected_mask[best_c] = True
        covered |= cov[best_c]
        gains.append(best_gain)

    total_covered_demand = int(od_f[np.ix_(covered, covered)].sum())
    coverage_ratio = (
        total_covered_demand / total_demand if total_demand > 0 else 0.0
    )

    return {
        "selected_candidates": selected,
        "coverage_ratio": float(coverage_ratio),
        "total_covered_demand": total_covered_demand,
        "gains": gains,
    }


def evaluate_fixed_selection(
    od: NDArray[np.integer],
    cand_covers_zones: NDArray[np.bool_],
    selected_candidates: list[int] | tuple[int, ...] | NDArray[np.integer],
) -> dict[str, Any]:
    """Evaluate a fixed candidate set on one OD scenario.

    This uses the same bilateral coverage definition as
    :func:`greedy_marginal_coverage` and ``VertiportEnv`` but does not
    optimize the candidate list. It is the core primitive for the PR4
    blind / robust diagnostics.
    """
    od_f = np.ascontiguousarray(od).astype(np.float64)
    cov = np.ascontiguousarray(cand_covers_zones).astype(bool)
    selected = [int(c) for c in selected_candidates]

    if od_f.ndim != 2 or od_f.shape[0] != od_f.shape[1]:
        raise ValueError(f"od must be [Z, Z]; got shape {od_f.shape}")
    if cov.ndim != 2 or cov.shape[1] != od_f.shape[0]:
        raise ValueError(
            f"cand_covers_zones must be [C, Z={od_f.shape[0]}]; "
            f"got shape {cov.shape}"
        )
    n_candidates = cov.shape[0]
    if any(c < 0 or c >= n_candidates for c in selected):
        raise ValueError(
            f"selected_candidates must be in [0, {n_candidates}); got {selected}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"selected_candidates contains duplicates: {selected}")

    covered = np.zeros(od_f.shape[0], dtype=bool)
    for c in selected:
        covered |= cov[c]

    total_demand = int(od_f.sum())
    total_covered_demand = int(od_f[np.ix_(covered, covered)].sum())
    coverage_ratio = (
        total_covered_demand / total_demand if total_demand > 0 else 0.0
    )
    return {
        "selected_candidates": selected,
        "coverage_ratio": float(coverage_ratio),
        "total_covered_demand": total_covered_demand,
        "total_demand": total_demand,
        "n_covered_zones": int(covered.sum()),
    }


def evaluate_fixed_selection_many(
    od_scenarios: NDArray[np.integer],
    cand_covers_zones: NDArray[np.bool_],
    selected_candidates: list[int] | tuple[int, ...] | NDArray[np.integer],
) -> dict[str, Any]:
    """Evaluate one fixed candidate sequence across all scenarios."""
    od = np.ascontiguousarray(od_scenarios)
    if od.ndim != 3 or od.shape[1] != od.shape[2]:
        raise ValueError(f"od_scenarios must be [S, Z, Z]; got shape {od.shape}")

    per = [
        evaluate_fixed_selection(od[s], cand_covers_zones, selected_candidates)
        for s in range(od.shape[0])
    ]
    coverages = np.array([r["coverage_ratio"] for r in per], dtype=np.float64)
    return {
        "selected_candidates": [int(c) for c in selected_candidates],
        "coverage_ratio_per_scenario": [float(c) for c in coverages],
        "mean_coverage": float(coverages.mean()),
        "std_coverage": float(coverages.std()),
        "min_coverage": float(coverages.min()),
        "max_coverage": float(coverages.max()),
        "total_covered_demand_per_scenario": [
            int(r["total_covered_demand"]) for r in per
        ],
    }


def greedy_blind_mean(
    od_scenarios: NDArray[np.integer],
    cand_covers_zones: NDArray[np.bool_],
    k_select: int,
) -> dict[str, Any]:
    """Run greedy once on the mean OD matrix, then evaluate fixed actions."""
    od = np.ascontiguousarray(od_scenarios)
    if od.ndim != 3 or od.shape[1] != od.shape[2]:
        raise ValueError(f"od_scenarios must be [S, Z, Z]; got shape {od.shape}")

    mean_od = od.astype(np.float64).mean(axis=0)
    greedy = greedy_marginal_coverage(mean_od, cand_covers_zones, k_select)
    fixed = evaluate_fixed_selection_many(
        od, cand_covers_zones, greedy["selected_candidates"]
    )
    fixed["mean_od_greedy_coverage_ratio"] = float(greedy["coverage_ratio"])
    fixed["mean_od_gains"] = [int(g) for g in greedy["gains"]]
    return fixed


def robust_cvar_greedy(
    od_scenarios: NDArray[np.integer],
    cand_covers_zones: NDArray[np.bool_],
    k_select: int,
    *,
    alpha: float = 0.3,
) -> dict[str, Any]:
    """Greedily select a fixed set by bottom-alpha marginal coverage gain.

    At each step every unselected candidate is scored by its marginal
    coverage-ratio gain on every scenario. The robust score is the mean
    over the bottom ``alpha`` fraction of scenarios. The selected set is
    then evaluated on all scenarios by :func:`evaluate_fixed_selection_many`.
    """
    od = np.ascontiguousarray(od_scenarios).astype(np.float64)
    cov = np.ascontiguousarray(cand_covers_zones).astype(bool)
    if od.ndim != 3 or od.shape[1] != od.shape[2]:
        raise ValueError(f"od_scenarios must be [S, Z, Z]; got shape {od.shape}")
    if cov.ndim != 2 or cov.shape[1] != od.shape[1]:
        raise ValueError(
            f"cand_covers_zones must be [C, Z={od.shape[1]}]; got shape {cov.shape}"
        )
    n_scenarios = od.shape[0]
    n_candidates = cov.shape[0]
    if not 1 <= k_select <= n_candidates:
        raise ValueError(
            f"k_select must be in [1, n_candidates={n_candidates}]; got {k_select}"
        )
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

    bottom_n = max(1, int(np.ceil(alpha * n_scenarios)))
    totals = od.reshape(n_scenarios, -1).sum(axis=1)
    denom = np.where(totals > 0.0, totals, 1.0)

    covered = np.zeros(od.shape[1], dtype=bool)
    selected_mask = np.zeros(n_candidates, dtype=bool)
    selected: list[int] = []
    robust_gains: list[float] = []

    for _ in range(k_select):
        delta = (cov & ~covered).astype(np.float64)  # [C, Z]
        u = covered.astype(np.float64)
        per_scenario_marginal = np.zeros((n_scenarios, n_candidates), dtype=np.float64)

        for s in range(n_scenarios):
            p = od[s] @ u
            q = u @ od[s]
            linear = delta @ (p + q)
            delta_od = delta @ od[s]
            quadratic = np.einsum("cz,cz->c", delta_od, delta)
            per_scenario_marginal[s] = (linear + quadratic) / denom[s]

        bottom = np.partition(per_scenario_marginal, bottom_n - 1, axis=0)[
            :bottom_n
        ]
        robust_score = bottom.mean(axis=0)
        robust_score[selected_mask] = -np.inf

        best_c = int(np.argmax(robust_score))
        selected.append(best_c)
        selected_mask[best_c] = True
        covered |= cov[best_c]
        robust_gains.append(float(robust_score[best_c]))

    fixed = evaluate_fixed_selection_many(od, cov, selected)
    fixed["alpha"] = float(alpha)
    fixed["bottom_n"] = int(bottom_n)
    fixed["robust_gains"] = robust_gains
    return fixed
