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
