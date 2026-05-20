"""Distributional metrics for OD samples (Stage 4B-3 in-loop diagnostics).

Cheap, dependency-free comparisons between a real OD sample stack and a
generated one. Used by ``experiments/run_stage4_train.py`` to log every
``save_every`` steps whether the diffusion model is producing OD slices
whose marginals match the data (row sums + column sums) and whose
sparsity matches the ~0.117 % real eVTOL nonzero ratio.

Heavy metrics (MMD with kernel choice, Jensen-Shannon, sample-grid
plots) belong in ``run_stage4_eval.py`` (Stage 4C) -- this module
intentionally stays small and side-effect free.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def nonzero_ratio(od: np.ndarray) -> float:
    """Fraction of nonzero entries in ``od`` (any shape)."""
    if od.size == 0:
        return float("nan")
    return float(np.count_nonzero(od)) / float(od.size)


def row_sums(od: np.ndarray) -> np.ndarray:
    """Sum over the last axis. ``[..., Z, Z]`` -> ``[..., Z]``."""
    return od.sum(axis=-1)


def col_sums(od: np.ndarray) -> np.ndarray:
    """Sum over the second-to-last axis. ``[..., Z, Z]`` -> ``[..., Z]``."""
    return od.sum(axis=-2)


def ks_stat_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic on 1-D samples.

    Pure-numpy implementation -- no scipy dependency. Returns the
    maximum absolute difference between the two empirical CDFs over
    the merged support, in ``[0, 1]``. Identical samples give exactly 0;
    disjoint supports give 1.
    """
    a_arr = np.asarray(a).ravel()
    b_arr = np.asarray(b).ravel()
    n_a = a_arr.size
    n_b = b_arr.size
    if n_a == 0 or n_b == 0:
        raise ValueError(f"ks_stat_1d: empty input (n_a={n_a}, n_b={n_b})")
    a_sorted = np.sort(a_arr)
    b_sorted = np.sort(b_arr)
    merged = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, merged, side="right") / n_a
    cdf_b = np.searchsorted(b_sorted, merged, side="right") / n_b
    return float(np.max(np.abs(cdf_a - cdf_b)))


def marginal_compare(real: np.ndarray, gen: np.ndarray) -> dict[str, Any]:
    """Compare a real OD stack to a generated one on cheap distributional stats.

    ``real`` and ``gen`` should both be non-negative real-valued arrays
    of shape ``[N, Z, Z]`` (or any leading batch dimension). The two
    stacks do NOT need to have the same ``N``: the per-stack means /
    stds are scale-invariant under sample-count, and the K-S statistic
    compares the empirical distributions of all row / column sums
    pooled across the stack.

    Returns a flat dict of float scalars so the result can be logged to
    TensorBoard or JSON line-by-line.
    """
    real_arr = np.asarray(real, dtype=np.float64)
    gen_arr = np.asarray(gen, dtype=np.float64)
    if real_arr.ndim < 2 or gen_arr.ndim < 2:
        raise ValueError(
            f"real/gen must have at least 2 trailing dims; got "
            f"real.shape={real_arr.shape}, gen.shape={gen_arr.shape}"
        )

    r_row = row_sums(real_arr).ravel()
    g_row = row_sums(gen_arr).ravel()
    r_col = col_sums(real_arr).ravel()
    g_col = col_sums(gen_arr).ravel()

    return {
        "real_nonzero_ratio": nonzero_ratio(real_arr),
        "gen_nonzero_ratio": nonzero_ratio(gen_arr),
        "real_row_sum_mean": float(r_row.mean()),
        "real_row_sum_std": float(r_row.std()),
        "gen_row_sum_mean": float(g_row.mean()),
        "gen_row_sum_std": float(g_row.std()),
        "real_col_sum_mean": float(r_col.mean()),
        "real_col_sum_std": float(r_col.std()),
        "gen_col_sum_mean": float(g_col.mean()),
        "gen_col_sum_std": float(g_col.std()),
        "row_sum_ks_stat": ks_stat_1d(r_row, g_row),
        "col_sum_ks_stat": ks_stat_1d(r_col, g_col),
        "gen_min": float(gen_arr.min()),
        "gen_max": float(gen_arr.max()),
        "gen_mean": float(gen_arr.mean()),
    }
