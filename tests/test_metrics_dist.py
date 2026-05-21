"""Tests for ``src/utils/metrics_dist.py`` (Stage 4B-3)."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics_dist import (
    col_sums,
    ks_stat_1d,
    marginal_compare,
    nonzero_ratio,
    row_sums,
)


# --- basic helpers --------------------------------------------------------


def test_nonzero_ratio_all_zero() -> None:
    assert nonzero_ratio(np.zeros((4, 4))) == 0.0


def test_nonzero_ratio_known_fraction() -> None:
    a = np.zeros((10,))
    a[0] = 1.0
    assert nonzero_ratio(a) == 0.1


def test_row_col_sums_correctness() -> None:
    od = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
    np.testing.assert_array_equal(row_sums(od), np.array([[3.0, 7.0]]))
    np.testing.assert_array_equal(col_sums(od), np.array([[4.0, 6.0]]))


# --- KS statistic ---------------------------------------------------------


def test_ks_stat_identical_is_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000)
    assert ks_stat_1d(a, a) == 0.0


def test_ks_stat_disjoint_supports_is_one() -> None:
    a = np.zeros(100)
    b = np.full(100, 10.0)
    assert ks_stat_1d(a, b) == 1.0


def test_ks_stat_separated_distributions_high() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 1000)
    b = rng.normal(5.0, 1.0, 1000)
    # 5-sigma offset -> KS statistic near 1.
    assert ks_stat_1d(a, b) > 0.9


def test_ks_stat_close_distributions_low() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 5000)
    b = rng.normal(0.0, 1.0, 5000)
    assert ks_stat_1d(a, b) < 0.1


def test_ks_stat_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ks_stat_1d(np.array([]), np.array([1.0, 2.0]))


# --- marginal_compare -----------------------------------------------------


def test_marginal_compare_real_vs_self_is_zero() -> None:
    rng = np.random.default_rng(0)
    od = rng.integers(0, 6, (10, 16, 16)).astype(np.float64)
    out = marginal_compare(od, od)
    assert out["row_sum_ks_stat"] == 0.0
    assert out["col_sum_ks_stat"] == 0.0
    assert out["real_nonzero_ratio"] == out["gen_nonzero_ratio"]
    assert out["real_row_sum_mean"] == pytest.approx(out["gen_row_sum_mean"])
    assert out["real_col_sum_std"] == pytest.approx(out["gen_col_sum_std"])


def test_marginal_compare_all_zero_gen_distinguishable() -> None:
    rng = np.random.default_rng(0)
    real = rng.integers(0, 6, (10, 16, 16)).astype(np.float64)
    gen = np.zeros_like(real)
    out = marginal_compare(real, gen)
    assert out["gen_nonzero_ratio"] == 0.0
    assert out["real_nonzero_ratio"] > 0.5
    # Real row/col sums spread across positive values; gen are all 0 -> high KS.
    assert out["row_sum_ks_stat"] > 0.5
    assert out["col_sum_ks_stat"] > 0.5


def test_marginal_compare_returns_flat_floats() -> None:
    rng = np.random.default_rng(0)
    real = rng.integers(0, 4, (4, 8, 8)).astype(np.float64)
    gen = rng.integers(0, 4, (4, 8, 8)).astype(np.float64)
    out = marginal_compare(real, gen)
    # Every value is a plain Python float (safe for json.dump / TB).
    for k, v in out.items():
        assert isinstance(v, float), f"{k} -> {type(v).__name__}"


def test_marginal_compare_rejects_low_dim() -> None:
    with pytest.raises(ValueError):
        marginal_compare(np.zeros(4), np.zeros(4))


def test_marginal_compare_allows_unequal_n() -> None:
    """Generated stack size doesn't have to match real stack size."""
    rng = np.random.default_rng(0)
    real = rng.integers(0, 4, (8, 8, 8)).astype(np.float64)
    gen = rng.integers(0, 4, (3, 8, 8)).astype(np.float64)
    out = marginal_compare(real, gen)
    assert "row_sum_ks_stat" in out and "col_sum_ks_stat" in out
