"""Tests for src/eval/posthoc_calibration.py (Stage 4B-5C PR5C-1A).

Uses only tiny synthetic matrices. Never reads real OD data or any
diffusion checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.posthoc_calibration import (
    DEFAULT_FAILED_DIFFUSION_BASELINE,
    acceptance_verdict,
    apply_threshold_and_scale,
    clip_nonnegative,
    compute_reference_stats,
    evaluate_calibrated_samples,
    grid_search_tau_scale,
    sparse_aggregate_stats,
    topk_pair_overlap,
)


# --- 1. clip_nonnegative does NOT use np.abs -----------------------------


def test_clip_nonnegative_pins_user_example() -> None:
    x = np.array([-2.0, -0.5, 0.0, 1.0])
    got = clip_nonnegative(x)
    np.testing.assert_array_equal(got, np.array([0.0, 0.0, 0.0, 1.0]))


def test_clip_nonnegative_distinguishable_from_abs() -> None:
    """A test that would PASS for np.maximum and FAIL for np.abs."""
    x = np.array([-3.5, -1.0, 0.0, 2.7])
    got = clip_nonnegative(x)
    # np.maximum(x, 0):
    np.testing.assert_array_equal(got, np.array([0.0, 0.0, 0.0, 2.7]))
    # np.abs would give [3.5, 1.0, 0.0, 2.7] -- explicitly different.
    abs_x = np.abs(x)
    assert not np.array_equal(got, abs_x), (
        "clip_nonnegative output equals np.abs(x); implementation might be "
        "using abs instead of np.maximum(x, 0)"
    )


def test_clip_nonnegative_preserves_shape_for_3d() -> None:
    x = np.full((3, 4, 5), -1.0)
    got = clip_nonnegative(x)
    assert got.shape == (3, 4, 5)
    assert (got == 0).all()


# --- 2. apply_threshold_and_scale ----------------------------------------


def test_apply_threshold_filters_below_tau() -> None:
    x = np.array([[0.1, 0.5, 1.0], [1.5, 2.0, 3.0]], dtype=np.float64)
    out = apply_threshold_and_scale(x, tau=1.0, scale=1.0)
    # tau=1.0 -> keep entries strictly > 1.0
    expected = np.array([[0, 0, 0], [2, 2, 3]], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.int32


def test_apply_threshold_high_tau_yields_all_zeros() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = apply_threshold_and_scale(x, tau=100.0, scale=1.0)
    assert out.shape == x.shape
    assert (out == 0).all()
    assert out.dtype == np.int32


def test_apply_scale_changes_magnitude() -> None:
    x = np.array([[5.0, 5.0]])
    out_a = apply_threshold_and_scale(x, tau=0.0, scale=1.0)
    out_b = apply_threshold_and_scale(x, tau=0.0, scale=0.5)
    out_c = apply_threshold_and_scale(x, tau=0.0, scale=2.0)
    np.testing.assert_array_equal(out_a.ravel(), [5, 5])
    np.testing.assert_array_equal(out_b.ravel(), [2, 2])  # round(2.5) == 2 (banker's)
    np.testing.assert_array_equal(out_c.ravel(), [10, 10])


def test_apply_threshold_clips_negatives_before_round() -> None:
    x = np.array([-5.0, -0.1, 0.5, 3.0])
    out = apply_threshold_and_scale(x, tau=0.4, scale=1.0)
    # -5 and -0.1 -> clipped to 0 -> not > tau -> 0
    # 0.5 > 0.4 -> round(0.5) = 0 (banker's rounding)
    # 3.0 > 0.4 -> 3
    np.testing.assert_array_equal(out, np.array([0, 0, 0, 3], dtype=np.int32))


def test_apply_output_is_nonnegative_int32() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.5, scale=2.0, size=(3, 16, 16))
    out = apply_threshold_and_scale(x, tau=0.3, scale=1.5)
    assert out.dtype == np.int32
    assert out.shape == x.shape
    assert out.min() >= 0


def test_apply_rejects_negative_tau_scale() -> None:
    x = np.zeros((2, 2))
    with pytest.raises(ValueError, match="tau"):
        apply_threshold_and_scale(x, tau=-0.1, scale=1.0)
    with pytest.raises(ValueError, match="scale"):
        apply_threshold_and_scale(x, tau=0.0, scale=-1.0)


# --- 3. sparse_aggregate_stats ------------------------------------------


def test_sparse_aggregate_stats_2d() -> None:
    arr = np.array([[0, 1, 0], [0, 0, 2], [3, 0, 0]], dtype=np.int32)
    s = sparse_aggregate_stats(arr)
    # 3 nonzero of 9 cells.
    assert s["nonzero_ratio"] == pytest.approx(3 / 9)
    assert s["total_mass"] == pytest.approx(6.0)
    assert s["max"] == pytest.approx(3.0)
    assert s["mean"] == pytest.approx(6 / 9)
    np.testing.assert_array_equal(s["row_sums"], [1, 2, 3])
    np.testing.assert_array_equal(s["col_sums"], [3, 1, 2])
    assert np.isfinite(s["entropy"])


def test_sparse_aggregate_stats_3d_summary() -> None:
    a = np.zeros((3, 4, 4), dtype=np.int32)
    a[0, 0, 1] = 5
    a[1, 1, 2] = 7
    a[2, 2, 3] = 9
    s = sparse_aggregate_stats(a)
    assert s["n_samples"] == 3
    np.testing.assert_array_equal(
        s["total_mass_per_sample"], np.array([5.0, 7.0, 9.0])
    )
    assert s["total_mass_mean"] == pytest.approx(21 / 3)
    np.testing.assert_array_equal(
        s["max_per_sample"], np.array([5.0, 7.0, 9.0])
    )
    assert s["max_mean"] == pytest.approx(7.0)
    assert s["row_sums_per_sample"].shape == (3, 4)
    assert s["col_sums_per_sample"].shape == (3, 4)
    assert np.all(np.isfinite(s["entropy_per_sample"]))


def test_sparse_aggregate_stats_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D or 3-D"):
        sparse_aggregate_stats(np.zeros(5))


# --- 4. compute_reference_stats + topk_pair_overlap ---------------------


def _build_known_topk_matrix() -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    """Two 5x5 matrices where the top-k pairs are hand-controlled.

    Diagonal is zero on both (matches Stage-3 drop_intra_zone=True).
    """
    z = 5
    real = np.zeros((z, z), dtype=np.float64)
    gen = np.zeros((z, z), dtype=np.float64)
    # Real top-3 off-diagonal: (0,1)=10, (2,3)=9, (1,4)=8
    real[0, 1] = 10
    real[2, 3] = 9
    real[1, 4] = 8
    real[3, 2] = 1  # noise
    real[4, 0] = 1  # noise
    # Gen top-5 off-diagonal: (0,1)=20 (hit), (1,4)=18 (hit), (4,2)=15 (miss),
    # (2,3)=12 (hit), (3,1)=8 (miss). So real_top_3 ∩ gen_top_5 = 3.
    gen[0, 1] = 20
    gen[1, 4] = 18
    gen[4, 2] = 15
    gen[2, 3] = 12
    gen[3, 1] = 8
    # Diagonal entries that should be IGNORED with exclude_diagonal=True.
    real[0, 0] = 1000.0
    gen[0, 0] = 9999.0
    real_top_3 = {0 * z + 1, 2 * z + 3, 1 * z + 4}
    gen_top_5 = {0 * z + 1, 1 * z + 4, 4 * z + 2, 2 * z + 3, 3 * z + 1}
    return real, gen, real_top_3, gen_top_5


def test_topk_pair_overlap_known_values() -> None:
    real, gen, real_top_3, gen_top_5 = _build_known_topk_matrix()
    overlap = topk_pair_overlap(gen, real, real_k=3, gen_k=5, exclude_diagonal=True)
    assert overlap == len(real_top_3 & gen_top_5) == 3


def test_topk_pair_overlap_diagonal_excluded_by_default() -> None:
    real, gen, _, _ = _build_known_topk_matrix()
    # With diagonal excluded, real_top_1 = (0,1); with diagonal included,
    # real_top_1 = (0,0). Verify the function picks the former.
    overlap_off = topk_pair_overlap(
        gen, real, real_k=1, gen_k=1, exclude_diagonal=True
    )
    # gen top-1 off-diagonal is (0,1), real top-1 off-diagonal is (0,1)
    # -> overlap 1.
    assert overlap_off == 1
    overlap_on = topk_pair_overlap(
        gen, real, real_k=1, gen_k=1, exclude_diagonal=False
    )
    # gen top-1 INCLUDING diagonal is (0,0); real top-1 is also (0,0)
    # -> overlap 1. (Same numeric result but for a different reason; the
    # discriminating fact is that the chosen pair is the diagonal.)
    assert overlap_on == 1
    # Now compare against a real with diagonal=0 but gen diag huge.
    real2 = real.copy()
    real2[0, 0] = 0.0
    gen2 = gen.copy()
    overlap_diag_excluded = topk_pair_overlap(
        gen2, real2, real_k=1, gen_k=1, exclude_diagonal=True
    )
    # Both real and gen top-1 off-diag is (0,1). Overlap = 1.
    assert overlap_diag_excluded == 1
    overlap_diag_included = topk_pair_overlap(
        gen2, real2, real_k=1, gen_k=1, exclude_diagonal=False
    )
    # gen top-1 including diag = (0,0); real top-1 = (0,1). No overlap.
    assert overlap_diag_included == 0


def test_topk_pair_overlap_is_deterministic_on_ties() -> None:
    z = 4
    arr = np.zeros((z, z))
    arr[0, 1] = 5
    arr[0, 2] = 5
    arr[1, 2] = 5  # three-way tie
    # Two calls must return the same set (deterministic stable sort).
    a = topk_pair_overlap(arr, arr, real_k=2, gen_k=2, exclude_diagonal=True)
    b = topk_pair_overlap(arr, arr, real_k=2, gen_k=2, exclude_diagonal=True)
    assert a == b
    # And the count is exactly 2 (real==gen, top-2 set fully overlaps).
    assert a == 2


def test_compute_reference_stats_keys() -> None:
    real, _, real_top_3, _ = _build_known_topk_matrix()
    ref = compute_reference_stats(real, real_k=3, exclude_diagonal=True)
    assert ref["real_top3_pairs"] == real_top_3
    assert ref["real_k"] == 3
    assert ref["exclude_diagonal"] is True
    assert ref["real_nonzero_ratio"] > 0
    assert ref["real_total_mass"] > 0
    assert np.isfinite(ref["real_entropy"])
    assert ref["real_row_sums"].shape == (5,)
    assert ref["real_col_sums"].shape == (5,)


# --- 5. grid_search_tau_scale ------------------------------------------


def _synthetic_signal_plus_noise(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [N, Z, Z] continuous samples + matching real_train_agg.

    Construction: 4 known signal cells per sample with continuous value
    ~5.0; the remaining 32 cells have small noise (~0.2). The TRUE
    optimum should be roughly tau ~ 1.0, scale ~ 1.0 -- it suppresses
    the 0.2 noise and keeps the 5.0 signal.

    ``real_train_agg`` mirrors the signal: identical 4 cells with
    integer value 5, all other cells 0.
    """
    rng = np.random.default_rng(seed)
    n, z = 6, 6
    samples = rng.uniform(0.0, 0.4, size=(n, z, z)).astype(np.float64)
    # Signal cells (off-diagonal).
    signal_cells = [(0, 1), (1, 3), (2, 5), (4, 0)]
    for i in range(n):
        for r, c in signal_cells:
            samples[i, r, c] = 5.0 + rng.normal(0.0, 0.1)
    real_agg = np.zeros((z, z), dtype=np.int32)
    for r, c in signal_cells:
        real_agg[r, c] = 5
    return samples, real_agg


def test_grid_search_finds_better_than_worst() -> None:
    samples, real_agg = _synthetic_signal_plus_noise(seed=0)
    best_tau, best_scale, best_metrics, grid = grid_search_tau_scale(
        samples,
        real_agg,
        tau_grid=np.linspace(0.1, 2.0, 20),
        scale_grid=np.geomspace(1e-3, 2.0, 20),
        lambda_mass=1.0,
    )
    objs = np.array([g["objective"] for g in grid])
    assert best_metrics["objective"] == float(objs.min())
    # Best objective must be MUCH smaller than worst point on this
    # signal-vs-noise toy: the optimum should be near 0; worst point
    # (e.g. tau=0.1, scale=2.0) produces orders-of-magnitude over-mass.
    assert best_metrics["objective"] < 0.5 * float(objs.max())
    # And the chosen (tau, scale) lies inside the grid.
    assert any(g["tau"] == best_tau and g["scale"] == best_scale for g in grid)


def test_grid_search_default_grids_are_consistent() -> None:
    """Defaults are documented; pin them so a silent change shows up."""
    samples, real_agg = _synthetic_signal_plus_noise(seed=1)
    _, _, _, grid = grid_search_tau_scale(samples, real_agg)
    # 20 x 20 = 400 grid points.
    assert len(grid) == 400


def test_grid_search_rejects_val_test_misuse_by_contract() -> None:
    """Sanity: ``real_train_agg`` shape must be 2-D, not [N, Z, Z]."""
    samples, real_agg = _synthetic_signal_plus_noise(seed=2)
    bad = np.stack([real_agg, real_agg], axis=0)  # [2, Z, Z]
    with pytest.raises(ValueError, match="2-D"):
        grid_search_tau_scale(samples, bad)


def test_grid_search_rejects_zero_reference() -> None:
    samples, _ = _synthetic_signal_plus_noise(seed=3)
    z = samples.shape[-1]
    with pytest.raises(ValueError, match="nonzero ratio or zero total mass"):
        grid_search_tau_scale(samples, np.zeros((z, z), dtype=np.int32))


# --- 6. evaluate_calibrated_samples ------------------------------------


def _expected_eval_keys() -> set[str]:
    return {
        "tau",
        "scale",
        "n_samples",
        "real_nonzero_ratio",
        "real_total_mass",
        "real_entropy",
    }


def _expected_per_sample_metric_keys() -> set[str]:
    return {
        "nonzero_ratio_x_real",
        "total_mass_ratio",
        "row_sum_ks_stat",
        "col_sum_ks_stat",
        "top20_pair_overlap",
        "top20_pair_overlap_against_top50",
        "gen_max",
        "gen_mean",
        "entropy",
    }


def test_evaluate_calibrated_samples_3d() -> None:
    samples, real_agg = _synthetic_signal_plus_noise(seed=5)
    out = evaluate_calibrated_samples(samples, real_agg, tau=1.0, scale=1.0)
    assert out["n_samples"] == samples.shape[0]
    keys = set(out.keys())
    assert _expected_eval_keys().issubset(keys)
    for k in _expected_per_sample_metric_keys():
        assert f"{k}_mean" in out and f"{k}_std" in out and f"{k}_per_sample" in out
        assert np.isfinite(out[f"{k}_mean"])
        assert np.isfinite(out[f"{k}_std"])


def test_evaluate_calibrated_samples_2d() -> None:
    _, real_agg = _synthetic_signal_plus_noise(seed=6)
    # Single matrix in continuous space.
    cont = np.zeros_like(real_agg, dtype=np.float64)
    cont[0, 1] = 5.0
    cont[1, 3] = 5.0
    cont[2, 5] = 5.0
    cont[4, 0] = 5.0
    cont[3, 3] = 0.2  # noise below tau
    out = evaluate_calibrated_samples(cont, real_agg, tau=1.0, scale=1.0)
    assert out["n_samples"] == 1
    for k in _expected_per_sample_metric_keys():
        assert k in out
        assert np.isfinite(out[k])


def test_evaluate_ks_stats_are_finite_under_zero_calibration() -> None:
    """All-zero calibrated output must yield finite KS, not NaN."""
    samples, real_agg = _synthetic_signal_plus_noise(seed=7)
    # Huge tau collapses every cell to 0.
    out = evaluate_calibrated_samples(samples, real_agg, tau=1e9, scale=1.0)
    assert np.isfinite(out["row_sum_ks_stat_mean"])
    assert np.isfinite(out["col_sum_ks_stat_mean"])
    assert out["nonzero_ratio_x_real_mean"] == pytest.approx(0.0)
    assert out["total_mass_ratio_mean"] == pytest.approx(0.0)


# --- 7. acceptance_verdict ---------------------------------------------


def _pass_like_metrics() -> dict[str, float]:
    return {
        "nonzero_ratio_x_real_mean": 1.0,
        "total_mass_ratio_mean": 1.0,
        "top20_pair_overlap_mean": 16.0,
        "row_sum_ks_stat_mean": 0.10,
        "col_sum_ks_stat_mean": 0.10,
    }


def test_acceptance_pass() -> None:
    assert acceptance_verdict(_pass_like_metrics()) == "pass"


def test_acceptance_mild_when_one_pass_gate_fails() -> None:
    m = _pass_like_metrics()
    # Bump nonzero ratio above 1.5x (PASS fails) but well below baseline 143.
    m["nonzero_ratio_x_real_mean"] = 2.7
    assert acceptance_verdict(m) == "mild"


def test_acceptance_mild_when_top20_below_floor() -> None:
    m = _pass_like_metrics()
    m["top20_pair_overlap_mean"] = 10.0  # below 12
    assert acceptance_verdict(m) == "mild"


def test_acceptance_fail_when_matches_diffusion_baseline() -> None:
    baseline = DEFAULT_FAILED_DIFFUSION_BASELINE
    m = {
        "nonzero_ratio_x_real_mean": baseline["nonzero_ratio_x_real"],
        "total_mass_ratio_mean": baseline["total_mass_ratio"],
        "top20_pair_overlap_mean": 0.0,
        "row_sum_ks_stat_mean": baseline["row_sum_ks_stat"],
        "col_sum_ks_stat_mean": baseline["col_sum_ks_stat"],
    }
    assert acceptance_verdict(m) == "fail"


def test_acceptance_uses_bare_key_when_no_mean_suffix() -> None:
    """Single-sample dict (no `_mean` suffix) is also accepted."""
    m = {
        "nonzero_ratio_x_real": 1.0,
        "total_mass_ratio": 1.0,
        "top20_pair_overlap": 14.0,
        "row_sum_ks_stat": 0.2,
        "col_sum_ks_stat": 0.2,
    }
    assert acceptance_verdict(m) == "pass"


def test_acceptance_thresholds_parametrisable() -> None:
    m = _pass_like_metrics()
    m["top20_pair_overlap_mean"] = 8.0
    # Default floor is 12 -> mild. Loosen the floor to 6 -> pass.
    assert acceptance_verdict(m) == "mild"
    assert (
        acceptance_verdict(m, pass_top20_overlap_min=6.0) == "pass"
    )
