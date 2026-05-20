"""Tests for src/data/od_dataset.py (Stage-4A: OD-slice dataset).

Uses a small synthetic OD tensor written to a tmp_path; no full
od_evtol.npy read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.od_dataset import (
    ODDataset,
    apply_norm,
    apply_norm_zero_pinned,
    compute_norm_stats,
    compute_norm_stats_nonzero,
    inverse_norm,
    inverse_norm_zero_pinned,
    load_norm_stats,
    next_pad_size,
    pad_hw,
    save_norm_stats,
    slot_to_condition,
    unpad,
)

# Synthetic tensor: 528 slots (11 days x 48), 12 zones (pads to 16).
T = 528
Z = 12
START = "2023-07-10 00:00:00"  # a Monday


@pytest.fixture
def od_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a small synthetic od_evtol.npy + od_meta.json; return paths."""
    rng = np.random.default_rng(0)
    # Long-tailed counts: mostly zeros, a few large -- mimics real OD.
    od = (rng.exponential(scale=1.5, size=(T, Z, Z)) * (rng.random((T, Z, Z)) < 0.3))
    od = od.astype(np.int32)
    od_path = tmp_path / "od_evtol.npy"
    np.save(od_path, od)

    meta = {"T": T, "n_zones": Z, "time_bin_min": 30, "start_datetime": START}
    meta_path = tmp_path / "od_meta.json"
    meta_path.write_text(json.dumps(meta))
    return od_path, meta_path


# --- padding helpers -------------------------------------------------------


def test_next_pad_size() -> None:
    assert next_pad_size(530, 16) == 544
    assert next_pad_size(12, 16) == 16
    assert next_pad_size(16, 16) == 16
    assert next_pad_size(1024, 16) == 1024


def test_pad_unpad_roundtrip() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((3, 12, 12)).astype(np.float32)
    padded = pad_hw(x, 16)
    assert padded.shape == (3, 16, 16)
    # Padded region is zero; original sits top-left.
    assert np.all(padded[:, 12:, :] == 0.0)
    assert np.all(padded[:, :, 12:] == 0.0)
    np.testing.assert_array_equal(unpad(padded, 12), x)


def test_pad_hw_rejects_too_small_pad_size() -> None:
    with pytest.raises(ValueError):
        pad_hw(np.zeros((1, 20, 20)), 16)


# --- normalization ---------------------------------------------------------


def test_norm_inverse_roundtrip_unclipped() -> None:
    # Large clip_val -> nothing is clipped -> exact round-trip.
    stats = {"mu": 0.4, "sigma": 1.1, "clip_val": 100.0}
    counts = np.array([0, 1, 5, 20, 100, 3000], dtype=np.float64)
    back = inverse_norm(apply_norm(counts, stats), stats)
    np.testing.assert_allclose(back, counts, rtol=1e-6, atol=1e-6)


def test_apply_norm_in_range_and_finite() -> None:
    stats = {"mu": 0.4, "sigma": 1.1, "clip_val": 3.0}
    counts = np.array([0, 1, 10, 10_000_000], dtype=np.float64)
    x = apply_norm(counts, stats)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_inverse_norm_is_nonnegative() -> None:
    stats = {"mu": 0.4, "sigma": 1.1, "clip_val": 3.0}
    x = np.linspace(-1.0, 1.0, 50)
    assert np.all(inverse_norm(x, stats) >= 0.0)


def test_save_load_norm_stats(tmp_path: Path) -> None:
    stats = {"mu": 0.123, "sigma": 0.987, "clip_val": 3.0}
    path = tmp_path / "stats.json"
    save_norm_stats(stats, path)
    loaded = load_norm_stats(path)
    assert loaded == pytest.approx(stats)


def test_compute_norm_stats_matches_numpy(od_files: tuple[Path, Path]) -> None:
    od_path, _ = od_files
    od = np.load(od_path)
    train_slots = range(0, 9 * 48)
    stats = compute_norm_stats(od, train_slots, clip_val=3.0)
    ref = np.log1p(od[: 9 * 48].astype(np.float64))
    assert stats["mu"] == pytest.approx(float(ref.mean()))
    assert stats["sigma"] == pytest.approx(float(ref.std()))
    assert stats["clip_val"] == 3.0


# --- conditioning ----------------------------------------------------------


def test_slot_to_condition() -> None:
    import datetime as dt

    start = dt.datetime.fromisoformat(START)  # Monday
    # slot 0: 00:00 Monday.
    assert slot_to_condition(0, start, 48, 30) == {
        "hour": 0,
        "day_of_week": 0,
        "is_weekend": 0,
    }
    # slot 5: 02:00 (5 * 30 min = 150 min), still Monday.
    assert slot_to_condition(5, start, 48, 30)["hour"] == 2
    # day 5 (slot 5*48=240): Saturday -> weekend.
    cond = slot_to_condition(5 * 48, start, 48, 30)
    assert cond["day_of_week"] == 5 and cond["is_weekend"] == 1


# --- ODDataset -------------------------------------------------------------


def test_dataset_length_per_split(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    train = ODDataset(od_path, meta_path, "train")
    val = ODDataset(od_path, meta_path, "val", norm_stats=train.norm_stats)
    test = ODDataset(od_path, meta_path, "test", norm_stats=train.norm_stats)
    assert len(train) == 9 * 48   # days 0..8
    assert len(val) == 48         # day 9
    assert len(test) == 48        # day 10


def test_dataset_window_reduces_length(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "val", window=4)
    assert len(ds) == 48 - 4 + 1


def test_sample_shape(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "train")
    assert ds.pad_size == 16  # Z=12 padded to multiple of 16
    x, _ = ds[0]
    assert x.shape == (1, 16, 16)
    assert x.dtype == np.float32

    ds_w = ODDataset(od_path, meta_path, "train", window=3)
    x_w, _ = ds_w[0]
    assert x_w.shape == (3, 16, 16)


def test_sample_condition_correct(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "train")
    # Sample 0 -> slot 0 -> 00:00 Monday.
    _, c0 = ds[0]
    assert c0 == {"hour": 0, "day_of_week": 0, "is_weekend": 0}
    # Sample 5 -> slot 5 -> 02:00 Monday.
    _, c5 = ds[5]
    assert c5["hour"] == 2 and c5["day_of_week"] == 0

    # val starts at slot 432 (day 9): (Monday + 9) % 7 == Wednesday.
    val = ODDataset(od_path, meta_path, "val", norm_stats=ds.norm_stats)
    _, cv = val[0]
    assert cv["day_of_week"] == 2 and cv["is_weekend"] == 0


def test_normalization_no_nan_and_in_range(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "train")
    for idx in (0, len(ds) // 2, len(ds) - 1):
        x, _ = ds[idx]
        assert np.all(np.isfinite(x))
        assert x.min() >= -1.0 and x.max() <= 1.0


def test_padded_region_is_constant(od_files: tuple[Path, Path]) -> None:
    # The padded raw zeros must normalize to the same value everywhere.
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "train")
    x, _ = ds[0]
    pad_block = x[:, 12:, :]
    assert np.all(pad_block == pad_block.flat[0])


def test_inverse_transform_nonneg_and_unpads(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "train")
    x, _ = ds[10]
    back = ds.inverse_transform(x)
    assert back.shape == (1, Z, Z)
    assert np.all(back >= 0.0)


def test_inverse_transform_recovers_small_counts(od_files: tuple[Path, Path]) -> None:
    # With clip_val high enough, a real slice round-trips through the
    # dataset transform within rounding tolerance.
    od_path, meta_path = od_files
    ds = ODDataset(od_path, meta_path, "train", clip_val=50.0)
    raw = np.load(od_path)[10].astype(np.float64)
    x, _ = ds[10]
    back = ds.inverse_transform(x)[0]
    np.testing.assert_allclose(back, raw, rtol=1e-4, atol=1e-3)


def test_val_does_not_recompute_stats(od_files: tuple[Path, Path]) -> None:
    # Passing norm_stats must be used verbatim (no leak from val data).
    od_path, meta_path = od_files
    given = {"mu": 0.1, "sigma": 2.0, "clip_val": 3.0}
    val = ODDataset(od_path, meta_path, "val", norm_stats=given)
    assert val.norm_stats == given


def test_rejects_bad_split(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    with pytest.raises(ValueError):
        ODDataset(od_path, meta_path, "validation")


# --- Stage 4B-5B: zero_pinned_nonzero scheme ------------------------------


_ZPIN_STATS = {
    "mu_nz": 0.7,
    "sigma_nz": 0.13,
    "clip_val": 20.0,
    "scheme": "zero_pinned_nonzero",
}


def test_zpin_apply_norm_pins_zero_to_minus_one() -> None:
    raw = np.array([[0, 1, 5], [0, 20, 0]], dtype=np.int32)
    x = apply_norm_zero_pinned(raw, _ZPIN_STATS)
    assert x.shape == raw.shape
    assert x.dtype == np.float32
    # Every raw-zero entry lands at exactly -1.0.
    assert np.all(x[raw == 0] == -1.0)
    # Every raw-nonzero entry sits strictly above -1.0 (and well above
    # the -0.5 inverse threshold under realistic stats).
    assert np.all(x[raw > 0] > -1.0)
    assert np.all(x[raw > 0] > -0.5)


def test_zpin_inverse_restores_zero_exactly() -> None:
    x = np.full((4, 4), -1.0, dtype=np.float32)
    back = inverse_norm_zero_pinned(x, _ZPIN_STATS)
    assert np.all(back == 0.0)


def test_zpin_roundtrip_small_counts() -> None:
    raw = np.array([0, 1, 5, 20], dtype=np.float64)
    x = apply_norm_zero_pinned(raw, _ZPIN_STATS)
    back = inverse_norm_zero_pinned(x, _ZPIN_STATS)
    # Zero is restored exact; small nonzero counts within rounding.
    assert back[0] == 0.0
    np.testing.assert_allclose(back[1:], raw[1:], rtol=1e-4, atol=1e-3)


def test_zpin_inverse_is_nonnegative() -> None:
    grid = np.linspace(-1.0, 1.0, 200)
    out = inverse_norm_zero_pinned(grid, _ZPIN_STATS)
    assert np.all(out >= 0.0)
    assert np.all(np.isfinite(out))


def test_compute_norm_stats_nonzero_matches_numpy(
    od_files: tuple[Path, Path],
) -> None:
    od_path, _ = od_files
    od = np.load(od_path)
    train_slots = range(0, 9 * 48)
    stats = compute_norm_stats_nonzero(od, train_slots, clip_val=20.0)
    train_block = od[: 9 * 48]
    nz = train_block[train_block > 0].astype(np.float64)
    ref = np.log1p(nz)
    assert stats["mu_nz"] == pytest.approx(float(ref.mean()))
    assert stats["sigma_nz"] == pytest.approx(float(ref.std()))
    assert stats["clip_val"] == 20.0
    assert stats["scheme"] == "zero_pinned_nonzero"


def test_dataset_zpin_scheme_padding_and_pin(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(
        od_path, meta_path, "train",
        scheme="zero_pinned_nonzero", clip_val=20.0,
    )
    assert ds.scheme == "zero_pinned_nonzero"
    assert ds.norm_stats["scheme"] == "zero_pinned_nonzero"
    assert "mu_nz" in ds.norm_stats and "sigma_nz" in ds.norm_stats

    x, _ = ds[0]
    assert x.shape == (1, 16, 16)
    raw = np.load(od_path)[0]
    # Inside the Z x Z region, zero entries are pinned at -1.0.
    inside = x[0, :Z, :Z]
    assert np.all(inside[raw == 0] == -1.0)
    # Padded region (rows/cols Z..pad-1) is all -1.0.
    assert np.all(x[:, Z:, :] == -1.0)
    assert np.all(x[:, :, Z:] == -1.0)


def test_dataset_zpin_inverse_round_trip(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    ds = ODDataset(
        od_path, meta_path, "train",
        scheme="zero_pinned_nonzero", clip_val=20.0,
    )
    x, _ = ds[10]
    back = ds.inverse_transform(x)
    raw = np.load(od_path)[10].astype(np.float64)
    assert back.shape == (1, Z, Z)
    # Zero entries restored to exact 0; nonzero entries within tight
    # tolerance (synthetic counts are small, far below the clip ceiling).
    assert np.all(back[0][raw == 0] == 0.0)
    nz_mask = raw > 0
    if nz_mask.any():
        np.testing.assert_allclose(
            back[0][nz_mask], raw[nz_mask], rtol=1e-4, atol=1e-3
        )


def test_dataset_rejects_unknown_scheme(od_files: tuple[Path, Path]) -> None:
    od_path, meta_path = od_files
    with pytest.raises(ValueError):
        ODDataset(od_path, meta_path, "train", scheme="not_a_scheme")


def test_dataset_scheme_mismatch_in_norm_stats(
    od_files: tuple[Path, Path],
) -> None:
    """If norm_stats carries an explicit scheme tag it must match."""
    od_path, meta_path = od_files
    zpin_stats = {
        "mu_nz": 0.7, "sigma_nz": 0.13, "clip_val": 20.0,
        "scheme": "zero_pinned_nonzero",
    }
    with pytest.raises(ValueError):
        ODDataset(
            od_path, meta_path, "val",
            scheme="global_clip", norm_stats=zpin_stats,
        )


def test_norm_stats_json_preserves_scheme_tag(tmp_path: Path) -> None:
    # save+load preserves the scheme tag and the scheme-specific keys
    # for both schemes, and the two scheme caches occupy distinct files.
    stats_g = {"mu": 0.1, "sigma": 0.2, "clip_val": 100.0, "scheme": "global_clip"}
    stats_z = {
        "mu_nz": 0.7, "sigma_nz": 0.13, "clip_val": 20.0,
        "scheme": "zero_pinned_nonzero",
    }
    p_g = tmp_path / "stats_global.json"
    p_z = tmp_path / "stats_zpin.json"
    save_norm_stats(stats_g, p_g)
    save_norm_stats(stats_z, p_z)
    # Distinct files (mtime / inode irrelevant here, just contents).
    loaded_g = load_norm_stats(p_g)
    loaded_z = load_norm_stats(p_z)
    assert loaded_g["scheme"] == "global_clip"
    assert loaded_z["scheme"] == "zero_pinned_nonzero"
    assert loaded_g["mu"] == pytest.approx(0.1)
    assert loaded_z["mu_nz"] == pytest.approx(0.7)
    # Cross-scheme keys do not appear in the wrong file.
    assert "mu_nz" not in loaded_g
    assert "mu" not in loaded_z
