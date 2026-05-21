"""Tests for src/data/bootstrap_od.py (Stage 4B-5C PR5C-2A).

Uses a tiny synthetic OD tensor; never reads the real Stage-3 file.

The synthetic tensor uses a distinct integer label per slot so that any
train/val/test leak shows up immediately in the aggregated sums and in
``used_slots``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.bootstrap_od import (
    ConditionalBootstrapSampler,
    VALID_MODES,
    aggregate_day_blocks,
    compute_bootstrap_summary,
    split_train_slots_into_day_blocks,
)

# Small but realistic synthetic geometry:
#   T = 192 slots = 4 days x 48 slots/day
#   Z = 3 zones
# Splits: train = days 0..1 (slots 0..95), val = day 2 (96..143),
#          test = day 3 (144..191).
SLOTS_PER_DAY = 48
TRAIN_DAYS = 2
VAL_DAYS = 1
TEST_DAYS = 1
N_DAYS = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
T_TOTAL = N_DAYS * SLOTS_PER_DAY
Z = 3

TRAIN_SLOTS = list(range(0, TRAIN_DAYS * SLOTS_PER_DAY))
VAL_SLOTS = list(
    range(TRAIN_DAYS * SLOTS_PER_DAY, (TRAIN_DAYS + VAL_DAYS) * SLOTS_PER_DAY)
)
TEST_SLOTS = list(
    range(
        (TRAIN_DAYS + VAL_DAYS) * SLOTS_PER_DAY,
        (TRAIN_DAYS + VAL_DAYS + TEST_DAYS) * SLOTS_PER_DAY,
    )
)


def _make_od() -> np.ndarray:
    """Synthetic OD: slot ``s`` has value ``s + 1`` everywhere.

    The +1 keeps every train slot strictly positive so an aggregate
    that accidentally drew a val/test slot (numerically larger) is
    easy to detect.
    """
    od = np.zeros((T_TOTAL, Z, Z), dtype=np.int32)
    for s in range(T_TOTAL):
        od[s] = s + 1
    return od


# --- split_train_slots_into_day_blocks ------------------------------------


def test_split_train_slots_into_day_blocks_cuts_into_full_days() -> None:
    blocks = split_train_slots_into_day_blocks(TRAIN_SLOTS, SLOTS_PER_DAY)
    assert len(blocks) == TRAIN_DAYS
    assert all(len(b) == SLOTS_PER_DAY for b in blocks)
    # Block 0 = slots 0..47, block 1 = slots 48..95.
    assert blocks[0] == list(range(0, SLOTS_PER_DAY))
    assert blocks[1] == list(range(SLOTS_PER_DAY, 2 * SLOTS_PER_DAY))
    # Together the blocks cover every train slot exactly once.
    flat = [s for b in blocks for s in b]
    assert flat == TRAIN_SLOTS


def test_split_partial_day_block_raises() -> None:
    bad = list(range(0, SLOTS_PER_DAY + 5))  # 53 slots: 1 full day + 5 extra
    with pytest.raises(ValueError, match="multiple of"):
        split_train_slots_into_day_blocks(bad, SLOTS_PER_DAY)


def test_split_empty_train_slots_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_train_slots_into_day_blocks([], SLOTS_PER_DAY)


def test_split_invalid_slots_per_day_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        split_train_slots_into_day_blocks(TRAIN_SLOTS, 0)


# --- ConditionalBootstrapSampler.sample() output contract -----------------


def test_sample_shape() -> None:
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od,
        TRAIN_SLOTS,
        slots_per_day=SLOTS_PER_DAY,
        n_days_per_scenario=11,
        n_omega=64,
        seed=42,
    )
    samples = sampler.sample()
    assert samples.shape == (64, Z, Z)


def test_sample_dtype_is_int32() -> None:
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od,
        TRAIN_SLOTS,
        slots_per_day=SLOTS_PER_DAY,
        n_days_per_scenario=4,
        n_omega=8,
        seed=0,
    )
    samples = sampler.sample()
    assert samples.dtype == np.int32


def test_sample_is_nonnegative() -> None:
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od,
        TRAIN_SLOTS,
        slots_per_day=SLOTS_PER_DAY,
        n_days_per_scenario=11,
        n_omega=16,
        seed=7,
    )
    samples = sampler.sample()
    assert samples.min() >= 0


# --- determinism ----------------------------------------------------------


def test_same_seed_bit_identical() -> None:
    od = _make_od()
    a = ConditionalBootstrapSampler(
        od, TRAIN_SLOTS, slots_per_day=SLOTS_PER_DAY, n_omega=8, seed=123
    ).sample()
    b = ConditionalBootstrapSampler(
        od, TRAIN_SLOTS, slots_per_day=SLOTS_PER_DAY, n_omega=8, seed=123
    ).sample()
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    od = _make_od()
    a = ConditionalBootstrapSampler(
        od, TRAIN_SLOTS, slots_per_day=SLOTS_PER_DAY, n_omega=32, seed=1
    ).sample()
    b = ConditionalBootstrapSampler(
        od, TRAIN_SLOTS, slots_per_day=SLOTS_PER_DAY, n_omega=32, seed=2
    ).sample()
    # Not "every entry differs" but "the two outputs are not identical".
    assert not np.array_equal(a, b)


# --- leakage prevention ---------------------------------------------------


def test_used_slots_subset_of_train_slots() -> None:
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od,
        TRAIN_SLOTS,
        slots_per_day=SLOTS_PER_DAY,
        n_days_per_scenario=11,
        n_omega=64,
        seed=42,
    )
    sampler.sample()
    assert sampler.used_slots is not None
    train_set = set(TRAIN_SLOTS)
    assert sampler.used_slots.issubset(train_set)
    # And explicitly no val / test slot leaks in.
    assert sampler.used_slots.isdisjoint(set(VAL_SLOTS))
    assert sampler.used_slots.isdisjoint(set(TEST_SLOTS))
    # Day-index trace: every drawn index must point into the train
    # day-block list, never beyond it.
    assert sampler.selected_day_indices is not None
    assert int(sampler.selected_day_indices.max()) < TRAIN_DAYS
    assert int(sampler.selected_day_indices.min()) >= 0


# --- scenario diversity ---------------------------------------------------


def test_scenarios_have_nonzero_variance() -> None:
    """64 scenarios drawn from 2 day blocks must not all be identical.

    With ``n_days_per_scenario=11`` and 2 train day blocks, each
    scenario's mass equals ``k1 * sum(block0) + k2 * sum(block1)`` for
    some ``(k1, k2)`` with ``k1 + k2 = 11``. The binomial draw means
    different scenarios pick different ``(k1, k2)`` and therefore
    different totals -- the per-scenario total variance must be > 0.
    """
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od,
        TRAIN_SLOTS,
        slots_per_day=SLOTS_PER_DAY,
        n_days_per_scenario=11,
        n_omega=64,
        seed=42,
    )
    samples = sampler.sample()
    per_scenario_totals = samples.reshape(samples.shape[0], -1).sum(axis=1)
    assert float(per_scenario_totals.var()) > 0.0
    # And the cell-level std across scenarios is nonzero too.
    assert float(samples.std(axis=0).max()) > 0.0


# --- aggregate_day_blocks correctness ------------------------------------


def test_aggregate_equals_sum_of_selected_blocks() -> None:
    od = _make_od()
    blocks = split_train_slots_into_day_blocks(TRAIN_SLOTS, SLOTS_PER_DAY)
    # Pick blocks [0, 1, 0] -- block 0 appears twice (with-replacement).
    selected = [blocks[0], blocks[1], blocks[0]]
    got = aggregate_day_blocks(od, selected)
    expected = np.zeros((Z, Z), dtype=np.int64)
    for block in selected:
        for s in block:
            expected += od[s].astype(np.int64)
    np.testing.assert_array_equal(got, expected.astype(np.int32))
    assert got.dtype == np.int32


def test_aggregate_empty_raises() -> None:
    od = _make_od()
    with pytest.raises(ValueError, match="empty"):
        aggregate_day_blocks(od, [])


# --- sampler-level aggregate correctness ---------------------------------


def test_sampler_aggregate_matches_selected_day_indices() -> None:
    """For each scenario, the emitted matrix equals the manual sum of
    the blocks recorded in ``selected_day_indices``."""
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od,
        TRAIN_SLOTS,
        slots_per_day=SLOTS_PER_DAY,
        n_days_per_scenario=5,
        n_omega=4,
        seed=99,
    )
    samples = sampler.sample()
    assert sampler.selected_day_indices is not None
    for i in range(sampler.n_omega):
        chosen_blocks = [sampler.day_blocks[j] for j in sampler.selected_day_indices[i]]
        expected = np.zeros((Z, Z), dtype=np.int64)
        for block in chosen_blocks:
            for s in block:
                expected += od[s].astype(np.int64)
        np.testing.assert_array_equal(samples[i], expected.astype(np.int32))


# --- error surfaces ------------------------------------------------------


def test_sampler_rejects_partial_day_block_at_construction() -> None:
    od = _make_od()
    bad = list(range(0, SLOTS_PER_DAY + 1))  # one extra slot
    with pytest.raises(ValueError, match="multiple of"):
        ConditionalBootstrapSampler(od, bad, slots_per_day=SLOTS_PER_DAY)


def test_sampler_rejects_empty_train_slots() -> None:
    od = _make_od()
    with pytest.raises(ValueError, match="empty"):
        ConditionalBootstrapSampler(od, [], slots_per_day=SLOTS_PER_DAY)


def test_sampler_rejects_unknown_mode() -> None:
    od = _make_od()
    with pytest.raises(ValueError, match="mode"):
        ConditionalBootstrapSampler(
            od, TRAIN_SLOTS, slots_per_day=SLOTS_PER_DAY, mode="poisson"
        )
    # And the supported-modes tuple is exactly the documented PR5C-2A set.
    assert VALID_MODES == ("day_block",)


def test_sampler_rejects_out_of_range_train_slots() -> None:
    od = _make_od()
    # Slot index past T-1 -- must fail before any sample read.
    bogus = list(range(T_TOTAL, T_TOTAL + SLOTS_PER_DAY))
    with pytest.raises(ValueError, match="outside"):
        ConditionalBootstrapSampler(od, bogus, slots_per_day=SLOTS_PER_DAY)


# --- summary helper ------------------------------------------------------


def test_bootstrap_summary_keys_and_shapes() -> None:
    od = _make_od()
    sampler = ConditionalBootstrapSampler(
        od, TRAIN_SLOTS, slots_per_day=SLOTS_PER_DAY, n_omega=4, seed=0
    )
    samples = sampler.sample()
    summary = compute_bootstrap_summary(samples)
    assert summary["n_omega"] == 4
    assert summary["z"] == Z
    assert summary["global_min"] >= 0
    assert summary["global_max"] >= summary["global_min"]
    assert summary["total_sum_mean"] > 0
