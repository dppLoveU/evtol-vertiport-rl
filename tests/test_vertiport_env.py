"""Unit tests for the Stage-5 PR1 VertiportEnv.

All tests use small hand-crafted synthetic data so the bilateral OD
coverage math is verifiable by hand; none depend on the real Stage-2/4
artifacts.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.envs.vertiport_env import VertiportEnv

# Hand-crafted scenario: |Z| = 4, |C| = 4.
#
# OD demand matrix M (rows = origin, cols = destination):
#     [[ 0, 10,  0,  0],
#      [ 5,  0,  0,  0],
#      [ 0,  0,  0, 20],
#      [ 0,  0,  3,  0]]
# total demand = 38.
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
#   cand 0 -> {0}
#   cand 1 -> {1}
#   cand 2 -> {2, 3}
#   cand 3 -> {0, 1}
_COV = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
        [True, True, False, False],
    ],
    dtype=bool,
)

_N_OMEGA = 8
_K_SELECT = 3


def _make_env(od: np.ndarray | None = None, k_select: int = _K_SELECT) -> VertiportEnv:
    """Build a small VertiportEnv; all omega slices share the same M."""
    if od is None:
        od = np.broadcast_to(_M, (_N_OMEGA, 4, 4)).copy()
    return VertiportEnv(od, _COV, k_select=k_select, normalize=True, seed=42)


def test_reset_returns_obs_and_info() -> None:
    env = _make_env()
    obs, info = env.reset(seed=42)

    assert set(obs) == {
        "selected_mask",
        "covered_zones",
        "remaining_budget",
        "current_coverage_ratio",
    }
    assert obs["selected_mask"].shape == (4,)
    assert obs["covered_zones"].shape == (4,)
    assert obs["remaining_budget"].shape == (1,)
    assert obs["current_coverage_ratio"].shape == (1,)
    assert not obs["selected_mask"].any()
    assert not obs["covered_zones"].any()
    assert obs["remaining_budget"][0] == _K_SELECT
    assert obs["current_coverage_ratio"][0] == 0.0

    assert info["selected_count"] == 0
    assert info["coverage_ratio"] == 0.0
    assert 0 <= info["scenario_idx"] < _N_OMEGA


def test_action_masks_shape_and_initial_state() -> None:
    env = _make_env()
    env.reset(seed=42)
    masks = env.action_masks()

    assert masks.shape == (4,)
    assert masks.dtype == bool
    assert masks.all()  # nothing selected yet


def test_step_increments_selected_count() -> None:
    env = _make_env()
    env.reset(seed=42)

    _, _, _, _, info = env.step(0)
    assert info["selected_count"] == 1
    assert info["selected_candidates"] == [0]

    _, _, _, _, info = env.step(1)
    assert info["selected_count"] == 2
    assert info["selected_candidates"] == [0, 1]


def test_repeated_action_is_masked_and_raises() -> None:
    env = _make_env()
    env.reset(seed=42)

    env.step(0)
    # Candidate 0 is now masked out.
    assert env.action_masks()[0] is np.False_ or not env.action_masks()[0]
    # invalid_action == "mask": stepping it again raises.
    with pytest.raises(ValueError):
        env.step(0)


def test_reward_is_nonnegative() -> None:
    env = _make_env()
    env.reset(seed=42)
    for action in (0, 1, 2):
        _, reward, _, _, _ = env.step(action)
        assert reward >= 0.0


def test_coverage_ratio_monotone_nondecreasing() -> None:
    env = _make_env()
    env.reset(seed=42)
    prev = 0.0
    for _ in range(_K_SELECT):
        action = int(np.flatnonzero(env.action_masks())[0])
        _, _, _, _, info = env.step(action)
        assert info["coverage_ratio"] >= prev - 1e-9
        prev = info["coverage_ratio"]


def test_terminated_after_k_steps() -> None:
    env = _make_env()
    env.reset(seed=42)
    for step_i in range(_K_SELECT):
        action = int(np.flatnonzero(env.action_masks())[0])
        _, _, terminated, truncated, _ = env.step(action)
        assert truncated is False
        if step_i < _K_SELECT - 1:
            assert terminated is False
        else:
            assert terminated is True


def test_bilateral_coverage_handcomputed() -> None:
    env = _make_env()
    env.reset(seed=42)

    # Step 1: place cand 0 -> covers zone {0}. Bilateral covered demand =
    # M[0, 0] = 0. gained = 0, reward = 0.
    _, reward, _, _, info = env.step(0)
    assert info["total_covered_demand"] == 0
    assert reward == pytest.approx(0.0)
    assert info["incremental_gain"] == pytest.approx(0.0)

    # Step 2: place cand 1 -> covered zones {0, 1}. Bilateral covered
    # demand = M[0,0]+M[0,1]+M[1,0]+M[1,1] = 0+10+5+0 = 15.
    _, reward, _, _, info = env.step(1)
    assert info["total_covered_demand"] == 15
    assert info["incremental_gain"] == pytest.approx(15.0)
    assert reward == pytest.approx(15.0 / _TOTAL)
    assert info["coverage_ratio"] == pytest.approx(15.0 / _TOTAL)

    # Step 3: place cand 2 -> covered zones {0, 1, 2, 3}. Bilateral
    # covered demand = full total = 38. gained = 38 - 15 = 23.
    _, reward, terminated, _, info = env.step(2)
    assert info["total_covered_demand"] == _TOTAL
    assert info["incremental_gain"] == pytest.approx(23.0)
    assert reward == pytest.approx(23.0 / _TOTAL)
    assert info["coverage_ratio"] == pytest.approx(1.0)
    assert terminated is True


def test_seed_reproduces_scenario_idx() -> None:
    env_a = _make_env()
    env_b = _make_env()
    _, info_a = env_a.reset(seed=123)
    _, info_b = env_b.reset(seed=123)
    assert info_a["scenario_idx"] == info_b["scenario_idx"]

    # Re-seeding the same env reproduces the index too.
    _, info_a2 = env_a.reset(seed=123)
    assert info_a2["scenario_idx"] == info_a["scenario_idx"]


def test_zero_total_demand_does_not_crash() -> None:
    od_zero = np.zeros((_N_OMEGA, 4, 4), dtype=np.int32)
    env = VertiportEnv(od_zero, _COV, k_select=_K_SELECT, normalize=True, seed=42)
    obs, info = env.reset(seed=42)
    assert info["coverage_ratio"] == 0.0

    for _ in range(_K_SELECT):
        action = int(np.flatnonzero(env.action_masks())[0])
        obs, reward, terminated, _, info = env.step(action)
        assert reward == 0.0
        assert info["coverage_ratio"] == 0.0
        assert np.isfinite(obs["current_coverage_ratio"][0])
    assert terminated is True


# -- Gymnasium API conformance (PR2) ----------------------------------


def test_is_gymnasium_env() -> None:
    import gymnasium as gym

    env = _make_env()
    assert isinstance(env, gym.Env)


def test_action_space_contains_valid_action() -> None:
    env = _make_env()
    env.reset(seed=42)
    valid = int(np.flatnonzero(env.action_masks())[0])
    assert env.action_space.contains(valid)
    assert env.action_space.n == 4


def test_observation_space_contains_obs_after_reset() -> None:
    env = _make_env()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)


def test_observation_space_contains_obs_after_step() -> None:
    env = _make_env()
    env.reset(seed=42)
    for _ in range(_K_SELECT):
        action = int(np.flatnonzero(env.action_masks())[0])
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)


def test_action_masks_dtype_and_shape() -> None:
    env = _make_env()
    env.reset(seed=42)
    masks = env.action_masks()
    assert masks.dtype == bool
    assert masks.shape == (env.n_candidates,)


def test_step_returns_gymnasium_five_tuple() -> None:
    env = _make_env()
    env.reset(seed=42)
    result = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# -- Demand-aware observation (PR2) -----------------------------------


def _make_env_demand(
    od: np.ndarray | None = None, k_select: int = _K_SELECT
) -> VertiportEnv:
    """Build a VertiportEnv with the per-scenario demand features on."""
    if od is None:
        od = np.broadcast_to(_M, (_N_OMEGA, 4, 4)).copy()
    return VertiportEnv(
        od,
        _COV,
        k_select=k_select,
        normalize=True,
        seed=42,
        include_demand_features=True,
    )


def test_demand_features_in_observation_space_after_reset() -> None:
    env = _make_env_demand()
    obs, _ = env.reset(seed=42)
    assert "demand_features" in obs
    assert env.observation_space.contains(obs)


def test_demand_features_shape_and_dtype() -> None:
    env = _make_env_demand()
    obs, _ = env.reset(seed=42)
    feats = obs["demand_features"]
    assert feats.shape == (4, 4)  # (n_zones, 4)
    assert feats.dtype == np.float32


def test_demand_features_values_in_unit_range() -> None:
    env = _make_env_demand()
    obs, _ = env.reset(seed=42)
    feats = obs["demand_features"]
    assert feats.min() >= 0.0
    assert feats.max() <= 1.0 + 1e-6


def test_demand_features_covered_indicator_initially_zero() -> None:
    env = _make_env_demand()
    obs, _ = env.reset(seed=42)
    # Column 3 is the covered-zone indicator: nothing covered yet.
    assert np.array_equal(obs["demand_features"][:, 3], np.zeros(4, dtype=np.float32))


def test_demand_features_covered_indicator_updates_on_step() -> None:
    env = _make_env_demand()
    env.reset(seed=42)
    # cand 0 covers zone {0}; after the step zone 0's indicator is 1.
    obs, _, _, _, _ = env.step(0)
    indicator = obs["demand_features"][:, 3]
    assert indicator[0] == 1.0
    assert np.array_equal(indicator, env._covered_zones.astype(np.float32))
    # The first three columns are scenario-fixed: unchanged across steps.
    obs2, _, _, _, _ = env.step(1)
    assert np.array_equal(
        obs["demand_features"][:, :3], obs2["demand_features"][:, :3]
    )


def test_demand_features_off_keeps_legacy_observation() -> None:
    env = _make_env()  # include_demand_features defaults to False
    obs, _ = env.reset(seed=42)
    assert "demand_features" not in obs
    assert set(obs) == {
        "selected_mask",
        "covered_zones",
        "remaining_budget",
        "current_coverage_ratio",
    }
    assert env.observation_space.contains(obs)


def test_demand_features_differ_across_scenarios() -> None:
    # Give scenario 3 a transposed (structurally different) OD matrix so
    # its normalized origin/destination columns differ from the rest.
    od = np.broadcast_to(_M, (_N_OMEGA, 4, 4)).copy()
    od[3] = _M.T
    env = VertiportEnv(
        od, _COV, k_select=_K_SELECT, normalize=True, seed=42,
        include_demand_features=True,
    )
    feats: dict[int, np.ndarray] = {}
    for s in range(200):
        obs, info = env.reset(seed=s)
        feats[info["scenario_idx"]] = obs["demand_features"].copy()
    assert 3 in feats and len(feats) >= 2
    others = [k for k in feats if k != 3]
    assert any(not np.allclose(feats[3], feats[k]) for k in others)
